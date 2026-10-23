import type { ProgressCallback } from '../core/index.ts'
import { TRANSLATE_CACHE_NAME } from '../core/cache.ts'
import { PAIRS, RUNTIME_CDN, type PairEntry } from './registry.ts'

/**
 * Default weights location: the runonweb mirror of Mozilla's Firefox Translations models on the
 * Hugging Face Hub (MPL-2.0). Layout: `<modelPath><from>-<to>/<file>?rev=<revision>`.
 * Self-host by pointing `modelPath` at a folder produced by `scripts/translate-models.mjs`.
 */
export const DEFAULT_MODEL_PATH = 'https://huggingface.co/midudev/firefox-translations/resolve/main/'

export type BergamotOptions = {
  /** Base URL of the model files. Default: Hugging Face mirror. */
  modelPath?: string
  /** Base URL of `bergamot-translator-worker.{js,wasm}`. Default: jsDelivr. */
  runtimePath?: string
  onProgress?: ProgressCallback
}

export type Route = PairEntry[]

function normalize(code: string): string {
  const c = code.trim()
  if (/^zh[-_]hans$/i.test(c) || /^zh[-_]cn$/i.test(c)) return 'zh'
  if (/^zh[-_]hant$/i.test(c) || /^zh[-_]tw$/i.test(c)) return 'zh-Hant'
  return c.toLowerCase().split(/[-_]/)[0] ?? c
}

/** Resolve a language pair to one direct model, or two models pivoting through English. */
export function resolveRoute(from: string, to: string): Route | null {
  const src = normalize(from)
  const tgt = normalize(to)
  if (src === tgt) return null
  const direct = PAIRS[`${src}-${tgt}`]
  if (direct) return [direct]
  const outbound = PAIRS[`${src}-en`]
  const inbound = PAIRS[`en-${tgt}`]
  if (outbound && inbound) return [outbound, inbound]
  return null
}

/** True when `from → to` can be served, directly or via English. */
export function hasPair(from: string, to: string): boolean {
  return resolveRoute(from, to) !== null
}

function keyOf(entry: PairEntry): string {
  return `${entry.from}-${entry.to}`
}

/** Relative paths are resolved against the page, since the Blob worker has no usable base URL. */
function absolute(base: string): string {
  const withSlash = base.endsWith('/') ? base : `${base}/`
  if (/^[a-z]+:/i.test(withSlash)) return withSlash
  const origin = typeof location !== 'undefined' ? location.href : 'http://localhost/'
  return new URL(withSlash, origin).href
}

function joinUrl(base: string, path: string): string {
  return base + path
}

async function openCache(): Promise<Cache | null> {
  try {
    if (typeof caches === 'undefined') return null
    return await caches.open(TRANSLATE_CACHE_NAME)
  } catch {
    return null
  }
}

async function readWithProgress(
  res: Response,
  onChunk: (loaded: number, total: number) => void
): Promise<ArrayBuffer> {
  const total = Number(res.headers.get('content-length') ?? 0)
  if (!res.body) return res.arrayBuffer()
  const reader = res.body.getReader()
  if (total > 0) {
    let out = new Uint8Array(total)
    let loaded = 0
    let lastEmit = 0
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      if (loaded + value.byteLength > out.byteLength) {
        const grown = new Uint8Array(Math.max(out.byteLength * 2, loaded + value.byteLength))
        grown.set(out.subarray(0, loaded))
        out = grown
      }
      out.set(value, loaded)
      loaded += value.byteLength
      const now = performance.now()
      if (loaded >= total || now - lastEmit >= 32) {
        lastEmit = now
        onChunk(loaded, total)
      }
    }
    const finalBytes = loaded === out.byteLength ? out : out.subarray(0, loaded)
    return finalBytes.byteOffset === 0 && finalBytes.byteLength === finalBytes.buffer.byteLength
      ? (finalBytes.buffer as ArrayBuffer)
      : finalBytes.slice().buffer
  }

  const chunks: Uint8Array[] = []
  let loaded = 0
  let lastEmit = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    loaded += value.byteLength
    const now = performance.now()
    if (now - lastEmit >= 32) {
      lastEmit = now
      onChunk(loaded, total)
    }
  }
  onChunk(loaded, total)
  const out = new Uint8Array(loaded)
  let offset = 0
  for (const c of chunks) {
    out.set(c, offset)
    offset += c.byteLength
  }
  return out.buffer
}

/**
 * Download one model file, through Cache Storage so the second load is instant and offline.
 */
async function fetchFile(
  url: string,
  file: string,
  onProgress: ProgressCallback | undefined,
  cachePromise: Promise<Cache | null>
): Promise<ArrayBuffer> {
  const cache = await cachePromise
  const hit = await cache?.match(url)
  if (hit) {
    onProgress?.({ status: 'progress', file, progress: 100 })
    return hit.arrayBuffer()
  }

  const res = await fetch(url, { credentials: 'omit' })
  if (!res.ok) throw new Error(`Could not download ${file} (HTTP ${res.status}) from ${url}`)

  const buffer = await readWithProgress(res, (loaded, total) => {
    onProgress?.({
      status: 'progress',
      file,
      progress: total > 0 ? (loaded / total) * 100 : undefined,
      loaded,
      total: total > 0 ? total : undefined,
    })
  })

  if (cache) {
    void cache
      .put(
        url,
        new Response(buffer, {
          headers: { 'Content-Type': 'application/octet-stream', 'Content-Length': String(buffer.byteLength) },
        })
      )
      .catch(() => {
        // Quota exceeded or private mode: keep going without a cache.
      })
  }
  return buffer
}

type Pending = { resolve: (v: unknown) => void; reject: (e: Error) => void }

/**
 * Bergamot (Marian NMT compiled to WASM) running in a Web Worker, fed with Firefox Translations
 * models. One instance can hold several language pairs.
 */
export class BergamotEngine {
  #modelPath: string
  #runtimePath: string
  #onProgress?: ProgressCallback
  #worker: Worker | null = null
  #workerUrl: string | null = null
  #ready: Promise<void> | null = null
  #pending = new Map<number, Pending>()
  #nextId = 1
  #loaded = new Map<string, Promise<void>>()

  constructor(options: BergamotOptions = {}) {
    this.#modelPath = absolute(options.modelPath ?? DEFAULT_MODEL_PATH)
    this.#runtimePath = absolute(options.runtimePath ?? RUNTIME_CDN)
    this.#onProgress = options.onProgress
  }

  get modelPath(): string {
    return this.#modelPath
  }

  #call<T>(name: string, args?: Record<string, unknown>, transfer: Transferable[] = []): Promise<T> {
    const worker = this.#worker
    if (!worker) return Promise.reject(new Error('Translator has been disposed'))
    const id = this.#nextId++
    return new Promise<T>((resolve, reject) => {
      this.#pending.set(id, { resolve: resolve as (v: unknown) => void, reject })
      worker.postMessage({ id, name, args }, transfer)
    })
  }

  /** Boot the worker and instantiate the WASM runtime. Idempotent. */
  init(): Promise<void> {
    if (this.#ready) return this.#ready
    if (typeof Worker === 'undefined' || typeof Blob === 'undefined') {
      return Promise.reject(new Error('runonweb/translate needs a browser with Web Workers'))
    }

    this.#onProgress?.({ status: 'loading', progress: 0 })
    this.#ready = (async () => {
      const { WORKER_SOURCE } = await import('./worker.ts')
      this.#workerUrl = URL.createObjectURL(new Blob([WORKER_SOURCE], { type: 'text/javascript' }))
      const worker = new Worker(this.#workerUrl)
      this.#worker = worker

      worker.addEventListener('message', (event: MessageEvent) => {
        const { id, result, error } = event.data as {
          id: number
          result?: unknown
          error?: { name?: string; message?: string; stack?: string }
        }
        const pending = this.#pending.get(id)
        if (!pending) return
        this.#pending.delete(id)
        if (error) {
          const err = new Error(error.message ?? 'Worker error')
          if (error.name) err.name = error.name
          if (error.stack) err.stack = error.stack
          pending.reject(err)
        } else {
          pending.resolve(result)
        }
      })
      worker.addEventListener('error', (event: ErrorEvent) => {
        const err = new Error(event.message || 'Translation worker crashed')
        for (const p of this.#pending.values()) p.reject(err)
        this.#pending.clear()
      })

      await this.#call<boolean>('init', { runtimePath: this.#runtimePath })
    })().catch((err) => {
      this.#ready = null
      throw err
    })
    return this.#ready
  }

  #fileUrl(entry: PairEntry, name: string): string {
    return joinUrl(this.#modelPath, `${keyOf(entry)}/${name}?rev=${encodeURIComponent(entry.revision)}`)
  }

  async #loadEntry(entry: PairEntry): Promise<void> {
    const key = keyOf(entry)
    const existing = this.#loaded.get(key)
    if (existing) return existing

    const promise = (async () => {
      const f = entry.files
      const vocabNames = f.vocab ? [f.vocab] : [f.srcvocab!, f.trgvocab!]
      const cachePromise = openCache()
      const [, [model, lex, ...vocabs]] = await Promise.all([
        this.init(),
        Promise.all(
          [f.model, f.lex, ...vocabNames].map((name) =>
            fetchFile(this.#fileUrl(entry, name), name, this.#onProgress, cachePromise)
          )
        ),
      ])
      this.#onProgress?.({ status: 'loading', file: key })
      await this.#call<boolean>('loadModel', { key, modelName: f.model, model, lex, vocabs }, [model, lex, ...vocabs])
    })()

    this.#loaded.set(key, promise)
    promise.catch(() => this.#loaded.delete(key))
    return promise
  }

  /** Download (or read from cache) and instantiate every model needed for `from → to`. */
  async load(from: string, to: string): Promise<Route> {
    const route = resolveRoute(from, to)
    if (!route) {
      throw new Error(
        `No Firefox Translations model for ${from} → ${to}. See PAIRS in 'runonweb/translate' for the supported pairs.`
      )
    }
    await Promise.all(route.map((entry) => this.#loadEntry(entry)))
    this.#onProgress?.({ status: 'ready', progress: 100 })
    return route
  }

  /** Translate one or more texts. Models are loaded on demand. */
  async translate(from: string, to: string, texts: string[], options: { html?: boolean } = {}): Promise<string[]> {
    const route = await this.load(from, to)
    return this.#call<string[]>('translate', { route: route.map(keyOf), texts, html: options.html ?? false })
  }

  /** Free a pair from WASM memory. */
  async unload(from: string, to: string): Promise<void> {
    const route = resolveRoute(from, to) ?? []
    for (const entry of route) {
      const key = keyOf(entry)
      if (!this.#loaded.has(key)) continue
      this.#loaded.delete(key)
      await this.#call('freeModel', { key })
    }
  }

  dispose(): void {
    const worker = this.#worker
    this.#worker = null
    this.#ready = null
    this.#loaded.clear()
    const err = new Error('Translator disposed')
    for (const p of this.#pending.values()) p.reject(err)
    this.#pending.clear()
    worker?.terminate()
    if (this.#workerUrl) URL.revokeObjectURL(this.#workerUrl)
    this.#workerUrl = null
  }
}
