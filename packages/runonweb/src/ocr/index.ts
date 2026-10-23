import type { Device, ProgressCallback, ProgressInfo, ResolvedDevice } from '../core/index.ts'
import { OCR_CACHE_NAME } from '../core/cache.ts'
import { resolveDevice, takeOrtTurn } from '../core/device.ts'
import type { PaddleOcrService } from 'paddleocr'
import { DEFAULT_OCR_SIZE, type OCRSize } from './sizes.ts'

export { DEFAULT_OCR_SIZE, OCR_SIZES, type OCRSize } from './sizes.ts'

const HF = {
  tiny: {
    det: 'https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_onnx/resolve/main/inference.onnx',
    rec: 'https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_rec_onnx/resolve/main/inference.onnx',
    yml: 'https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_rec_onnx/resolve/main/inference.yml',
  },
  small: {
    det: 'https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_onnx/resolve/main/inference.onnx',
    rec: 'https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx/resolve/main/inference.onnx',
    yml: 'https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx/resolve/main/inference.yml',
  },
  medium: {
    det: 'https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det_onnx/resolve/main/inference.onnx',
    rec: 'https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx/resolve/main/inference.onnx',
    yml: 'https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx/resolve/main/inference.yml',
  },
} as const

const PRESET = {
  tiny: 'PP-OCRv6_tiny',
  small: 'PP-OCRv6_small',
  medium: 'PP-OCRv6_medium',
} as const

const CACHE = OCR_CACHE_NAME
const ORT_WASM = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/'

export type OCROptions = {
  /** Model tier. Defaults to `small` (best size/quality). */
  size?: OCRSize
  /** Inference device. Defaults to `auto` (WebGPU when available, else WASM). */
  device?: Device
  onProgress?: ProgressCallback
}

export type OCRImageInput = Blob | File | string | HTMLImageElement | ImageData | HTMLCanvasElement

export type OCRBox = {
  xmin: number
  ymin: number
  xmax: number
  ymax: number
}

export type OCRLine = {
  text: string
  score: number
  box: OCRBox
}

export type OCRResult = {
  text: string
  lines: OCRLine[]
  confidence: number
}

/**
 * On-device OCR with PP-OCRv6 (PaddleOCR, Apache-2.0).
 *
 * @example
 * ```ts
 * import { OCR } from 'runonweb/ocr'
 *
 * const ocr = new OCR() // size: 'small'
 * await ocr.load()
 * const { text, lines } = await ocr.read(imageFile)
 * ```
 */
export class OCR {
  #size: OCRSize
  #device: Device
  #onProgress?: ProgressCallback
  #service: PaddleOcrService | null = null
  #loading: Promise<void> | null = null
  #pending: Promise<void> = Promise.resolve()
  #resolvedDevice: ResolvedDevice | null = null
  #disposed = false
  #disposeDone: Promise<void> | null = null

  constructor(options: OCROptions = {}) {
    this.#size = options.size ?? DEFAULT_OCR_SIZE
    this.#device = options.device ?? 'auto'
    this.#onProgress = options.onProgress
  }

  get size(): OCRSize {
    return this.#size
  }

  get device(): ResolvedDevice | null {
    return this.#resolvedDevice
  }

  async load(): Promise<void> {
    if (this.#disposed) throw new Error('OCR model was disposed')
    if (this.#service) return
    if (this.#loading) return this.#loading

    this.#loading = (async () => {
      const urls = HF[this.#size]
      this.#onProgress?.({ status: 'loading', progress: 0 })

      const cachePromise = openOcrCache()
      const totals = new OcrProgressTotals(this.#onProgress)

      const [det, rec, yml, ort, preferred, { PaddleOcrService: PaddleOcrServiceCtor }] = await Promise.all([
        fetchCached(urls.det, 'det.onnx', totals, cachePromise),
        fetchCached(urls.rec, 'rec.onnx', totals, cachePromise),
        fetchCachedText(urls.yml, cachePromise),
        loadOrt(),
        resolveDevice(this.#device),
        import('paddleocr'),
      ])

      const charactersDictionary = parseCharacterDict(yml)
      if (charactersDictionary[charactersDictionary.length - 1] !== ' ') {
        charactersDictionary.push(' ')
      }

      const service = await createService({
        ort,
        PaddleOcrServiceCtor,
        size: this.#size,
        det,
        rec,
        charactersDictionary,
        device: preferred,
      })

      if (this.#disposed) {
        await takeOrtTurn(() => service.ocr.destroy())
        return
      }

      this.#service = service.ocr
      this.#resolvedDevice = service.device
      this.#onProgress?.({ status: 'ready', progress: 100 })
    })()

    try {
      await this.#loading
    } finally {
      this.#loading = null
    }
  }

  read(image: OCRImageInput): Promise<OCRResult> {
    if (this.#disposed) return Promise.reject(new Error('OCR model was disposed'))
    const job = this.#read(image)
    this.#pending = this.#pending.then(
      () => job.then(
        () => undefined,
        () => undefined,
      ),
    )
    return job
  }

  /**
   * Drop the sessions. Waits for an in-flight `read` so `destroy` cannot
   * release a session that is still inside OrtRun.
   */
  async dispose(): Promise<void> {
    if (this.#disposeDone) return this.#disposeDone
    this.#disposed = true
    this.#disposeDone = (async () => {
      await this.#pending
      await this.#loading
      const service = this.#service
      this.#service = null
      this.#resolvedDevice = null
      if (!service) return
      await takeOrtTurn(() => service.destroy())
    })()
    return this.#disposeDone
  }

  async #read(image: OCRImageInput): Promise<OCRResult> {
    if (this.#disposed) throw new Error('OCR model was disposed')
    await this.load()
    if (this.#disposed) throw new Error('OCR model was disposed')
    const service = this.#service
    if (!service) throw new Error('OCR model failed to load')

    this.#onProgress?.({ status: 'recognizing' })
    const pixels = await imageToPixels(image)
    const raw = await takeOrtTurn(() => {
      if (this.#disposed) throw new Error('OCR model was disposed')
      return service.recognize(pixels)
    })
    const processed = service.processRecognition(raw)

    const lines: OCRLine[] = raw.map((item) => ({
      text: item.text,
      score: item.confidence,
      box: {
        xmin: item.box.x,
        ymin: item.box.y,
        xmax: item.box.x + item.box.width,
        ymax: item.box.y + item.box.height,
      },
    }))

    this.#onProgress?.({ status: 'done' })
    return {
      text: processed.text,
      lines,
      confidence: processed.confidence,
    }
  }
}

export async function read(image: OCRImageInput, options?: OCROptions): Promise<OCRResult> {
  const ocr = new OCR(options)
  try {
    return await ocr.read(image)
  } finally {
    await ocr.dispose()
  }
}

export type { ProgressInfo, Device }

type OrtApi = {
  env: { wasm: { wasmPaths: string; numThreads: number } }
  Tensor: unknown
  InferenceSession: {
    create: (model: ArrayBuffer, options?: { executionProviders?: string[] }) => Promise<unknown>
  }
}

let ortPromise: Promise<OrtApi> | null = null

function loadOrt(): Promise<OrtApi> {
  if (ortPromise) return ortPromise
  ortPromise = (async () => {
    const ort = (await import('onnxruntime-web')) as unknown as OrtApi
    ort.env.wasm.wasmPaths = ORT_WASM
    ort.env.wasm.numThreads = 1
    return ort
  })()
  return ortPromise
}

async function createService(opts: {
  ort: OrtApi
  PaddleOcrServiceCtor: typeof PaddleOcrService
  size: OCRSize
  det: ArrayBuffer
  rec: ArrayBuffer
  charactersDictionary: string[]
  device: ResolvedDevice
}): Promise<{ ocr: PaddleOcrService; device: ResolvedDevice }> {
  const tryDevice = async (device: ResolvedDevice) => {
    const providers = device === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm']
    const wrapped = {
      Tensor: opts.ort.Tensor,
      InferenceSession: {
        create: (modelBuffer: ArrayBuffer) =>
          opts.ort.InferenceSession.create(modelBuffer, { executionProviders: providers }),
      },
    }
    // Only clone buffers when trying WebGPU first, so fallback to WASM still has intact buffers.
    const needsFallbackCopy = device === 'webgpu'
    return opts.PaddleOcrServiceCtor.createInstance({
      ort: wrapped as never,
      modelPreset: PRESET[opts.size],
      detection: { modelBuffer: needsFallbackCopy ? opts.det.slice(0) : opts.det },
      recognition: {
        modelBuffer: needsFallbackCopy ? opts.rec.slice(0) : opts.rec,
        charactersDictionary: opts.charactersDictionary,
      },
    })
  }

  try {
    return { ocr: await tryDevice(opts.device), device: opts.device }
  } catch (err) {
    if (opts.device === 'wasm') throw err
    return { ocr: await tryDevice('wasm'), device: 'wasm' }
  }
}

class OcrProgressTotals {
  #loaded = new Map<string, number>()
  #total = new Map<string, number>()
  #cb?: ProgressCallback
  #lastEmit = 0

  constructor(cb?: ProgressCallback) {
    this.#cb = cb
  }

  update(file: string, loaded: number, total: number, force = false): void {
    this.#loaded.set(file, loaded)
    if (total > 0) this.#total.set(file, total)
    if (!this.#cb) return
    const now = performance.now()
    if (!force && loaded < total && now - this.#lastEmit < 32) return
    this.#lastEmit = now
    let sumLoaded = 0
    let sumTotal = 0
    for (const v of this.#loaded.values()) sumLoaded += v
    for (const v of this.#total.values()) sumTotal += v
    const progress = sumTotal > 0 ? (sumLoaded / sumTotal) * 100 : undefined
    this.#cb({ status: 'progress', file, loaded: sumLoaded, total: sumTotal, progress })
  }
}

function openOcrCache(): Promise<Cache | null> {
  if (typeof caches === 'undefined') return Promise.resolve(null)
  return caches.open(CACHE).catch(() => null)
}

async function fetchCached(
  url: string,
  file: string,
  totals: OcrProgressTotals,
  cachePromise: Promise<Cache | null>
): Promise<ArrayBuffer> {
  const cache = await cachePromise
  const hit = await cache?.match(url)
  if (hit) {
    const buf = await hit.arrayBuffer()
    totals.update(file, buf.byteLength, buf.byteLength, true)
    return buf
  }

  const res = await fetch(url)
  if (!res.ok) throw new Error(`Could not load ${file} (${res.status})`)

  const total = Number(res.headers.get('content-length') ?? 0)
  if (!res.body || !total) {
    const buf = await res.arrayBuffer()
    totals.update(file, buf.byteLength, buf.byteLength, true)
    void cache?.put(
      url,
      new Response(buf, { headers: { 'content-length': String(buf.byteLength) } })
    )
    return buf
  }

  totals.update(file, 0, total, true)
  const reader = res.body.getReader()
  let out = new Uint8Array(total)
  let received = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    if (received + value.byteLength > out.byteLength) {
      const grown = new Uint8Array(Math.max(out.byteLength * 2, received + value.byteLength))
      grown.set(out.subarray(0, received))
      out = grown
    }
    out.set(value, received)
    received += value.byteLength
    totals.update(file, received, total, received >= total)
  }

  const finalBytes = received === out.byteLength ? out : out.subarray(0, received)
  const buf =
    finalBytes.byteOffset === 0 && finalBytes.byteLength === finalBytes.buffer.byteLength
      ? (finalBytes.buffer as ArrayBuffer)
      : finalBytes.slice().buffer
  void cache?.put(
    url,
    new Response(buf, { headers: { 'content-length': String(buf.byteLength) } })
  )
  return buf
}

async function fetchCachedText(url: string, cachePromise: Promise<Cache | null>): Promise<string> {
  const cache = await cachePromise
  const hit = await cache?.match(url)
  if (hit) return hit.text()
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Could not load dictionary (${res.status})`)
  const text = await res.text()
  void cache?.put(url, new Response(text))
  return text
}

function parseCharacterDict(yml: string): string[] {
  const start = yml.indexOf('character_dict:')
  if (start < 0) throw new Error('PP-OCRv6 dictionary missing character_dict')
  const chars: string[] = []
  for (const rawLine of yml.slice(start).split('\n').slice(1)) {
    const item = rawLine.match(/^\s*-\s+(.*)$/)
    if (!item) {
      if (chars.length && /^\S/.test(rawLine)) break
      continue
    }
    chars.push(unquoteYaml(item[1] ?? ''))
  }
  if (chars.length < 100) throw new Error('PP-OCRv6 dictionary looks empty')
  return chars
}

function unquoteYaml(value: string): string {
  const v = value.trim()
  if (v === "''" || v === '""') return ''
  if ((v.startsWith("'") && v.endsWith("'")) || (v.startsWith('"') && v.endsWith('"'))) {
    const inner = v.slice(1, -1)
    if (v.startsWith("'")) return inner.replace(/''/g, "'")
    return inner.replace(/\\"/g, '"')
  }
  return v
}

async function imageToPixels(image: OCRImageInput): Promise<{ width: number; height: number; data: Uint8Array }> {
  if (image instanceof ImageData) {
    return {
      width: image.width,
      height: image.height,
      data: new Uint8Array(image.data.buffer, image.data.byteOffset, image.data.byteLength),
    }
  }

  let source: CanvasImageSource
  if (typeof image === 'string') {
    source = await loadHtmlImage(image)
  } else if (image instanceof HTMLImageElement) {
    source = image
  } else if (image instanceof HTMLCanvasElement) {
    source = image
  } else {
    source = await createImageBitmap(image)
  }

  const width = 'naturalWidth' in source ? source.naturalWidth || source.width : source.width
  const height = 'naturalHeight' in source ? source.naturalHeight || source.height : source.height
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('Could not get 2d context')
  ctx.drawImage(source, 0, 0)
  const { data } = ctx.getImageData(0, 0, width, height)
  if (typeof ImageBitmap !== 'undefined' && source instanceof ImageBitmap) source.close()
  return {
    width,
    height,
    data: new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
  }
}

function loadHtmlImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error(`Could not load image: ${src}`))
    img.src = src
  })
}
