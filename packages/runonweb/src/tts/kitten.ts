import type { ProgressCallback } from '../core/index.ts'
import { KITTEN_CACHE_NAME, takeOrtTurn } from '../core/index.ts'
import { splitUtterances } from './split.ts'
import { toFloat32 } from './wav.ts'
import type { SpeakChunk } from './types.ts'

const HF = 'https://huggingface.co/KittenML/kitten-tts-nano-0.8-int8/resolve/main'
const CACHE = KITTEN_CACHE_NAME
const ORT_WASM = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/'
const SAMPLE_RATE = 24_000
const AUDIO_TRIM = 5_000

const VOICE_ALIASES: Record<string, string> = {
  bella: 'expr-voice-2-f',
  jasper: 'expr-voice-2-m',
  luna: 'expr-voice-3-f',
  bruno: 'expr-voice-3-m',
  rosie: 'expr-voice-4-f',
  hugo: 'expr-voice-4-m',
  kiki: 'expr-voice-5-f',
  leo: 'expr-voice-5-m',
}

const SPEED_PRIORS: Record<string, number> = {
  'expr-voice-2-f': 0.8,
  'expr-voice-2-m': 0.8,
  'expr-voice-3-m': 0.8,
  'expr-voice-3-f': 0.8,
  'expr-voice-4-m': 0.9,
  'expr-voice-4-f': 0.8,
  'expr-voice-5-m': 0.8,
  'expr-voice-5-f': 0.8,
}

const PAD = '$'
const PUNCTUATION = ';:,.!?¡¿—…"«»"" '
const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
const LETTERS_IPA =
  "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘''ᵻ"
const SYMBOLS = [PAD, ...PUNCTUATION, ...LETTERS, ...LETTERS_IPA]
const SYMBOL_TO_ID = new Map(SYMBOLS.map((s, i) => [s, i]))

type OrtApi = {
  env: { wasm: { wasmPaths: string; numThreads: number; simd?: boolean } }
  Tensor: new (type: string, data: Float32Array | BigInt64Array, dims: number[]) => unknown
  InferenceSession: {
    create: (
      model: ArrayBuffer,
      options?: { executionProviders?: string[] }
    ) => Promise<{ run: (feeds: Record<string, unknown>) => Promise<Record<string, { data: Float32Array }>> }>
  }
}

type VoiceEntry = { data: Float32Array; shape: number[] }

export type LoadedKitten = {
  session: Awaited<ReturnType<OrtApi['InferenceSession']['create']>>
  voices: Record<string, VoiceEntry>
  ort: OrtApi
}

export async function loadKitten(onProgress?: ProgressCallback): Promise<LoadedKitten> {
  onProgress?.({ status: 'loading', progress: 0 })
  const cachePromise = openKittenCache()
  const [model, voices, ort] = await Promise.all([
    fetchCached(`${HF}/kitten_tts_nano_v0_8.onnx`, 'kitten.onnx', onProgress, cachePromise),
    fetchCached(`${HF}/voices.npz`, 'voices.npz', onProgress, cachePromise).then(loadNpz),
    loadOrt(),
  ])
  const session = await ort.InferenceSession.create(model, { executionProviders: ['wasm'] })
  onProgress?.({ status: 'ready', progress: 100 })
  return { session, voices, ort }
}

export async function* streamKitten(
  loaded: LoadedKitten,
  text: string,
  voice: string,
  speed: number
): AsyncGenerator<SpeakChunk> {
  const utterances = splitUtterances(text)
  if (utterances.length === 0) return

  let nextPhonemesPromise: Promise<string> | null = phonemizeEnglish(ensurePunctuation(utterances[0]!))

  for (let i = 0; i < utterances.length; i++) {
    const piece = utterances[i]!
    const currentPhonemesPromise = nextPhonemesPromise ?? phonemizeEnglish(ensurePunctuation(piece))
    nextPhonemesPromise =
      i + 1 < utterances.length ? phonemizeEnglish(ensurePunctuation(utterances[i + 1]!)) : null

    const phonemes = await currentPhonemesPromise
    const audio = await inferKitten(loaded, phonemes, voice, speed)
    yield { audio, samplingRate: SAMPLE_RATE, text: piece }
  }
}

export function disposeKitten(loaded: LoadedKitten | null) {
  const session = loaded?.session as { release?: () => Promise<void> } | undefined
  void session?.release?.()
}

async function inferKitten(
  loaded: LoadedKitten,
  phonemes: string,
  voiceId: string,
  speed: number
): Promise<Float32Array> {
  const key = VOICE_ALIASES[voiceId] ?? voiceId
  const voice = loaded.voices[key]
  if (!voice) throw new Error(`Unknown Kitten voice: ${voiceId}`)

  const tokenIds = cleanPhonemes(phonemes)
  const [numStyles, styleDim] = voice.shape
  const refId = Math.min(tokenIds.length, (numStyles ?? 1) - 1)
  const dim = styleDim ?? 256
  const style = voice.data.subarray(refId * dim, (refId + 1) * dim)
  const scaled = speed * (SPEED_PRIORS[key] ?? 1)

  const inputIds = new BigInt64Array(tokenIds.length)
  for (let i = 0; i < tokenIds.length; i++) inputIds[i] = BigInt(tokenIds[i]!)

  const feeds = {
    input_ids: new loaded.ort.Tensor('int64', inputIds, [1, tokenIds.length]),
    style: new loaded.ort.Tensor('float32', style, [1, dim]),
    speed: new loaded.ort.Tensor('float32', new Float32Array([scaled]), [1]),
  }
  const results = await takeOrtTurn(() => loaded.session.run(feeds))
  const output = results[Object.keys(results)[0] ?? '']
  const data = toFloat32(output?.data ?? [])
  return data.subarray(0, Math.max(0, data.length - AUDIO_TRIM))
}

function cleanPhonemes(phonemes: string): number[] {
  const ids: number[] = []
  for (const ch of phonemes) {
    const id = SYMBOL_TO_ID.get(ch)
    if (id !== undefined) ids.push(id)
  }
  return [0, ...ids, 10, 0]
}

function ensurePunctuation(text: string): string {
  const t = text.trim()
  if (!t) return t
  return /[.!?,;:]$/.test(t) ? t : `${t},`
}

let phonemizerPromise: Promise<typeof import('phonemizer')> | null = null

async function phonemizeEnglish(text: string): Promise<string> {
  phonemizerPromise ??= import('phonemizer')
  const { phonemize } = await phonemizerPromise
  const chunks = text.split(/([;:,.!?¡¿—…"«»"()\n]+)/)
  let out = ''
  for (const chunk of chunks) {
    if (!chunk) continue
    if (/^[;:,.!?¡¿—…"«»"()\s]+$/.test(chunk)) {
      out += chunk
      continue
    }
    const ipa = await phonemize(chunk, 'en-us')
    out += ipa.join(' ').replace(/_/g, '')
  }
  return out.trim()
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

function openKittenCache(): Promise<Cache | null> {
  if (typeof caches === 'undefined') return Promise.resolve(null)
  return caches.open(CACHE).catch(() => null)
}

async function fetchCached(
  url: string,
  file: string,
  onProgress: ProgressCallback | undefined,
  cachePromise: Promise<Cache | null>
): Promise<ArrayBuffer> {
  const cache = await cachePromise
  const hit = await cache?.match(url)
  if (hit) {
    onProgress?.({ status: 'progress', progress: 100, file })
    return hit.arrayBuffer()
  }

  const res = await fetch(url)
  if (!res.ok) throw new Error(`Could not load ${file} (${res.status})`)

  const total = Number(res.headers.get('content-length') ?? 0)
  if (!res.body || !total) {
    const buf = await res.arrayBuffer()
    void cache?.put(
      url,
      new Response(buf, { headers: { 'content-length': String(buf.byteLength) } })
    )
    return buf
  }

  const reader = res.body.getReader()
  let out = new Uint8Array(total)
  let received = 0
  let lastEmit = 0
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
    const now = performance.now()
    if (received >= total || now - lastEmit >= 32) {
      lastEmit = now
      onProgress?.({ status: 'progress', progress: (received / total) * 100, file, loaded: received, total })
    }
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

async function loadNpz(buffer: ArrayBuffer): Promise<Record<string, VoiceEntry>> {
  const view = new DataView(buffer)
  const bytes = new Uint8Array(buffer)
  const voices: Record<string, VoiceEntry> = {}

  const eocd = findEocd(view, bytes.length)
  const count = view.getUint16(eocd + 10, true)
  let central = view.getUint32(eocd + 16, true)
  const decoder = new TextDecoder()

  const tasks: Array<Promise<[string, VoiceEntry]>> = []

  for (let i = 0; i < count; i++) {
    if (view.getUint32(central, true) !== 0x02014b50) break
    const method = view.getUint16(central + 10, true)
    const compSize = view.getUint32(central + 20, true)
    const rawSize = view.getUint32(central + 24, true)
    const nameLen = view.getUint16(central + 28, true)
    const extraLen = view.getUint16(central + 30, true)
    const commentLen = view.getUint16(central + 32, true)
    const localOff = view.getUint32(central + 42, true)
    const name = decoder.decode(bytes.subarray(central + 46, central + 46 + nameLen))
    central += 46 + nameLen + extraLen + commentLen
    if (!name.endsWith('.npy')) continue

    const localNameLen = view.getUint16(localOff + 26, true)
    const localExtraLen = view.getUint16(localOff + 28, true)
    const dataStart = localOff + 30 + localNameLen + localExtraLen
    const compressed = bytes.subarray(dataStart, dataStart + compSize)
    const key = name.replace(/\.npy$/, '').split('/').pop() ?? name

    tasks.push(
      (async (): Promise<[string, VoiceEntry]> => {
        const raw =
          method === 0 ? compressed : method === 8 ? await inflateRaw(compressed, rawSize) : null
        if (!raw) throw new Error(`Unsupported zip method ${method} in voices.npz`)
        return [key, parseNpy(raw, decoder)]
      })()
    )
  }

  const resolved = await Promise.all(tasks)
  for (const [key, entry] of resolved) {
    voices[key] = entry
  }

  if (!Object.keys(voices).length) throw new Error('voices.npz contained no arrays')
  return voices
}

function findEocd(view: DataView, length: number): number {
  for (let i = length - 22; i >= Math.max(0, length - 22 - 0xffff); i--) {
    if (view.getUint32(i, true) === 0x06054b50) return i
  }
  throw new Error('voices.npz is not a valid zip')
}

async function inflateRaw(data: Uint8Array, outLen: number): Promise<Uint8Array> {
  const stream = new DecompressionStream('deflate-raw')
  const writer = stream.writable.getWriter()
  void writer.write(data as unknown as BufferSource)
  void writer.close()
  const reader = stream.readable.getReader()
  if (outLen > 0) {
    const out = new Uint8Array(outLen)
    let offset = 0
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      out.set(value, offset)
      offset += value.byteLength
    }
    return offset === outLen ? out : out.subarray(0, offset)
  }
  const chunks: Uint8Array[] = []
  let received = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    received += value.byteLength
  }
  const out = new Uint8Array(received)
  let offset = 0
  for (const chunk of chunks) {
    out.set(chunk, offset)
    offset += chunk.byteLength
  }
  return out
}

function parseNpy(bytes: Uint8Array, decoder: TextDecoder): VoiceEntry {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const major = bytes[6] ?? 1
  const headerLen = major >= 2 ? view.getUint32(8, true) : view.getUint16(8, true)
  const headerOffset = major >= 2 ? 12 : 10
  const header = decoder.decode(bytes.subarray(headerOffset, headerOffset + headerLen))
  const shapeStr = header.match(/'shape'\s*:\s*\(([^)]*)\)/)?.[1]?.trim() ?? ''
  const shape = shapeStr === '' ? [1] : shapeStr.split(',').map((s) => Number(s.trim())).filter((n) => !Number.isNaN(n))
  const start = headerOffset + headerLen
  const avail = bytes.byteLength - start
  const usable = avail - (avail % 4)
  const byteOffset = bytes.byteOffset + start
  const data =
    byteOffset % 4 === 0
      ? new Float32Array(bytes.buffer, byteOffset, usable / 4)
      : new Float32Array(bytes.buffer.slice(byteOffset, byteOffset + usable))
  return { data, shape }
}

export { SAMPLE_RATE as KITTEN_SAMPLE_RATE }
