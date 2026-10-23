import type { Device, ProgressCallback, ResolvedDevice } from '../core/index.ts'
import { SUPERTONIC_CACHE_NAME, resolveDevice, toProgressInfo } from '../core/index.ts'
import { splitUtterances } from './split.ts'
import { toFloat32 } from './wav.ts'
import { SUPERTONIC_LANGUAGES, type SupertonicLanguage } from './voices.ts'
import type { SpeakChunk } from './types.ts'

/**
 * Supertonic 2 (Supertone). One 44.1 kHz voice model for English, Korean, Spanish,
 * Portuguese and French; 10 preset voices shared across languages. OpenRAIL-M license.
 */
const DEFAULT_MODEL = 'onnx-community/Supertonic-TTS-2-ONNX'
const SAMPLE_RATE = 44_100
const VOICE_CACHE = SUPERTONIC_CACHE_NAME

export { SUPERTONIC_LANGUAGES, type SupertonicLanguage }

type RawAudio = { audio?: Float32Array | number[]; data?: Float32Array | number[]; sampling_rate?: number }

type SupertonicPipe = {
  (
    text: string,
    options: { speaker_embeddings: Float32Array; num_inference_steps?: number; speed?: number }
  ): Promise<RawAudio>
  dispose?: () => Promise<void>
}

export type LoadedSupertonic = {
  pipe: SupertonicPipe
  modelId: string
  device: ResolvedDevice
}

export async function loadSupertonic(options: {
  model?: string
  device?: Device
  onProgress?: ProgressCallback
}): Promise<LoadedSupertonic> {
  options.onProgress?.({ status: 'loading', progress: 0 })

  const [device, { pipeline, env }] = await Promise.all([
    resolveDevice(options.device ?? 'auto'),
    import('@huggingface/transformers'),
  ])
  env.allowLocalModels = false
  const modelId = options.model ?? DEFAULT_MODEL

  const pipe = await pipeline('text-to-speech', modelId, {
    device,
    dtype: 'fp32',
    progress_callback: (data: Record<string, unknown>) => {
      options.onProgress?.(toProgressInfo(data))
    },
  })

  options.onProgress?.({ status: 'ready', progress: 100 })
  return { pipe: pipe as unknown as SupertonicPipe, modelId, device }
}

export async function* streamSupertonic(
  st: LoadedSupertonic,
  text: string,
  voice: string,
  speed: number,
  language: string,
  steps = 5
): AsyncGenerator<SpeakChunk> {
  const lang = (SUPERTONIC_LANGUAGES as readonly string[]).includes(language) ? language : 'en'
  const style = await loadVoice(st.modelId, voice)
  for (const piece of splitUtterances(text)) {
    const raw = await st.pipe(`<${lang}>${piece}</${lang}>`, {
      speaker_embeddings: style,
      num_inference_steps: steps,
      speed,
    })
    yield { audio: toFloat32(raw.audio ?? raw.data ?? []), samplingRate: raw.sampling_rate || SAMPLE_RATE, text: piece }
  }
}

const voiceCache = new Map<string, Promise<Float32Array>>()

/** Supertonic voice ids are `F1`…`F5` and `M1`…`M5`; we expose them as `st_f1`… */
export function supertonicVoiceFile(voice: string): string {
  const m = /^(?:st_)?([fm])([1-5])$/i.exec(voice)
  if (!m) throw new Error(`Unknown Supertonic voice "${voice}"`)
  return `${m[1]!.toUpperCase()}${m[2]}`
}

function loadVoice(modelId: string, voice: string): Promise<Float32Array> {
  const file = supertonicVoiceFile(voice)
  const key = `${modelId}/${file}`
  let pending = voiceCache.get(key)
  if (!pending) {
    pending = fetchVoice(modelId, file).catch((err) => {
      voiceCache.delete(key)
      throw err
    })
    voiceCache.set(key, pending)
  }
  return pending
}

async function fetchVoice(modelId: string, file: string): Promise<Float32Array> {
  const url = `https://huggingface.co/${modelId}/resolve/main/voices/${file}.bin`
  let cache: Cache | null = null
  try {
    cache = await caches.open(VOICE_CACHE)
    const hit = await cache.match(url)
    if (hit) return new Float32Array(await hit.arrayBuffer())
  } catch {
    cache = null
  }
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Voice "${file}" not found (${res.status})`)
  const buffer = await res.arrayBuffer()
  void cache
    ?.put(
      url,
      new Response(buffer, {
        headers: {
          'content-type': 'application/octet-stream',
          'content-length': String(buffer.byteLength),
        },
      })
    )
    .catch(() => undefined)
  return new Float32Array(buffer)
}

export function disposeSupertonic(st: LoadedSupertonic | null) {
  void st?.pipe.dispose?.()
}

export { DEFAULT_MODEL as SUPERTONIC_MODEL, SAMPLE_RATE as SUPERTONIC_SAMPLE_RATE }
