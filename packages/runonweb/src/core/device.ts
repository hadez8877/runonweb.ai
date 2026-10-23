export type Device = 'webgpu' | 'wasm' | 'auto'
export type ResolvedDevice = 'webgpu' | 'wasm'

type GpuProbeResult = {
  available: boolean
  shaderF16: boolean
}

let gpuProbePromise: Promise<GpuProbeResult> | null = null

function probeWebGPU(): Promise<GpuProbeResult> {
  if (gpuProbePromise) return gpuProbePromise

  gpuProbePromise = (async (): Promise<GpuProbeResult> => {
    try {
      const gpu = (globalThis as { navigator?: { gpu?: GPU } }).navigator?.gpu
      if (!gpu) return { available: false, shaderF16: false }
      const adapter = await gpu.requestAdapter()
      if (!adapter) return { available: false, shaderF16: false }
      return {
        available: true,
        shaderF16: Boolean(adapter.features?.has('shader-f16')),
      }
    } catch {
      return { available: false, shaderF16: false }
    }
  })()

  return gpuProbePromise
}

/**
 * Detect whether WebGPU is available in this environment.
 * Result (including the in-flight probe) is cached after the first check.
 */
export async function isWebGPUAvailable(): Promise<boolean> {
  return (await probeWebGPU()).available
}

/**
 * Detect whether the WebGPU adapter supports `shader-f16`.
 * Reuses the cached WebGPU adapter probe.
 */
export async function supportsShaderF16(): Promise<boolean> {
  return (await probeWebGPU()).shaderF16
}

/**
 * Resolve `auto` to `webgpu` when available, otherwise `wasm`.
 */
export async function resolveDevice(device: Device = 'auto'): Promise<ResolvedDevice> {
  if (device === 'webgpu' || device === 'wasm') return device
  return (await isWebGPUAvailable()) ? 'webgpu' : 'wasm'
}

/**
 * Global queue for direct ONNX Runtime Web (WebGPU/WASM) session runs.
 * ONNX Runtime Web's WebGPU backend uses a single global execution context
 * and throws "Session already started" if two `session.run()` calls overlap.
 */
let ortQueue: Promise<unknown> = Promise.resolve()

export function takeOrtTurn<T>(fn: () => Promise<T>): Promise<T> {
  const next = ortQueue.then(fn, fn)
  ortQueue = next.then(
    () => undefined,
    () => undefined
  )
  return next
}
