/**
 * Hero background: an ordered-dither field computed every frame by the browser.
 * Low-res canvas scaled up with image-rendering: pixelated, so the pixels match Geist Pixel.
 */
import { onPage } from './lifecycle'

onPage(() => {
const canvas = document.querySelector<HTMLCanvasElement>('[data-dither]')

if (canvas) {
  const ctx = canvas.getContext('2d', { alpha: true })!
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches
  const accent = getComputedStyle(document.documentElement).getPropertyValue('--color-accent').trim() || '#ffb340'

  const bayer = [
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
  ].map((row) => row.map((v) => (v + 0.5) / 16))

  let W = 0
  let H = 0
  let rectLeft = 0
  let rectTop = 0
  let rectWidth = 1
  let rectHeight = 1
  let img: ImageData
  let px: Uint8ClampedArray
  let nxArr = new Float32Array(0)
  let nx7Arr = new Float32Array(0)
  let fadeXArr = new Float32Array(0)
  const target = { x: 0.7, y: 0.4 }
  const pointer = { x: 0.7, y: 0.4 }
  const [ar, ag, ab] = hexToRgb(accent)

  function resize() {
    const rect = canvas!.getBoundingClientRect()
    rectLeft = rect.left
    rectTop = rect.top
    rectWidth = Math.max(1, rect.width)
    rectHeight = Math.max(1, rect.height)
    const scale = 6 // CSS px per canvas px
    W = Math.max(32, Math.floor(rect.width / scale))
    H = Math.max(24, Math.floor(rect.height / scale))
    canvas!.width = W
    canvas!.height = H
    img = ctx.createImageData(W, H)
    px = img.data
    for (let i = 0; i < px.length; i += 4) {
      px[i] = ar
      px[i + 1] = ag
      px[i + 2] = ab
    }
    nxArr = new Float32Array(W)
    nx7Arr = new Float32Array(W)
    fadeXArr = new Float32Array(W)
    for (let x = 0; x < W; x++) {
      const nx = x / W
      nxArr[x] = nx
      nx7Arr[x] = nx * 7
      fadeXArr[x] = Math.pow(nx, 1.6) * 0.5
    }
  }

  let raf = 0
  let inView = true
  let pageVisible = !document.hidden

  function shouldAnimate() {
    return !reduce && inView && pageVisible
  }

  function startLoop() {
    if (!raf && shouldAnimate()) {
      raf = requestAnimationFrame(frame)
    }
  }

  function stopLoop() {
    if (raf) {
      cancelAnimationFrame(raf)
      raf = 0
    }
  }

  function frame(now: number) {
    raf = 0
    const t = now / 1000
    pointer.x += (target.x - pointer.x) * 0.08
    pointer.y += (target.y - pointer.y) * 0.08
    const pxPtr = pointer.x
    const pyPtr = pointer.y
    const aspect = H / W
    const t06 = t * 0.6
    const t04 = t * 0.4

    for (let y = 0; y < H; y++) {
      const ny = y / H
      const dy = (ny - pyPtr) * aspect
      const dy2 = dy * dy
      const fadeY = 1 - ny * 0.35
      const sinY = Math.sin(ny * 5 - t04) * 1.3 + t06
      const bayerRow = bayer[y & 3]!
      const rowOffset = y * W

      for (let x = 0; x < W; x++) {
        const dx = nxArr[x]! - pxPtr
        const glow = Math.exp(-(dx * dx + dy2) * 10)
        const wave = 0.5 + 0.5 * Math.sin(nx7Arr[x]! + sinY)
        const v = Math.min(1, wave * fadeXArr[x]! * fadeY + glow * 0.55)
        const on = v > bayerRow[x & 3]!
        const i = (rowOffset + x) * 4
        px[i + 3] = on ? 255 * Math.min(1, 0.3 + v * 0.5) : 0
      }
    }
    ctx.putImageData(img, 0, 0)
    if (shouldAnimate()) raf = requestAnimationFrame(frame)
  }

  function hexToRgb(hex: string): [number, number, number] {
    const h = hex.replace('#', '')
    const n = parseInt(h.length === 3 ? h.split('').map((c) => c + c).join('') : h, 16)
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
  }

  resize()
  addEventListener('resize', resize, { passive: true })
  const host = canvas.parentElement ?? canvas
  const onPointerMove = (e: PointerEvent) => {
    target.x = (e.clientX - rectLeft) / rectWidth
    target.y = (e.clientY - rectTop) / rectHeight
  }
  host.addEventListener('pointermove', onPointerMove, { passive: true })

  const io =
    typeof IntersectionObserver !== 'undefined'
      ? new IntersectionObserver((entries) => {
          inView = entries[0]?.isIntersecting ?? true
          if (inView) {
            const rect = canvas!.getBoundingClientRect()
            rectLeft = rect.left
            rectTop = rect.top
            startLoop()
          } else {
            stopLoop()
          }
        })
      : null
  io?.observe(canvas)

  const onVisibilityChange = () => {
    pageVisible = !document.hidden
    if (pageVisible) startLoop()
    else stopLoop()
  }
  document.addEventListener('visibilitychange', onVisibilityChange)

  raf = requestAnimationFrame(frame)
  return () => {
    stopLoop()
    io?.disconnect()
    document.removeEventListener('visibilitychange', onVisibilityChange)
    host.removeEventListener('pointermove', onPointerMove)
    removeEventListener('resize', resize)
  }
}
})
