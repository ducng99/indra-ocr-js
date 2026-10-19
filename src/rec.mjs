// Line crop rule and batching, ported from indra_ocr/rec_preproc.py.
import { resizeGray, roundHalfEven } from "./imaging.mjs";

export const LINE_HEIGHT = 32;
export const PAD_Y = 0.2;
export const PAD_X = 0.3;
const WIDTH_MULTIPLE = 8;

// crop_line: expand the line box by PAD_X / PAD_Y of its height, crop from the
// page (white outside), resize to `height` px tall keeping the aspect ratio,
// return a float32 ink map {ink: Float32Array, w}.
export function cropLine(gray, box, height = LINE_HEIGHT, maxW = 4096) {
  const [x0, y0, x1, y1] = box;
  const h = Math.max(1, y1 - y0);
  const X0 = x0 - PAD_X * h;
  const X1 = x1 + PAD_X * h;
  const Y0 = y0 - PAD_Y * h;
  const Y1 = y1 + PAD_Y * h;
  const H = gray.height;
  const W = gray.width;
  const ix0 = Math.floor(X0);
  const iy0 = Math.floor(Y0);
  const ix1 = Math.ceil(X1);
  const iy1 = Math.ceil(Y1);
  const cw = Math.max(1, ix1 - ix0);
  const ch = Math.max(1, iy1 - iy0);
  const crop = new Uint8Array(ch * cw).fill(255);
  const sx0 = Math.max(0, ix0);
  const sy0 = Math.max(0, iy0);
  const sx1 = Math.min(W, ix1);
  const sy1 = Math.min(H, iy1);
  if (sx1 > sx0 && sy1 > sy0) {
    for (let r = sy0; r < sy1; r++) {
      const srcRow = r * W;
      const dstRow = (r - iy0) * cw;
      for (let c = sx0; c < sx1; c++) crop[dstRow + (c - ix0)] = gray.data[srcRow + c];
    }
  }
  const outW = Math.max(WIDTH_MULTIPLE, Math.min(maxW, roundHalfEven((cw * height) / ch)));
  const resized = resizeGray(crop, cw, ch, outW, height, ch > height ? "area" : "linear");
  const ink = new Float32Array(resized.length);
  for (let i = 0; i < resized.length; i++) ink[i] = 1 - resized[i] / 255;
  return { ink, w: outW };
}

// pad_batch: right-pad ink maps to one width (a multiple of 8).
// Returns {arr: Float32Array [B,1,H,Wm], w: Wm, widths: BigInt64Array}.
export function padBatch(lines) {
  const H = lines[0].ink.length / lines[0].w;
  let Wm = Math.max(...lines.map((l) => l.w));
  Wm = Math.ceil(Wm / WIDTH_MULTIPLE) * WIDTH_MULTIPLE;
  const arr = new Float32Array(lines.length * H * Wm);
  const widths = new BigInt64Array(lines.length);
  lines.forEach((l, i) => {
    for (let r = 0; r < H; r++) {
      const src = r * l.w;
      const dst = i * H * Wm + r * Wm;
      for (let c = 0; c < l.w; c++) arr[dst + c] = l.ink[src + c];
    }
    widths[i] = BigInt(l.w);
  });
  return { arr, w: Wm, widths };
}

// width_batches: group line indices into batches of similar width (see the
// Python docstring: less compute goes to padding, and allocations stay small).
export function widthBatches(widths, maxBatch = 64, maxRatio = 1.3, minWidth = 64, maxArea = 64 * 640) {
  const order = widths.map((w, i) => i).sort((a, b) => widths[a] - widths[b]);
  const batches = [];
  for (const i of order) {
    const last = batches[batches.length - 1];
    if (
      last &&
      last.length < maxBatch &&
      Math.max(widths[i], minWidth) <= maxRatio * Math.max(widths[last[0]], minWidth) &&
      (last.length + 1) * Math.max(widths[i], minWidth) <= maxArea
    ) {
      last.push(i);
    } else {
      batches.push([i]);
    }
  }
  return batches;
}
