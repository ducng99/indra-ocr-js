// Image helpers, ported from the Python code's cv2 usage so the ONNX models
// see (near-)identical inputs.  Only bilinear and box ("area") filtering is
// needed, both separable, so a plain typed-array implementation is enough —
// and it keeps the code browser-portable (image decoding lives in
// node-io.mjs / the app).

// Round half to even (needed to match the reference implementation's output).
export function roundHalfEven(x) {
  const f = Math.floor(x);
  const d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

export const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
export const median = (xs) => {
  if (!xs.length) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

// Per-axis filter taps, matching cv2's separable resize:
//  - shrinking with mode "area": overlap-weighted box average
//  - anything else: bilinear (INTER_AREA falls back to it, and callers pass
//    "linear" where cv2 passes INTER_LINEAR)
function taps(srcN, dstN, mode) {
  const out = [];
  if (mode === "area" && dstN < srcN) {
    const inv = srcN / dstN;
    for (let d = 0; d < dstN; d++) {
      const x0 = d * inv;
      const x1 = (d + 1) * inv;
      const t = [];
      for (let i = Math.floor(x0); i < Math.ceil(x1); i++) {
        const w = (Math.min(x1, i + 1) - Math.max(x0, i)) / inv; // normalized
        if (w > 0) t.push([i, w]);
      }
      out.push(t);
    }
    return out;
  }
  const scale = srcN / dstN;
  for (let d = 0; d < dstN; d++) {
    const s = (d + 0.5) * scale - 0.5;
    let i0 = Math.floor(s);
    const f = s - i0;
    let i1 = i0 + 1;
    if (i0 < 0) i0 = 0;
    if (i1 > srcN - 1) i1 = srcN - 1;
    out.push(i0 === i1 || f <= 0 ? [[i0, 1]] : [[i0, 1 - f], [i1, f]]);
  }
  return out;
}

// Resize a uint8 grayscale image (row-major) to [dw, dh].
export function resizeGray(src, sw, sh, dw, dh, mode = "linear") {
  if (dw === sw && dh === sh) return src;
  const tx = taps(sw, dw, mode);
  const ty = taps(sh, dh, mode);
  const tmp = new Float64Array(dw * sh); // x pass
  for (let y = 0; y < sh; y++) {
    const row = y * sw;
    for (let d = 0; d < dw; d++) {
      let a = 0;
      for (const [i, w] of tx[d]) a += src[row + i] * w;
      tmp[y * dw + d] = a;
    }
  }
  const out = new Uint8Array(dw * dh); // y pass + round
  for (let x = 0; x < dw; x++) {
    for (let d = 0; d < dh; d++) {
      let a = 0;
      for (const [i, w] of ty[d]) a += tmp[i * dw + x] * w;
      out[d * dw + x] = clamp(Math.round(a), 0, 255);
    }
  }
  return out;
}

// cv2.getRotationMatrix2D: the 2x3 affine matrix (row-major, 6 numbers) that
// rotates `angleDeg` degrees counter-clockwise about (cx, cy), scaled.
export function rotMatrix2D(cx, cy, angleDeg, scale = 1) {
  const a = (angleDeg * Math.PI) / 180;
  const alpha = scale * Math.cos(a);
  const beta = scale * Math.sin(a);
  return [
    alpha,
    beta,
    (1 - alpha) * cx - beta * cy,
    -beta,
    alpha,
    beta * cx + (1 - alpha) * cy,
  ];
}

// cv2.warpAffine with INTER_LINEAR: the output pixel at (x, y) samples the
// source at the inverse of the forward matrix `m`, bilinearly, each tap border
// -checked like cv2 ("constant" fills out-of-image taps with 0, cv2's default;
// "replicate" clamps to the edge pixels).  Same output size as the input;
// element type follows the source (Uint8Array or Float32Array).
export function warpAffine(src, m, border = "constant") {
  const { data, width: sw, height: sh } = src;
  const dw = src.width;
  const dh = src.height;
  // invert [m0 m1 m2; m3 m4 m5]
  const det = m[0] * m[4] - m[1] * m[3];
  const i0 = m[4] / det;
  const i1 = -m[1] / det;
  const i2 = (m[1] * m[5] - m[2] * m[4]) / det;
  const i3 = -m[3] / det;
  const i4 = m[0] / det;
  const i5 = (m[2] * m[3] - m[0] * m[5]) / det;
  const rep = border === "replicate";
  const px = (x, y) => {
    if (rep) {
      x = x < 0 ? 0 : x >= sw ? sw - 1 : x;
      y = y < 0 ? 0 : y >= sh ? sh - 1 : y;
    } else if (x < 0 || x >= sw || y < 0 || y >= sh) return 0;
    return data[y * sw + x];
  };
  const out = new data.constructor(dw * dh);
  for (let y = 0; y < dh; y++) {
    const rx = i1 * y + i2;
    const ry = i4 * y + i5;
    const row = y * dw;
    for (let x = 0; x < dw; x++) {
      const sx = i0 * x + rx;
      const sy = i3 * x + ry;
      const ix = Math.floor(sx);
      const iy = Math.floor(sy);
      const fx = sx - ix;
      const fy = sy - iy;
      out[row + x] =
        px(ix, iy) * (1 - fx) * (1 - fy) +
        px(ix + 1, iy) * fx * (1 - fy) +
        px(ix, iy + 1) * (1 - fx) * fy +
        px(ix + 1, iy + 1) * fx * fy;
    }
  }
  return { data: out, width: dw, height: dh };
}

// np.rot90(m, -k): the image turned k quarter-turns clockwise.
export function rotQuarterCW(gray, k) {
  let { data, width: w, height: h } = gray;
  for (let n = 0; n < k; n++) {
    const out = new Uint8Array(w * h); // dims swap: [h, w] -> [w, h]
    for (let y = 0; y < w; y++)
      for (let x = 0; x < h; x++) out[y * h + x] = data[(h - 1 - x) * w + y];
    data = out;
    [w, h] = [h, w];
  }
  return { data, width: w, height: h };
}
