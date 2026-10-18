// Line detection pre/post-processing, ported from indra_ocr/det_post.py.
import { resizeGray, roundPy } from "./imaging.mjs";

const INPUT_MULTIPLE = 32;
const OUT_STRIDE = 2;
const CORE_X_SHRINK = 0.15;

// page_to_input: uint8 page -> float32 ink map (0 = white), long side scaled,
// padded up to a multiple of 32 with white.  Returns the map and the scale.
export function pageToInput(gray, longSide) {
  const { data, width: w, height: h } = gray;
  const scale = longSide / Math.max(h, w);
  const nh = Math.max(1, roundPy(h * scale));
  const nw = Math.max(1, roundPy(w * scale));
  const small =
    nh === h && nw === w
      ? data
      : resizeGray(data, w, h, nw, nh, scale < 1 ? "area" : "linear");
  const ph = Math.ceil(nh / INPUT_MULTIPLE) * INPUT_MULTIPLE;
  const pw = Math.ceil(nw / INPUT_MULTIPLE) * INPUT_MULTIPLE;
  const x = new Float32Array(ph * pw);
  for (let r = 0; r < nh; r++) {
    for (let c = 0; c < nw; c++) x[r * pw + c] = 1 - small[r * nw + c] / 255;
  }
  return { x, ph, pw, scale };
}

// decode(): out [3, h, w] (core logit, top dist, bottom dist) at half
// resolution -> boxes [N, 5] (x0, y0, x1, y1, score) in input pixels.
// Connected components (4-connectivity) of core > thresh; per component the
// mean of the per-pixel reconstructed line top / bottom and mean probability.
export function decode(out, h, w, thresh = 0.5, minArea = 2) {
  const n = h * w;
  const core = out.subarray(0, n);
  const dTop = out.subarray(n, 2 * n);
  const dBot = out.subarray(2 * n, 3 * n);
  const labels = new Int32Array(n); // 0 = background, ids in scan order
  const prob = new Float32Array(n);
  for (let i = 0; i < n; i++) prob[i] = 1 / (1 + Math.exp(-core[i]));
  const stats = [{ area: 0 }]; // per label, index 0 unused
  const stack = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    if (labels[i] !== 0 || prob[i] <= thresh) continue;
    const lab = stats.length;
    let sp = 0;
    stack[sp++] = i;
    labels[i] = lab;
    const st = { area: 0, x0: w, x1: -1, sumTop: 0, sumBot: 0, sumSc: 0 };
    while (sp > 0) {
      const j = stack[--sp];
      const r = (j / w) | 0;
      const c = j - r * w;
      st.area++;
      if (c < st.x0) st.x0 = c;
      if (c > st.x1) st.x1 = c;
      const ys = (r + 0.5) * OUT_STRIDE;
      st.sumTop += ys - dTop[j];
      st.sumBot += ys + dBot[j];
      st.sumSc += prob[j];
      if (c > 0 && labels[j - 1] === 0 && prob[j - 1] > thresh) {
        labels[j - 1] = lab;
        stack[sp++] = j - 1;
      }
      if (c < w - 1 && labels[j + 1] === 0 && prob[j + 1] > thresh) {
        labels[j + 1] = lab;
        stack[sp++] = j + 1;
      }
      if (r > 0 && labels[j - w] === 0 && prob[j - w] > thresh) {
        labels[j - w] = lab;
        stack[sp++] = j - w;
      }
      if (r < h - 1 && labels[j + w] === 0 && prob[j + w] > thresh) {
        labels[j + w] = lab;
        stack[sp++] = j + w;
      }
    }
    stats.push(st);
  }
  const boxes = [];
  for (let lab = 1; lab < stats.length; lab++) {
    const st = stats[lab];
    if (st.area < minArea) continue;
    const y0 = st.sumTop / st.area;
    const y1 = st.sumBot / st.area;
    const lh = Math.max(1, y1 - y0);
    const ex = Math.min(CORE_X_SHRINK * lh, 0.5 * lh);
    boxes.push([
      st.x0 * OUT_STRIDE - ex,
      y0,
      (st.x1 + 1) * OUT_STRIDE + ex,
      y1,
      st.sumSc / st.area,
    ]);
  }
  return boxes;
}
