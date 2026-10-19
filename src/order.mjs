// Reading order, ported from indra_ocr/order_post.py: line features, the
// order model's decoding (blocks -> lines inside blocks) and the geometric
// baselines (XY-cut, top-down rows).

import { median, roundHalfEven } from "./imaging.mjs";
import { isWideChar, isDigitChar } from "./unicode.mjs";

const N_FEATS = 16;
// chandra block labels of the layout head ("cls") output
export const BLOCK_CLASSES = [
  "Text",
  "Section-Header",
  "Page-Header",
  "Page-Footer",
  "List-Group",
  "Table",
  "Caption",
  "Footnote",
  "Table-Of-Contents",
  "Form",
];
export const GRID_LABELS = new Set(
  ["Table", "Form", "Table-Of-Contents"].map((c) => BLOCK_CLASSES.indexOf(c))
);

const SENTENCE_END = ".!?:;。！？：；．";

// str.isdigit() of a word (py): non-empty and every character a digit.
const isDigitWord = (s) => !!s && [...s].every(isDigitChar);
const isUpper = (c) => c !== c.toLowerCase() && c === c.toUpperCase();

// Length with East Asian wide characters counted twice.
const displayLen = (t) => {
  let n = 0;
  for (const c of t) n += isWideChar(c) ? 2 : 1;
  return n;
};

// line_features: [N, 4] boxes + texts -> Float32Array [N, 16] features.
export function lineFeatures(boxes, pageW, pageH, texts) {
  const n = boxes.length;
  const out = new Float32Array(n * N_FEATS);
  if (!n) return out;
  const medH = median(boxes.map((b) => Math.max(b[3] - b[1], 1e-3)));
  for (let i = 0; i < n; i++) {
    const [x0, y0, x1, y1] = boxes[i];
    const w = Math.max(x1 - x0, 1e-3);
    const h = Math.max(y1 - y0, 1e-3);
    const f = i * N_FEATS;
    out[f + 0] = x0 / pageW;
    out[f + 1] = y0 / pageH;
    out[f + 2] = x1 / pageW;
    out[f + 3] = y1 / pageH;
    out[f + 4] = (x0 + x1) / (2 * pageW);
    out[f + 5] = (y0 + y1) / (2 * pageH);
    out[f + 6] = w / pageW;
    out[f + 7] = Math.log(h / medH);
    out[f + 8] = Math.log(w / h) / 4;
    out[f + 9] = (medH / pageH) * 20;
    const t = (texts?.[i] ?? "").trim();
    if (!t) continue;
    let digits = 0;
    for (const c of t) if (isDigitChar(c)) digits++;
    let head = t.slice(0, 3);
    while (head.endsWith(".") || head.endsWith(")")) head = head.slice(0, -1);
    // length in Latin-letter widths: a CJK character is about two wide, so
    // a full CJK line does not read as a short table cell
    out[f + 10] = Math.log1p(displayLen(t)) / 5;
    out[f + 11] = digits / t.length;
    out[f + 12] = Number(
      "•-–▪◦*·→".includes(t[0]) || (isDigitWord(head) && t.length > 3)
    );
    out[f + 13] = Number(t.endsWith("-"));
    out[f + 14] = Number(SENTENCE_END.includes(t[t.length - 1]));
    out[f + 15] = Number(isUpper(t[0]));
  }
  return out;
}

// log P(i before j) from a pairwise logit
const lpOf = (a) => (a >= 0 ? -Math.log1p(Math.exp(-a)) : a - Math.log1p(Math.exp(a)));

// decode_order: order from a pairwise logit matrix.  Start from the row-sum
// ranking, then local search with insertion moves (see the Python docstring).
export function decodeOrder(a, n, sweeps = 4) {
  if (n <= 1) return Array.from({ length: n }, (_, i) => i);
  const score = new Float64Array(n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) score[i] += lpOf(a[j * n + i]) - lpOf(a[i * n + j]);
  let order = Array.from({ length: n }, (_, i) => i).sort((p, q) => score[p] - score[q]);
  for (let s = 0; s < sweeps; s++) {
    let improved = false;
    for (const x of [...order]) {
      const rest = order.filter((v) => v !== x);
      const before = rest.map((r) => lpOf(a[r * n + x])); // x placed after these
      const after = rest.map((r) => lpOf(a[x * n + r])); // x placed before these
      const gains = new Float64Array(n);
      let run = 0;
      for (let k = 1; k < n; k++) gains[k] = run += before[k - 1];
      run = 0;
      for (let k = n - 2; k >= 0; k--) gains[k] += run += after[k];
      let best = 0;
      for (let k = 1; k < n; k++) if (gains[k] > gains[best]) best = k;
      const cur = order.indexOf(x);
      if (gains[best] > gains[cur] + 1e-6) {
        rest.splice(best, 0, x);
        order = rest;
        improved = true;
      }
    }
    if (!improved) break;
  }
  return order;
}

// _cluster: average-linkage clustering of pairwise "same group" logits
// (average-linkage over the same-block logits; see the Python docstring:
// merge the two groups with the highest mean logit while it stays positive).
export function cluster(same, n) {
  if (!n) return [];
  const M = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) M[i] = same[i];
  for (let i = 0; i < n; i++) M[i * n + i] = -Infinity;
  const alive = new Uint8Array(n).fill(1);
  const best = new Int32Array(n);
  const bestV = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let b = 0;
    for (let j = 1; j < n; j++) if (M[i * n + j] > M[i * n + b]) b = j;
    best[i] = b;
    bestV[i] = M[i * n + b];
  }
  const members = Array.from({ length: n }, (_, i) => [i]);
  const S = new Float64Array(M); // sum of logits between groups
  const size = new Float64Array(n).fill(1);
  for (let it = 0; it < n - 1; it++) {
    let a = 0;
    for (let i = 1; i < n; i++) if (bestV[i] > bestV[a]) a = i;
    let b = best[a];
    if (bestV[a] <= 0) break;
    if (b < a) [a, b] = [b, a];
    for (let j = 0; j < n; j++) S[a * n + j] += S[b * n + j];
    for (let j = 0; j < n; j++) S[j * n + a] += S[j * n + b];
    S[a * n + a] = 0;
    size[a] += size[b];
    alive[b] = 0;
    members[a].push(...members[b]);
    const row = new Float64Array(n);
    for (let j = 0; j < n; j++)
      row[j] = alive[j] && j !== a ? S[a * n + j] / (size[a] * size[j]) : -Infinity;
    for (let j = 0; j < n; j++) {
      M[a * n + j] = M[j * n + a] = row[j];
      M[b * n + j] = M[j * n + b] = -Infinity;
    }
    bestV[b] = -Infinity;
    let rb = 0;
    for (let j = 1; j < n; j++) if (row[j] > row[rb]) rb = j;
    best[a] = rb;
    bestV[a] = row[rb];
    for (let j = 0; j < n; j++) {
      if (!alive[j] || j === a) continue;
      if (best[j] === a || best[j] === b) {
        let nb = 0;
        for (let k = 1; k < n; k++) if (M[j * n + k] > M[j * n + nb]) nb = k;
        best[j] = nb;
        bestV[j] = M[j * n + nb];
      }
      if (row[j] > bestV[j]) {
        best[j] = a;
        bestV[j] = row[j];
      }
    }
  }
  const out = [];
  for (let i = 0; i < n; i++) if (alive[i]) out.push(members[i].sort((p, q) => p - q));
  return out;
}

// order_top_down: rows top to bottom (centres within half a line height share
// a row), left to right.
export function topDown(boxes) {
  if (!boxes.length) return [];
  const medH = median(boxes.map((b) => b[3] - b[1]));
  return boxes
    .map((b, i) => [roundHalfEven((b[1] + b[3]) / 2 / Math.max(medH * 0.5, 1e-3)), b[0], i])
    .sort((p, q) => p[0] - q[0] || p[1] - q[1])
    .map((t) => t[2]);
}

// order_xycut: recursive XY-cut, split at the widest whitespace gap.
export function xyCut(boxes) {
  const bestCut = (b, axis) => {
    const lo = b.map((bb) => (axis ? bb[1] : bb[0]));
    const hi = b.map((bb) => (axis ? bb[3] : bb[2]));
    const o = lo.map((_, i) => i).sort((p, q) => lo[p] - lo[q]);
    let end = hi[o[0]];
    let bestGap = 0;
    let bestK = -1;
    for (let k = 1; k < o.length; k++) {
      const gap = lo[o[k]] - end;
      if (gap > bestGap) {
        bestGap = gap;
        bestK = k;
      }
      end = Math.max(end, hi[o[k]]);
    }
    return [bestGap, bestK, o];
  };
  const rec = (ids) => {
    if (ids.length <= 1) return [...ids];
    const b = ids.map((i) => boxes[i]);
    const [gy, ky, oy] = bestCut(b, 1);
    const [gx, kx, ox] = bestCut(b, 0);
    if (ky < 0 && kx < 0) return topDown(b).map((k) => ids[k]);
    const [k, o] = gy >= gx ? [ky, oy] : [kx, ox];
    return rec(o.slice(0, k).map((i) => ids[i])).concat(rec(o.slice(k).map((i) => ids[i])));
  };
  return rec(boxes.map((_, i) => i));
}

// xycut_blocks: blocks in reading order by recursive XY-cut, for pages the
// order model does not know (vertical pages, read turned).  A cut needs a
// whitespace gap of at least `minGap` median line heights; the widest gap cuts
// first, the upper or left part first.  Lines inside a block go by `xyCut`.
export function xycutBlocks(boxes, minGap = 1.0) {
  if (!boxes.length) return [];
  const h = median(boxes.map((b) => b[3] - b[1]));
  const bestCut = (b, axis) => {
    const lo = b.map((bb) => (axis ? bb[1] : bb[0]));
    const hi = b.map((bb) => (axis ? bb[3] : bb[2]));
    const o = lo.map((_, i) => i).sort((p, q) => lo[p] - lo[q]);
    let end = hi[o[0]];
    let bestGap = 0;
    let bestK = -1;
    for (let k = 1; k < o.length; k++) {
      const gap = lo[o[k]] - end;
      if (gap > bestGap) {
        bestGap = gap;
        bestK = k;
      }
      end = Math.max(end, hi[o[k]]);
    }
    return [bestGap, bestK, o];
  };
  const rec = (ids) => {
    const b = ids.map((i) => boxes[i]);
    if (ids.length > 1) {
      const [gy, ky, oy] = bestCut(b, 1);
      const [gx, kx, ox] = bestCut(b, 0);
      const [gap, k, o] = gy >= gx ? [gy, ky, oy] : [gx, kx, ox];
      if (k >= 0 && gap >= minGap * h)
        return rec(o.slice(0, k).map((i) => ids[i])).concat(
          rec(o.slice(k).map((i) => ids[i]))
        );
    }
    return [xyCut(b).map((k) => ids[k])];
  };
  return rec(boxes.map((_, i) => i));
}

// decode_blocks_order: blocks from the same-block head (average-linkage
// clustering), lines inside a block by geometry, blocks by the mean pairwise
// order logit between them.  a / same are the order model's [N, N] outputs
// (row-major Float32Array), cls its [N, 10] layout logits.
export function decodeBlocksOrder(a, same, boxes, cls) {
  const n = boxes.length;
  if (n <= 1) return Array.from({ length: n }, (_, i) => i);
  const blocks = cluster(same, n);
  const inner = blocks.map((b) => {
    let isGrid = false;
    if (cls) {
      const sum = new Float64Array(BLOCK_CLASSES.length);
      for (const i of b)
        for (let c = 0; c < BLOCK_CLASSES.length; c++) sum[c] += cls[i * BLOCK_CLASSES.length + c];
      let am = 0;
      for (let c = 1; c < BLOCK_CLASSES.length; c++) if (sum[c] > sum[am]) am = c;
      isGrid = GRID_LABELS.has(am);
    }
    const bx = b.map((i) => boxes[i]);
    const local = isGrid ? topDown(bx) : xyCut(bx);
    return local.map((k) => b[k]);
  });
  const m = blocks.length;
  const ab = new Float64Array(m * m);
  for (let p = 0; p < m; p++)
    for (let q = p + 1; q < m; q++) {
      let s = 0;
      for (const i of blocks[p]) for (const j of blocks[q]) s += a[i * n + j];
      const v = s / (blocks[p].length * blocks[q].length);
      ab[p * m + q] = v;
      ab[q * m + p] = -v;
    }
  return decodeOrder(ab, m).flatMap((k) => inner[k]);
}
