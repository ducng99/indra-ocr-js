// The indra-ocr pipeline, ported from indra_ocr/pipeline.py and friends.
// Pure ESM (no Node or browser APIs): the ONNX Runtime sessions and the
// Tensor class come from an injected backend — src/index.mjs builds one on
// onnxruntime-node, src/browser.mjs on onnxruntime-web (wasm).
//
// The model folder holds det.onnx, rec.onnx, meta.json and usually order.onnx
// and orient.onnx.  With script.onnx and a group recognizer (rec_cjk.onnx,
// rec_rtl.onnx, rec_world.onnx), each line goes to the recognizer of its
// script group (GROUPS); every recognizer also reads Latin, and a group
// without its recognizer is read by rec.onnx.  A right-to-left recognizer
// reads in visual order: rtl.flip turns its text into logical order.  With
// orient.onnx, a page turned by a quarter or half turn is turned upright
// before detection (result.rotation; boxes are in the upright page), and a
// page set in vertical columns is read a quarter turn, columns as lines.
// lm.npz / lm_<group>.npz add beam search with a token n-gram language model
// for the lines a recognizer is unsure of.
//
// Not ported here (optional in the Python package and absent from the current
// model revision): mask.onnx (CenterMask line masking).
import { pageToInput, decodeBoxes } from "./det.mjs";
import { cropLine, isDarkLine, padBatch, widthBatches, PAD_X } from "./rec.mjs";
import {
  lineFeatures,
  decodeBlocksOrder,
  decodeOrder,
  xyCut,
  xycutBlocks,
  GRID_LABELS,
  BLOCK_CLASSES,
} from "./order.mjs";
import { NgramLM, beamSearch } from "./lm.mjs";
import { MATH_MIN_CONF, MathReader } from "./math.mjs";
import { pageBlocks, pageMarkdown } from "./layout.mjs";
import { Vocab } from "./vocab.mjs";
import { flip, hasRtl, isRtl } from "./rtl.mjs";
import {
  clamp,
  median,
  resizeGray,
  rotQuarterCW,
  rotMatrix2D,
  roundHalfEven,
  warpAffine,
} from "./imaging.mjs";

const ADAPTIVE_MIN_H = 9.0;
const ADAPTIVE_TARGET_H = 12.0;
const ADAPTIVE_MAX_SIDE = 2048;
const ORIENT_SIDE = 768;
const ORIENT_MIN = 0.95;
const VERTICAL = 4; // orient.onnx class (5-class models): the page is set in vertical columns
const MIN_CONF = 0.5;
const SPLIT_GAP = 3.0;
const CELL_GAP = 1.8; // cut a table line into cells at a wider gap (label rule in tables)
const GUTTER_GAP = 1.6;
const GUTTER_EDGES = 3;

// Beam search with a language model (lm.npz for the base recognizer,
// lm_<group>.npz for a group) reads a recognizer's lines under this confidence;
// the rest keep the greedy reading.  alpha weighs the LM, beta rewards each
// token (meta.json "lm_params" overrides them per group).
export const LM_GATE = 0.9;
export const LM_ALPHA = 0.5;
export const LM_BETA = 1.5;
const LM_BEAM = 8;
// A class under this probability at a step is not tried by the beam search.
const LM_CAND_FLOOR = 3e-3;

// Recognizer groups, in script.onnx's class order unless meta.json names them
// ("script_groups").  base is rec.onnx.  An older script.onnx has one output:
// the chance that the line holds Chinese characters.
const GROUPS = ["base", "cjk", "rtl", "other"];

// A line goes to one recognizer when the classifier is surer than this;
// otherwise base reads it as well and the more confident reading wins.
const ROUTE_SURE = 0.95;

// det.onnx picture channel: a region is a picture above this probability and
// this share of the page area.
const PICTURE_MIN = 0.5;
const PICTURE_MIN_AREA = 0.004;

const DESKEW_MAX = 45.0; // degrees of tilt searched (quarter turns cover the rest)
const DESKEW_MIN = 0.3; // smaller tilts are left alone (the search step is 0.1)
const PREVIEW_SIDE = 1600; // the skew search and orient.onnx read thumbnails of at most 800

const round1 = (v) => roundHalfEven(v * 10) / 10;
const round2 = (v) => roundHalfEven(v * 100) / 100;
const round4 = (v) => roundHalfEven(v * 1e4) / 1e4;

// Two kinds of lines carry no text of their own (see pipeline.py tidy_lines):
// a line of leader characters only is dropped, and a lone list marker joins
// the line that starts within MARKER_GAP line heights after it on the same row.
const LEADER_CHARS = new Set(".·…‥_-–—="); // dot leaders, fill-in rules
const DASHES = new Set("-–—"); // a lone short dash can be a table cell; other lone marks are not
const BULLETS = new Set("•◦▪▫●○■□‣⁃∙*"); // list markers (not dashes: a "-" can be a table cell)
// A numbered list marker: "1.", "12)", "a.", "B)", "(3)", "(b)", "(iv)".
const ENUMERATOR = /^(?:\d{1,3}|[A-Za-z])[.)]$|^\((?:\d{1,3}|[A-Za-z]|[ivxlcIVXLC]{1,5})\)$/;
// A list marker and its item's text are at most this many line heights apart:
// the label rule (training labels cut a text-layer line only at wider gaps).
const MARKER_GAP = 2.5;

function tidyLines(boxes, rec, rtl = false, space = null) {
  const keep = rec.map(() => true);
  const h = boxes.map((b) => b[3] - b[1]);
  const w = boxes.map((b) => b[2] - b[0]);
  for (let i = 0; i < rec.length; i++) {
    const core = rec[i].text.replace(/ /g, "");
    if (core && [...core].every((c) => LEADER_CHARS.has(c))) {
      const wide = w[i] >= 3 * h[i];
      const isDash = [...core].some((c) => DASHES.has(c));
      if (core.length >= 3 || wide || !isDash) keep[i] = false;
    }
  }
  for (let i = 0; i < rec.length; i++) {
    const t = rec[i].text.trim();
    if (!keep[i] || !(BULLETS.has(t) || ENUMERATOR.test(t))) continue;
    let best = null;
    let bestGap = null;
    for (let j = 0; j < rec.length; j++) {
      if (j === i || !keep[j] || !rec[j].text.trim()) continue;
      const ov = Math.min(boxes[i][3], boxes[j][3]) - Math.max(boxes[i][1], boxes[j][1]);
      if (ov < 0.5 * Math.min(h[i], h[j])) continue;
      const gap = rtl ? boxes[i][0] - boxes[j][2] : boxes[j][0] - boxes[i][2];
      if (
        -0.3 * h[i] <= gap &&
        gap <= MARKER_GAP * Math.max(h[i], h[j]) &&
        (bestGap === null || gap < bestGap)
      ) {
        best = j;
        bestGap = gap;
      }
    }
    if (best === null) continue;
    const kj = rec[best].toks;
    const chars = kj.length > 0 && typeof kj[0][0] === "string";
    const sp = chars ? " " : space;
    // same kind of tokens (text-layer chars or recognizer ids) only
    if (!rtl && rec[i].toks.length && kj.length && sp !== null) {
      if (typeof rec[i].toks[0][0] === "string" === chars) {
        const x = (boxes[i][2] + boxes[best][0]) / 2;
        rec[best].toks = [
          ...rec[i].toks,
          [sp, x, ...rec[i].toks[rec[i].toks.length - 1].slice(2)],
          ...kj,
        ];
      }
    }
    rec[best] = { text: `${t} ${rec[best].text.trim()}`, conf: rec[best].conf, toks: rec[best].toks };
    boxes[best] = [
      Math.min(boxes[i][0], boxes[best][0]),
      Math.min(boxes[i][1], boxes[best][1]),
      Math.max(boxes[i][2], boxes[best][2]),
      Math.max(boxes[i][3], boxes[best][3]),
    ];
    keep[i] = false;
  }
  const idx = [];
  for (let i = 0; i < keep.length; i++) if (keep[i]) idx.push(i);
  return { boxes: idx.map((i) => boxes[i]), rec: idx.map((i) => rec[i]) };
}

export class IndraOCR {
  /** Build the pipeline on a backend: { Tensor, meta: async () => object,
   *  json: async (name) => object, bytes: async (name) => Uint8Array,
   *  session: async (name) => { run, outputNames? } }.
   *  opts.{order, orient, cjk, lm} turn off the optional models (a missing
   *  file is also fine): cjk covers the script classifier and the group
   *  recognizers, lm the n-gram language models.  opts.adaptive toggles the
   *  second, higher-resolution detection pass; opts.math opts in to reading
   *  math lines as LaTeX (needs the math models in the model folder);
   *  opts.tableHeads builds tables from the order model's row / column / cell
   *  heads instead of from geometry; opts.furniture keeps page headers and
   *  footers in the markdown. */
  static async create(backend, opts = {}) {
    const o = new IndraOCR();
    o.Tensor = backend.Tensor;
    o.tableHeads = opts.tableHeads ?? false;
    o.furniture = opts.furniture ?? false;
    const meta = await backend.meta();
    o.vocab = Vocab.fromMeta(meta.vocab);
    o.longSide = meta.det_long_side;
    o.height = meta.rec_height;
    // rec.onnx may read a tall window (the line and the page around it);
    // the classifiers and group recognizers read the line band only.
    o.window = meta.rec_window ?? 1;
    o.det = await backend.session("det.onnx");
    o.rec = await backend.session(opts.recInt8 ? "rec.int8.onnx" : "rec.onnx");
    o.#pic = null; // the last page's picture channel, (logit map, scale)
    o.order = o.orient = null;
    if (opts.order !== false) {
      try {
        o.order = await backend.session("order.onnx");
      } catch {
        /* geometry-only reading order */
      }
    }
    if (opts.orient !== false) {
      try {
        o.orient = await backend.session("orient.onnx");
      } catch {
        /* no quarter-turn detection */
      }
    }
    // Scripts: a line classifier routes each line to its group's recognizer
    // (rec_<group>.onnx with meta["vocab_<group>"]), in script.onnx's class
    // order meta["script_groups"] (default GROUPS).
    o.script = null;
    o.groups = new Map();
    o.scriptGroups = meta.script_groups ?? GROUPS;
    if (opts.cjk !== false) {
      for (const key of Object.keys(meta)) {
        if (!key.startsWith("vocab_")) continue;
        const g = key.slice("vocab_".length);
        try {
          const sess = await backend.session(`rec_${g}.onnx`);
          o.groups.set(g, { sess, vocab: Vocab.fromMeta(meta[key]) });
        } catch {
          /* group recognizer missing from the model folder */
        }
      }
      if (o.groups.size) {
        try {
          o.script = await backend.session("script.onnx");
        } catch {
          o.script = null;
        }
      }
    }
    // Language models: lm.npz for rec.onnx, lm_<group>.npz for a group's
    // recognizer.  Each needs its recognizer's top-k outputs (newer exports);
    // meta.lm_params names the groups that ship one (and sets their alpha,
    // beta and gate).
    o.lms = new Map(); // session -> {lm, alpha, beta, gate}
    if (opts.lm !== false) {
      const params = meta.lm_params ?? {};
      const hasLmOuts = (sess) =>
        !!sess.outputNames &&
        ["blank", "topv", "topi"].every((n) => sess.outputNames.includes(n));
      const loadLm = async (name, sess) => {
        if (!hasLmOuts(sess)) return;
        try {
          const lm = await NgramLM.load(await backend.bytes(name === "base" ? "lm.npz" : `lm_${name}.npz`));
          const p = params[name] ?? {};
          o.lms.set(sess, {
            lm,
            alpha: p.alpha ?? LM_ALPHA,
            beta: p.beta ?? LM_BETA,
            gate: p.gate ?? LM_GATE,
          });
        } catch {
          /* no language model for this recognizer */
        }
      };
      await loadLm("base", o.rec);
      // A group recognizer with an lm_<group>.npz in the model folder gets one;
      // meta.lm_params only sets its alpha, beta and gate.
      for (const name of o.groups.keys()) {
        const g = o.groups.get(name);
        if (g) await loadLm(name, g.sess);
      }
    }
    // Math: a line classifier routes lines with math to a MathReader (math.mjs).
    o.math = null;
    if (opts.math) {
      if (backend.json) o.math = await MathReader.load(backend);
      if (!o.math)
        throw new Error(
          `math: true needs the math models (mathcls.onnx, math_enc.onnx, math_dec_init.onnx, math_dec_step.onnx, math_meta.json)`
        );
    }
    o.adaptive = opts.adaptive ?? true;
    o._pageRtl = false; // set by #assemble: the page reads right to left
    return o;
  }

  #pic;

  async #detectAt(gray, longSide) {
    const { x, ph, pw, scale } = pageToInput(gray, longSide);
    const { out } = await this.det.run({ x: new this.Tensor("float32", x, [1, 1, ph, pw]) });
    const [, c, h, w] = out.dims;
    this.#pic = c > 3 ? { logit: out.data.subarray(3 * h * w, 4 * h * w), w, h, scale } : null;
    return { boxes: decodeBoxes(out.data, h, w), scale };
  }

  /** Line boxes (x0, y0, x1, y1, score) of a page, in page pixels. */
  async detect(gray) {
    let { boxes, scale } = await this.#detectAt(gray, this.longSide);
    if (this.adaptive && boxes.length >= 8) {
      const medH = median(boxes.map((b) => b[3] - b[1]));
      if (medH < ADAPTIVE_MIN_H) {
        let ls = Math.floor(
          Math.min(ADAPTIVE_MAX_SIDE, (this.longSide * ADAPTIVE_TARGET_H) / Math.max(medH, 1))
        );
        ls = Math.floor(ls / 32) * 32;
        if (ls > this.longSide) ({ boxes, scale } = await this.#detectAt(gray, ls));
      }
    }
    for (const b of boxes) {
      b[0] = clamp(b[0] / scale, 0, gray.width);
      b[2] = clamp(b[2] / scale, 0, gray.width);
      b[1] = clamp(b[1] / scale, 0, gray.height);
      b[3] = clamp(b[3] / scale, 0, gray.height);
    }
    return boxes;
  }

  /** Text-line boxes and picture boxes [P, 4] (pixels; none from a det.onnx
   *  without the picture channel). */
  async detectAll(gray) {
    this.#pic = null;
    const boxes = await this.detect(gray);
    const pic = this.#pic;
    if (!pic) return [boxes, []];
    return [boxes, pictureBoxes(pic.logit, pic.w, pic.h, pic.scale, gray.width, gray.height)];
  }

  /** Per line crop, the group whose recognizer reads it (script.onnx), and
   *  whether the classifier is sure of it (probability over ROUTE_SURE); a
   *  group without its recognizer goes to base. */
  async #lineGroups(crops, batch = 64) {
    const names = crops.map(() => "base");
    const sure = crops.map(() => true);
    for (const part of widthBatches(crops.map((c) => c.w), batch)) {
      const { arr, w, widths } = padBatch(part.map((i) => crops[i]));
      const { p } = await this.script.run({
        x: new this.Tensor("float32", arr, [part.length, 1, this.height, w]),
        widths: new this.Tensor("int64", widths, [part.length]),
      });
      const multi = p.dims.length > 1; // [B] hanzi chance (old) vs [B, C] group odds
      const C = multi ? p.dims[1] : 1;
      part.forEach((i, k) => {
        let g;
        let pk;
        if (!multi) {
          const v = p.data[k];
          g = v > 0.5 ? "cjk" : "base";
          pk = Math.max(v, 1 - v);
        } else {
          let am = 0;
          for (let c = 1; c < C; c++) if (p.data[k * C + c] > p.data[k * C + am]) am = c;
          g = this.scriptGroups[am];
          pk = p.data[k * C + am];
        }
        names[i] = this.groups.has(g) ? g : "base";
        sure[i] = pk > ROUTE_SURE;
      });
    }
    return { names, sure };
  }

  /** Recognize the lines: [{text, conf, toks}] per box; tokens are
   *  [(vocab id | char, page x)] at the CTC peaks.  A line under MIN_CONF
   *  comes back empty.  vertical: the page is a vertical one turned a quarter
   *  turn (every line to the CJK recognizer). */
  async recognize(gray, boxes, batch = 64, vertical = false) {
    // Light-on-dark is decided once per line box, for both crops.
    const win = this.window;
    const dark = win > 1 ? boxes.map((b) => isDarkLine(gray, b)) : null;
    const crops = boxes.map((b, i) =>
      cropLine(gray, b, this.height, 4096, 1, dark ? dark[i] : null)
    );
    const tall =
      win > 1
        ? boxes.map((b, i) => cropLine(gray, b, this.height, 4096, win, dark ? dark[i] : null))
        : crops;
    const out = crops.map(() => ({ text: "", conf: 0, toks: [] }));
    let groups;
    if (vertical && this.groups.has("cjk")) {
      // A vertical page's lines are CJK columns, characters on their side;
      // the CJK recognizer reads those (and the Latin set across them).
      const { sess, vocab } = this.groups.get("cjk");
      groups = [[sess, vocab, crops.map((_, i) => i), true]];
    } else if (this.script && crops.length) {
      const { names, sure } = await this.#lineGroups(crops, batch);
      // A line the classifier is not sure about is read by base as well;
      // the reading with the higher confidence wins (below).
      groups = [
        [this.rec, this.vocab, names.flatMap((g, j) => (g === "base" || !sure[j] ? [j] : [])), false],
      ].concat(
        [...this.groups].map(([name, { sess, vocab }]) => [
          sess,
          vocab,
          names.flatMap((g, j) => (g === name ? [j] : [])),
          true,
        ])
      );
    } else {
      groups = [[this.rec, this.vocab, crops.map((_, i) => i), false]];
    }
    const bestConf = new Float64Array(crops.length).fill(-1);
    for (const [sess, vocab, sel, asChars] of groups) {
      if (!sel.length) continue;
      for (const part of widthBatches(sel.map((j) => crops[j].w), batch)) {
        const idx = part.map((i) => sel[i]);
        const src = tall === crops || sess !== this.rec ? crops : tall;
        const { arr, w: Wm, widths } = padBatch(idx.map((j) => src[j]));
        const lm = this.lms.get(sess);
        const H = arr.length / (part.length * Wm);
        const res = await sess.run({
          x: new this.Tensor("float32", arr, [part.length, 1, H, Wm]),
          widths: new this.Tensor("int64", widths, [part.length]),
        });
        const { ids, conf, lengths } = res;
        const hasLm = lm && res.blank && res.topv && res.topi;
        const T = ids.dims[1];
        const stride = Wm / Math.max(T, 1);
        for (let k = 0; k < idx.length; k++) {
          const j = idx[k];
          const n = Number(lengths.data[k]);
          const seq = Array.from({ length: n }, (_, t) => Number(ids.data[k * T + t]));
          const keep = [];
          for (let t = 0; t < n; t++)
            if (seq[t] !== 0 && (t === 0 || seq[t] !== seq[t - 1])) keep.push(t);
          const [x0, y0, x1, y1] = boxes[j];
          const h = Math.max(1, y1 - y0);
          const ix0 = Math.floor(x0 - PAD_X * h);
          const cw = Math.ceil(x1 + PAD_X * h) - ix0;
          const f = cw / crops[j].w;
          let kept = keep.map((t) => seq[t]);
          let steps = keep;
          const c = keep.length
            ? keep.reduce((s, t) => s + conf.data[k * T + t], 0) / keep.length
            : 0;
          if (hasLm && c >= MIN_CONF && c < lm.gate) {
            const Kb = res.topv.dims[2];
            [kept, steps] = beamSearch(
              res.blank.data.subarray(k * T, k * T + n),
              res.topv.data.subarray(k * T * Kb, k * T * Kb + n * Kb),
              res.topi.data.subarray(k * T * Kb, k * T * Kb + n * Kb),
              lm.lm,
              lm.alpha,
              lm.beta,
              LM_BEAM,
              0.999,
              LM_CAND_FLOOR
            );
          }
          const xs = steps.map((t) => ix0 + (t + 0.5) * stride * f);
          // A group recognizer's ids mean nothing in this.vocab: its tokens
          // are characters, like the text-layer tokens.
          const toks = asChars
            ? kept.map((id, i) => [vocab.chars[id - 1], xs[i]])
            : kept.map((id, i) => [id, xs[i]]);
          if (c <= bestConf[j]) continue;
          bestConf[j] = c;
          out[j] =
            c >= MIN_CONF
              ? { text: vocab.decode(kept), conf: c, toks }
              : { text: "", conf: c, toks: [] };
        }
      }
    }
    if (this.math && crops.length) {
      const flags = await this.math.flags(crops);
      const mathIdx = flags.flatMap((f, j) => (f ? [j] : []));
      if (mathIdx.length) {
        const reads = await this.math.read(gray, mathIdx.map((j) => boxes[j]));
        // Only a confident reading with math replaces the line (one token
        // holds the whole line: it is never split).
        mathIdx.forEach((j, k) => {
          const { text, conf } = reads[k];
          if (text.includes("$") && conf >= MATH_MIN_CONF) {
            const xc = (boxes[j][0] + boxes[j][2]) / 2;
            out[j] = { text, conf: 1.0, toks: [[text, xc]] };
          }
        });
      }
    }
    return out;
  }

  /** Reading order of the (split) lines with order.onnx: {order, layout},
   *  where layout = {cls, same, tab, classes} when the model has the layout
   *  heads (cls [n, C] block logits, same [n, n] same-block logits, tab
   *  [3, n, n] same-row / -column / -cell logits when tableHeads is on). */
  async readingOrder(boxes, texts, wh) {
    if (boxes.length < 2 || !this.order) {
      return { order: boxes.length ? xyCut(boxes) : [], layout: null };
    }
    const feats = lineFeatures(boxes, wh[0], wh[1], texts);
    const n = boxes.length;
    const outs = await this.order.run({
      feats: new this.Tensor("float32", feats, [1, n, 16]),
      mask: new this.Tensor("bool", new Uint8Array(n).fill(1), [1, n]),
    });
    const a = outs.a.data;
    const cls = outs.cls ? outs.cls.data : null;
    if (outs.same) {
      // Layout heads: blocks from the same-block head, lines inside a block by
      // geometry, blocks by the order head.
      return {
        order: decodeBlocksOrder(a, outs.same.data, boxes, cls),
        layout: {
          cls,
          same: outs.same.data,
          tab: this.tableHeads && outs.tab ? outs.tab.data : null,
          classes: BLOCK_CLASSES,
        },
      };
    }
    return { order: decodeOrder(a, n), layout: null };
  }

  #isSpace(tok) {
    return tok === " " || tok === this.vocab.index.get(" ");
  }

  #decodeToks(toks) {
    if (!toks.length) return "";
    if (typeof toks[0][0] === "string") {
      let text = toks
        .map((tk) => tk[0])
        .join("")
        .split(/\s+/)
        .filter(Boolean)
        .join(" ");
      if (toks[0].length === 2) {
        text = text.normalize("NFC"); // Hangul jamo -> syllables
        if (hasRtl(text)) text = flip(text, this._pageRtl || null);
      }
      return text;
    }
    return this.vocab.decode(toks.map((tk) => tk[0]));
  }

  /** Column gutters per line (see pipeline.py gutter_cuts). */
  #gutterCuts(boxes, rec) {
    const lefts = boxes.map((b) => b[0]);
    const rights = boxes.map((b) => b[2]);
    return boxes.map((b, i) => {
      const h = Math.max(1, b[3] - b[1]);
      const xs = rec[i].toks.filter((tk) => !this.#isSpace(tk[0])).map((tk) => tk[1]);
      const mine = [];
      for (let k = 1; k < xs.length; k++) {
        const a = xs[k - 1];
        const c = xs[k];
        if (c - a <= GUTTER_GAP * h) continue;
        const gl = a + 0.35 * h;
        const gr = c - 0.35 * h;
        let near = 0;
        for (let m = 0; m < boxes.length; m++) {
          if (m === i) continue;
          if (Math.abs(rights[m] - gl) < 0.5 * h || Math.abs(lefts[m] - gr) < 0.5 * h) near++;
        }
        if (near >= GUTTER_EDGES) mine.push((a + c) / 2);
      }
      return mine;
    });
  }

  /** Cut a line where two neighbouring characters are more than `gap` line
   *  heights apart (or at a column gutter in `at`). */
  #splitLine(box, toks, gap, at = []) {
    const h = Math.max(1, box[3] - box[1]);
    const segs = [[]];
    let prev = null;
    const ats = [...at].sort((p, q) => p - q);
    for (const tk of toks) {
      const x = tk[1];
      if (!this.#isSpace(tk[0])) {
        const cut =
          prev !== null && (x - prev > gap * h || ats.some((c) => prev < c && c < x));
        if (cut) segs.push([]);
        prev = x;
      }
      segs[segs.length - 1].push(tk);
    }
    if (segs.length === 1) return [[box.slice(0, 4), this.#decodeToks(toks), toks]];
    const out = [];
    for (const sg of segs) {
      const text = this.#decodeToks(sg).trim();
      const xs = sg.filter((tk) => !this.#isSpace(tk[0])).map((tk) => tk[1]);
      if (text && xs.length) {
        out.push([
          [
            Math.max(box[0], Math.min(...xs) - 0.35 * h),
            box[1],
            Math.min(box[2], Math.max(...xs) + 0.35 * h),
            box[3],
          ],
          text,
          sg,
        ]);
      }
    }
    return out;
  }

  /** Lines -> split at wide gaps -> tidy (drop leaders, join markers) ->
   *  reading order -> blocks + markdown.  vertical: the page is a vertical
   *  page turned a quarter turn. */
  async #assemble(wh, boxes, rec, t, vertical = false, pictures = null) {
    const t0 = Date.now();
    // A right-to-left page: lines read right to left (#decodeToks), and
    // reading order and markdown work on the page mirrored left to right, so
    // the first column is the right one.  Letter counts do not depend on order.
    let nRtl = 0;
    let letters = 0;
    for (const r of rec)
      for (const c of r.text)
        if (/\p{L}/u.test(c)) {
          letters++;
          if (isRtl(c)) nRtl++;
        }
    this._pageRtl = nRtl > letters - nRtl;
    const width = wh[0];
    const geoBox = (b) => (this._pageRtl ? [width - b[2], b[1], width - b[0], b[3]] : b);

    // Text-layer labels split a line at wide gaps; do the same (SPLIT_GAP).
    const nb = [];
    const nrec = [];
    const cuts = boxes.length ? this.#gutterCuts(boxes, rec) : [];
    boxes.forEach((b, i) => {
      for (const [bx, tx, tk] of this.#splitLine(b, rec[i].toks, SPLIT_GAP, cuts[i])) {
        nb.push(bx);
        nrec.push({ text: tx, conf: rec[i].conf, toks: tk });
      }
    });
    const tidied = tidyLines(nb, nrec, this._pageRtl, this.vocab.index.get(" ") ?? null);
    let rboxes = tidied.boxes;
    let rrec = tidied.rec;
    let texts = rrec.map((r) => r.text);
    let { order, layout } = await this.readingOrder(rboxes.map(geoBox), texts, wh);
    if (vertical && layout) {
      // The order model saw no vertical pages: turned, their columns of short
      // lines look like one table read row by row across tiers.  Blocks and
      // order come from XY-cut instead, without grid classes.
      const xblocks = xycutBlocks(rboxes);
      order = xblocks.flat();
      const nB = rboxes.length;
      const same = new Float32Array(nB * nB).fill(-1);
      for (const b of xblocks) for (const i of b) for (const j of b) same[i * nB + j] = 1;
      const C = layout.cls.length / nB;
      const cls = layout.cls.slice();
      for (const c of GRID_LABELS) for (let i = 0; i < nB; i++) cls[i * C + c] = -1e4;
      layout = { ...layout, cls, same, tab: null };
    } else if (layout && layout.tab) {
      // The table heads learned one line per cell, but the detector can join
      // cells: split table lines into cells, then run the order model again.
      const tableCls = BLOCK_CLASSES.indexOf("Table");
      const isTable = rboxes.map((_, i) => {
        let am = 0;
        for (let c = 1; c < BLOCK_CLASSES.length; c++)
          if (layout.cls[i * BLOCK_CLASSES.length + c] > layout.cls[i * BLOCK_CLASSES.length + am])
            am = c;
        return am === tableCls;
      });
      if (isTable.some(Boolean)) {
        const b2 = [];
        const r2 = [];
        rboxes.forEach((b, i) => {
          const parts = isTable[i]
            ? this.#splitLine(b, rrec[i].toks, CELL_GAP)
            : [[b, rrec[i].text, rrec[i].toks]];
          for (const [bx, tx, tk] of parts) {
            b2.push(bx);
            r2.push({ text: tx, conf: rrec[i].conf, toks: tk });
          }
        });
        if (b2.length !== rboxes.length) {
          rboxes = b2;
          rrec = r2;
          texts = rrec.map((r) => r.text);
          ({ order, layout } = await this.readingOrder(rboxes.map(geoBox), texts, wh));
        }
      }
    }
    t.orderMs = Date.now() - t0;
    t.totalMs = t.orientMs + t.detMs + t.recMs + t.orderMs;
    const note = vertical ? "order=xycut" : "order=model";
    const lines = order.map((i) => ({
      box: rboxes[i].slice(0, 4).map(round1),
      text: rrec[i].text,
      conf: round4(rrec[i].conf),
    }));
    let md = null;
    let blocks = [];
    if (layout) {
      blocks = pageBlocks(rboxes.map(geoBox), texts, order, layout.cls, layout.same);
      for (const b of blocks) b.bbox = geoBox(b.bbox).map(round1); // back on the page as it is
      // Inside tables, labels split cells at smaller gaps (CELL_GAP).
      const cellsMap = new Map(
        rboxes.map((b, i) => [
          i,
          this.#splitLine(b, rrec[i].toks, CELL_GAP).map(([bx, tx]) => [geoBox(bx), tx]),
        ])
      );
      const hasPics = pictures && pictures.length;
      md = pageMarkdown(
        rboxes.map(geoBox),
        texts,
        order,
        layout.cls,
        layout.same,
        {
          keepFurniture: this.furniture,
          cells: cellsMap,
          tab: this.tableHeads ? layout.tab : null,
          pictures: hasPics ? pictures.map(geoBox) : null,
        }
      );
      if (hasPics) {
        // picture blocks, in reading order by the same rule as the markdown
        for (const p of pictures) {
          const cy = (p[1] + p[3]) / 2;
          const at = blocks.findIndex(
            (b) =>
              (b.bbox[1] + b.bbox[3]) / 2 > cy &&
              Math.min(p[2], b.bbox[2]) > Math.max(p[0], b.bbox[0])
          );
          blocks.splice(at === -1 ? blocks.length : at, 0, {
            label: "Picture",
            bbox: [round1(p[0]), round1(p[1]), round1(p[2]), round1(p[3])],
            text: "",
          });
        }
      }
    } else if (rboxes.length < 2) md = texts.join("\n");
    return { lines, md, blocks, note };
  }

  /** uint8 ink map (255 = ink) of the page shrunk to `side` pixels: 8-bit
   *  images rotate faster than float ones. */
  #inkThumb(gray, side) {
    const { width: w, height: h } = gray;
    const s = side / Math.max(h, w);
    const nw = Math.max(1, roundHalfEven(w * s));
    const nh = Math.max(1, roundHalfEven(h * s));
    const small = resizeGray(gray.data, w, h, nw, nh, "area");
    const ink = new Uint8Array(nw * nh);
    for (let i = 0; i < ink.length; i++) ink[i] = 255 - small[i];
    return { data: ink, width: nw, height: nh };
  }

  /** How sharply the horizontal ink profile changes row to row, the page
   *  turned by `a` degrees: highest when the text rows are level. */
  #rowSharpness(ink, a) {
    const r = warpAffine(ink, rotMatrix2D(ink.width / 2, ink.height / 2, a));
    const w = ink.width;
    const h = ink.height;
    const p = new Float64Array(h);
    for (let y = 0; y < h; y++) {
      let s = 0;
      const row = y * w;
      for (let x = 0; x < w; x++) s += r.data[row + x];
      p[y] = s;
    }
    let sh = 0;
    for (let y = 1; y < h; y++) {
      const d = p[y] - p[y - 1];
      sh += d * d;
    }
    return sh;
  }

  /** Degrees (counter-clockwise) that straighten the text rows of a page: the
   *  angle whose horizontal ink profile changes most sharply row to row.
   *  Searched over +-DESKEW_MAX in 2 degree steps on a half-size thumbnail;
   *  the two best angles and 0 are then refined in 0.5 and 0.1 degree steps on
   *  the full thumbnail (`side` pixels), where small text rows are still
   *  resolved. */
  #estimateSkew(gray, side = 800) {
    const fine = this.#inkThumb(gray, side);
    let sum = 0;
    for (let i = 0; i < fine.data.length; i++) sum += fine.data[i];
    if (sum / fine.data.length < 0.003 * 255) return 0; // a blank page has no rows to straighten
    const coarse = this.#inkThumb(gray, Math.floor(side / 2));
    const grid = [];
    for (let i = 0; ; i++) {
      const a = -DESKEW_MAX + i * 2.0;
      if (a > DESKEW_MAX + 1e-6) break;
      grid.push(a);
    }
    const scores = grid.map((a) => this.#rowSharpness(coarse, a));
    let iA = -1;
    let iB = -1;
    for (let i = 0; i < grid.length; i++) {
      if (iA < 0 || scores[i] > scores[iA]) {
        iB = iA;
        iA = i;
      } else if (iB < 0 || scores[i] > scores[iB]) iB = i;
    }
    const starts = [0.0, grid[iB], grid[iA]];
    const near = [];
    const seen = new Set();
    for (const a0 of starts) {
      for (let k = 0; k <= 8; k++) {
        const a = a0 - 2.0 + k * 0.5;
        if (!seen.has(a)) {
          seen.add(a);
          near.push(a);
        }
      }
    }
    // The 0.5 degree ranges of nearby starts overlap: each angle is scored once.
    let a1 = near[0];
    let bestV = -Infinity;
    for (const a of near) {
      const v = this.#rowSharpness(fine, a);
      if (v > bestV) {
        bestV = v;
        a1 = a;
      }
    }
    const last = [];
    for (let k = 0; ; k++) {
      const a = a1 - 0.5 + k * 0.1;
      if (a > a1 + 0.5 + 1e-6) break;
      last.push(a);
    }
    let best = last[0];
    bestV = -Infinity;
    for (const a of last) {
      const v = this.#rowSharpness(fine, a);
      if (v > bestV) {
        bestV = v;
        best = a;
      }
    }
    return best;
  }

  /** The page shrunk once to at most PREVIEW_SIDE pixels a side (area
   *  averaging): the skew search and the orientation model read it instead of
   *  the full scan. */
  #preview(gray, side = PREVIEW_SIDE) {
    const s = side / Math.max(gray.height, gray.width);
    if (s >= 1) return gray;
    const nw = Math.max(1, roundHalfEven(gray.width * s));
    const nh = Math.max(1, roundHalfEven(gray.height * s));
    return { data: resizeGray(gray.data, gray.width, gray.height, nw, nh, "area"), width: nw, height: nh };
  }

  /** The page rotated by `degrees` counter-clockwise about its centre (same size). */
  #rotatePage(gray, degrees) {
    return warpAffine(gray, rotMatrix2D(gray.width / 2, gray.height / 2, degrees), "replicate");
  }

  /** A box of the page that #rotatePage(.., degrees) made, mapped back to the
   *  page before it: the bounding box of the four mapped corners, clipped. */
  #unrotateBox(box, degrees, w, h) {
    const m = rotMatrix2D(w / 2, h / 2, -degrees);
    const xs = [box[0], box[2], box[0], box[2]];
    const ys = [box[1], box[1], box[3], box[3]];
    const cx = xs.map((x, i) => m[0] * x + m[1] * ys[i] + m[2]);
    const cy = ys.map((y, i) => m[3] * xs[i] + m[4] * y + m[5]);
    return [
      round1(clamp(Math.min(...cx), 0, w)),
      round1(clamp(Math.min(...cy), 0, h)),
      round1(clamp(Math.max(...cx), 0, w)),
      round1(clamp(Math.max(...cy), 0, h)),
    ];
  }

  /** The page turned upright (`k` quarter turns, or orient.onnx's choice) and
   *  straightened (#estimateSkew): {gray, turns, skew} — turns of quarter-turns
   *  undone, degrees of skew undone. */
  async upright(gray, turn = true, k = null) {
    const a = this.#estimateSkew(gray);
    if (k === null)
      k = turn
        ? await this.orientation(Math.abs(a) >= DESKEW_MIN ? this.#rotatePage(gray, a) : gray)
        : 0;
    return this.#turn(gray, k, a, null);
  }

  /** `small`: the page's preview (#preview), for measuring the angle again. */
  #turn(gray, k, a, small) {
    // A half turn keeps the rows, so the angle stands; after a quarter turn the
    // rows were columns, so the angle is measured again on the turned page.
    const q = k === VERTICAL ? 1 : -k;
    if (k === VERTICAL) {
      // A quarter turn counter-clockwise: columns become lines, the first
      // (right) column on top, each read left to right.
      gray = rotQuarterCW(gray, 3);
    } else if (k) {
      gray = rotQuarterCW(gray, k);
    }
    if (k === 1 || k === 3 || k === VERTICAL) {
      const turned = small ? rotQuarterCW(small, q === 1 ? 3 : -q) : gray;
      a = this.#estimateSkew(turned);
    }
    if (Math.abs(a) >= DESKEW_MIN) gray = this.#rotatePage(gray, a);
    else a = 0;
    return { gray, turns: k, skew: a };
  }

  /** Quarter turns (counter-clockwise) the page content is turned by, from
   *  orient.onnx: 0 without the model or when no turn is more likely than
   *  ORIENT_MIN; VERTICAL for a page set in vertical columns. */
  async orientation(gray) {
    if (!this.orient) return 0;
    const side = ORIENT_SIDE;
    const h = gray.height;
    const w = gray.width;
    const s = side / Math.max(h, w);
    const nh = Math.max(1, roundHalfEven(h * s));
    const nw = Math.max(1, roundHalfEven(w * s));
    const small =
      nh === h && nw === w ? gray.data : resizeGray(gray.data, w, h, nw, nh, "area");
    const x = new Float32Array(side * side);
    for (let r = 0; r < nh; r++)
      for (let c = 0; c < nw; c++) x[r * side + c] = 1 - small[r * nw + c] / 255;
    const { p } = await this.orient.run({
      x: new this.Tensor("float32", x, [1, 1, side, side]),
    });
    let k = 0;
    for (let i = 1; i < p.data.length; i++) if (p.data[i] > p.data[k]) k = i;
    return p.data[k] > ORIENT_MIN ? k : 0;
  }

  /** OCR of a page image ({data: Uint8Array, width, height}, grayscale):
   *  {lines: [{box, text, conf}] in reading order, rotation, skew, vertical,
   *  markdown, blocks, pictures, text, note, timings}.  turn=false: the page
   *  is known to be upright (skips the orientation and deskew passes). */
  async run(gray, turn = true) {
    const t = {};
    let t0 = Date.now();
    const small = this.#preview(gray);
    const a = this.#estimateSkew(small);
    // The orientation model reads the page straightened by its row angle (a
    // tilted page could read as turned).
    const k = turn
      ? await this.orientation(Math.abs(a) >= DESKEW_MIN ? this.#rotatePage(small, a) : small)
      : 0;
    const orientMs = Date.now() - t0;
    const page = await this.#read(gray, k, a, small);
    page.timings.orientMs += orientMs;
    return page;
  }

  async #read(gray, k, a, small) {
    const t = {};
    let t0 = Date.now();
    const up = this.#turn(gray, k, a, small);
    gray = up.gray;
    t.orientMs = Date.now() - t0;
    t0 = Date.now();
    const [boxes, pictures] = await this.detectAll(gray);
    t.detMs = Date.now() - t0;
    t0 = Date.now();
    const vertical = up.turns === VERTICAL;
    const rec = await this.recognize(gray, boxes, 64, vertical);
    t.recMs = Date.now() - t0;
    const asm = await this.#assemble(
      [gray.width, gray.height],
      boxes,
      rec,
      t,
      vertical,
      vertical ? null : pictures
    );
    const rotation = vertical ? 0 : 90 * up.turns;
    const skew = round2(up.skew);
    if (up.skew) {
      // boxes back on the page as turned, before the deskew
      for (const ln of asm.lines)
        ln.box = this.#unrotateBox(ln.box, up.skew, gray.width, gray.height);
      for (const b of asm.blocks)
        b.bbox = this.#unrotateBox(b.bbox, up.skew, gray.width, gray.height);
    }
    if (vertical) {
      // back on the page as it is: (x, y) -> (W - y, x)
      const W = gray.height;
      const back = (b) => [W - b[3], b[0], W - b[1], b[2]];
      for (const ln of asm.lines) ln.box = back(ln.box);
      for (const b of asm.blocks) b.bbox = back(b.bbox);
    }
    return {
      lines: asm.lines,
      source: "ocr",
      rotation,
      skew,
      vertical,
      markdown: asm.md,
      blocks: asm.blocks,
      pictures: asm.blocks.filter((b) => b.label === "Picture").map((b) => b.bbox),
      text: asm.lines.map((l) => l.text).join("\n"),
      note: asm.note,
      timings: t,
    };
  }
}

// Picture regions [P, 4] in page pixels from det.onnx's picture channel (at
// half the detector's input scale `s`): connected regions above PICTURE_MIN,
// at least PICTURE_MIN_AREA of the page.
function pictureBoxes(logit, w, h, s, pageW, pageH) {
  const n = logit.length;
  // mask: logit > log(PICTURE_MIN / (1 - PICTURE_MIN)) = 0
  const mask = new Uint8Array(n);
  for (let i = 0; i < n; i++) mask[i] = logit[i] > 0 ? 1 : 0;
  // connected components (8-connectivity) with bounding boxes
  const labels = new Int32Array(n);
  const stats = [];
  const stack = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    if (!mask[i] || labels[i]) continue;
    const lab = stats.length + 1;
    let sp = 0;
    stack[sp++] = i;
    labels[i] = lab;
    const st = { x0: w, x1: -1, y0: h, y1: -1 };
    while (sp > 0) {
      const j = stack[--sp];
      const r = (j / w) | 0;
      const c = j - r * w;
      if (c < st.x0) st.x0 = c;
      if (c > st.x1) st.x1 = c;
      if (r < st.y0) st.y0 = r;
      if (r > st.y1) st.y1 = r;
      for (let dr = -1; dr <= 1; dr++) {
        const rr = r + dr;
        if (rr < 0 || rr >= h) continue;
        for (let dc = -1; dc <= 1; dc++) {
          const cc = c + dc;
          if (cc < 0 || cc >= w) continue;
          const q = rr * w + cc;
          if (mask[q] && !labels[q]) {
            labels[q] = lab;
            stack[sp++] = q;
          }
        }
      }
    }
    stats.push(st);
  }
  const f = 2.0 / s;
  const minArea = PICTURE_MIN_AREA * pageH * pageW;
  const out = [];
  for (const st of stats) {
    const b = [
      st.x0 * f,
      st.y0 * f,
      Math.min((st.x1 + 1) * f, pageW),
      Math.min((st.y1 + 1) * f, pageH),
    ];
    if ((b[2] - b[0]) * (b[3] - b[1]) >= minArea) out.push(b);
  }
  return out;
}
