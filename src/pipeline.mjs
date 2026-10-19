// The indra-ocr pipeline, ported from indra_ocr/pipeline.py and friends.
// Pure ESM (no Node or browser APIs): the ONNX Runtime sessions and the
// Tensor class come from an injected backend — src/index.mjs builds one on
// onnxruntime-web (wasm).
//
// Not ported here (the model is optional in the Python package and missing
// from the current model revision): mask.onnx (CenterMask line masking).
import { pageToInput, decodeBoxes } from "./det.mjs";
import { cropLine, padBatch, widthBatches, PAD_X } from "./rec.mjs";
import { lineFeatures, decodeBlocksOrder, decodeOrder, xyCut, BLOCK_CLASSES } from "./order.mjs";
import { MATH_MIN_CONF, MathReader } from "./math.mjs";
import { pageBlocks, pageMarkdown } from "./layout.mjs";
import { Vocab } from "./vocab.mjs";
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
const MIN_CONF = 0.5;
const SPLIT_GAP = 3.0;
const CELL_GAP = 1.8; // cut a table line into cells at a wider gap (label rule in tables)
const GUTTER_GAP = 1.6;
const GUTTER_EDGES = 3;
const DESKEW_MAX = 5.0; // degrees of tilt searched
const DESKEW_MIN = 0.3; // smaller tilts are left alone (the search step is 0.1)

const round1 = (v) => Math.round(v * 10) / 10;
const round4 = (v) => Math.round(v * 1e4) / 1e4;

export class IndraOCR {
  /** Build the pipeline on a backend: { Tensor, meta: async () => object,
   *  json: async (name) => object, session: async (name) => { run } }.
   *  opts.{order, orient, cjk} turn off the optional models (a missing file is
   *  also fine); opts.adaptive toggles the second, higher-resolution detection
   *  pass; opts.math opts in to reading math lines as LaTeX (needs the math
   *  models in the model folder); opts.tableHeads builds tables from the order
   *  model's row / column / cell heads instead of from geometry; opts.furniture
   *  keeps page headers and footers in the markdown. */
  static async create(backend, opts = {}) {
    const o = new IndraOCR();
    o.Tensor = backend.Tensor;
    o.tableHeads = opts.tableHeads ?? false;
    o.furniture = opts.furniture ?? false;
    const meta = await backend.meta();
    o.vocab = Vocab.fromMeta(meta.vocab);
    o.longSide = meta.det_long_side;
    o.height = meta.rec_height;
    o.det = await backend.session("det.onnx");
    o.rec = await backend.session("rec.onnx");
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
    // Chinese: a line classifier routes lines with hanzi to their own recognizer.
    o.script = o.recCjk = o.vocabCjk = null;
    if (opts.cjk !== false) {
      try {
        o.script = await backend.session("script.onnx");
        o.recCjk = await backend.session("rec_cjk.onnx");
        o.vocabCjk = Vocab.fromMeta(meta.vocab_cjk);
      } catch {
        /* Latin-only model folder */
      }
    }
    // Math: a line classifier routes lines with math to a MathReader (math.mjs).
    o.math = null;
    if (opts.math) {
      if (backend.json) o.math = await MathReader.load(backend);
      if (!o.math) throw new Error(`math: true needs the math models (mathcls.onnx, math_enc.onnx, math_dec_init.onnx, math_dec_step.onnx, math_meta.json)`);
    }
    o.adaptive = opts.adaptive ?? true;
    return o;
  }

  async #detectAt(gray, longSide) {
    const { x, ph, pw, scale } = pageToInput(gray, longSide);
    const { out } = await this.det.run({ x: new this.Tensor("float32", x, [1, 1, ph, pw]) });
    const [, , h, w] = out.dims;
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

  /** Per line crop: does it hold Chinese characters (script.onnx)? */
  async #hanziLines(crops) {
    const out = new Array(crops.length).fill(false);
    for (const part of widthBatches(crops.map((c) => c.w))) {
      const { arr, w, widths } = padBatch(part.map((i) => crops[i]));
      const { p } = await this.script.run({
        x: new this.Tensor("float32", arr, [part.length, 1, this.height, w]),
        widths: new this.Tensor("int64", widths, [part.length]),
      });
      part.forEach((i, k) => (out[i] = p.data[k] > 0.5));
    }
    return out;
  }

  /** Recognize the lines: [(text, conf, tokens)] per box; tokens are
   *  [(vocab id | char, page x)] at the CTC peaks. */
  async recognize(gray, boxes, batch = 64) {
    const crops = boxes.map((b) => cropLine(gray, b, this.height));
    const out = crops.map(() => ({ text: "", conf: 0, toks: [] }));
    let groups = [[this.rec, this.vocab, crops.map((_, i) => i), false]];
    if (this.script) {
      const hanzi = await this.#hanziLines(crops);
      groups = [
        [this.rec, this.vocab, hanzi.flatMap((h, i) => (h ? [] : [i])), false],
        [this.recCjk, this.vocabCjk, hanzi.flatMap((h, i) => (h ? [i] : [])), true],
      ];
    }
    for (const [sess, vocab, sel, asChars] of groups) {
      if (!sel.length) continue;
      for (const part of widthBatches(sel.map((j) => crops[j].w), batch)) {
        const { arr, w: Wm, widths } = padBatch(part.map((j) => crops[j]));
        const { ids, conf, lengths } = await sess.run({
          x: new this.Tensor("float32", arr, [part.length, 1, this.height, Wm]),
          widths: new this.Tensor("int64", widths, [part.length]),
        });
        const T = ids.dims[1];
        const stride = Wm / Math.max(T, 1);
        part.forEach((j, k) => {
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
          const xs = keep.map((t) => ix0 + (t + 0.5) * stride * f);
          const kept = keep.map((t) => seq[t]);
          const toks = asChars
            ? kept.map((id, i) => [vocab.chars[id - 1], xs[i]])
            : kept.map((id, i) => [id, xs[i]]);
          const c =
            keep.reduce((s, t) => s + conf.data[k * T + t], 0) / Math.max(keep.length, 1);
          out[j] =
            c >= MIN_CONF
              ? { text: vocab.decode(kept), conf: c, toks }
              : { text: "", conf: c, toks: [] };
        });
      }
    }
    if (this.math && crops.length) {
      const flags = await this.math.flags(crops);
      const mathIdx = flags.flatMap((f, j) => (f ? [j] : []));
      if (mathIdx.length) {
        const reads = await this.math.read(
          gray,
          mathIdx.map((j) => boxes[j])
        );
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
    if (typeof toks[0][0] === "string")
      return toks.map((tk) => tk[0]).join("").split(/\s+/).filter(Boolean).join(" ");
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

  /** Lines -> split at wide gaps -> reading order -> blocks + markdown. */
  async #assemble(wh, boxes, rec, t) {
    const t0 = Date.now();
    const cuts = this.#gutterCuts(boxes, rec);
    let nb = [];
    let nrec = [];
    boxes.forEach((b, i) => {
      for (const [bx, tx, tk] of this.#splitLine(b, rec[i].toks, SPLIT_GAP, cuts[i])) {
        nb.push(bx);
        nrec.push({ text: tx, conf: rec[i].conf, toks: tk });
      }
    });
    const texts = nrec.map((r) => r.text);
    let { order, layout } = await this.readingOrder(nb, texts, wh);
    if (layout && layout.tab) {
      // The table heads learned one line per cell, but the detector can join
      // cells: split table lines into cells, then run the order model again.
      const tableCls = BLOCK_CLASSES.indexOf("Table");
      const isTable = nb.map((_, i) => {
        let am = 0;
        for (let c = 1; c < BLOCK_CLASSES.length; c++)
          if (layout.cls[i * BLOCK_CLASSES.length + c] > layout.cls[i * BLOCK_CLASSES.length + am]) am = c;
        return am === tableCls;
      });
      if (isTable.some(Boolean)) {
        const b2 = [];
        const r2 = [];
        nb.forEach((b, i) => {
          const parts = isTable[i]
            ? this.#splitLine(b, nrec[i].toks, CELL_GAP)
            : [[b, nrec[i].text, nrec[i].toks]];
          for (const [bx, tx, tk] of parts) {
            b2.push(bx);
            r2.push({ text: tx, conf: nrec[i].conf, toks: tk });
          }
        });
        if (b2.length !== nb.length) {
          nb = b2;
          nrec = r2;
          const texts2 = nrec.map((r) => r.text);
          ({ order, layout } = await this.readingOrder(nb, texts2, wh));
          texts.length = 0;
          texts.push(...texts2);
        }
      }
    }
    t.orderMs = Date.now() - t0;
    const lines = order.map((i) => ({
      box: nb[i].slice(0, 4).map(round1),
      text: nrec[i].text,
      conf: round4(nrec[i].conf),
    }));
    let md = null;
    let blocks = [];
    if (layout) {
      blocks = pageBlocks(nb, texts, order, layout.cls, layout.same);
      // Inside tables, labels split cells at smaller gaps (CELL_GAP).
      const cellsMap = new Map(
        nb.map((b, i) => [
          i,
          this.#splitLine(b, nrec[i].toks, CELL_GAP).map(([bx, tx]) => [bx, tx]),
        ])
      );
      md = pageMarkdown(nb, texts, order, layout.cls, layout.same, {
        keepFurniture: this.furniture,
        cells: cellsMap,
        tab: layout.tab,
      });
    } else if (nb.length < 2) md = texts.join("\n");
    return { lines, md, blocks };
  }

  /** Degrees (counter-clockwise) that straighten the text rows of a page: the
   *  angle whose horizontal ink profile changes most sharply row to row, on a
   *  page thumbnail, searched in 0.5 degree steps and then 0.1 degree steps. */
  #estimateSkew(gray, side = 800) {
    const { width: w, height: h } = gray;
    const s = side / Math.max(h, w);
    const nw = Math.max(1, roundHalfEven(w * s));
    const nh = Math.max(1, roundHalfEven(h * s));
    const small = resizeGray(gray.data, w, h, nw, nh, "area");
    const ink = new Float32Array(nw * nh);
    let sum = 0;
    for (let i = 0; i < ink.length; i++) {
      ink[i] = (255 - small[i]) / 255;
      sum += ink[i];
    }
    if (sum / ink.length < 0.003) return 0; // a blank page has no rows to straighten
    const sharpness = (a) => {
      const r = warpAffine({ data: ink, width: nw, height: nh }, rotMatrix2D(nw / 2, nh / 2, a));
      let sh = 0;
      for (let y = 1; y < nh; y++) {
        let d = 0;
        for (let x = 0; x < nw; x++) d += r.data[y * nw + x] - r.data[(y - 1) * nw + x];
        sh += d * d;
      }
      return sh;
    };
    let best = 0;
    let bestV = -Infinity;
    for (let a = -DESKEW_MAX; a <= DESKEW_MAX + 1e-6; a += 0.5) {
      const v = sharpness(a);
      if (v > bestV) {
        bestV = v;
        best = a;
      }
    }
    bestV = -Infinity;
    for (let a = best - 0.5; a <= best + 0.5 + 1e-6; a += 0.1) {
      const v = sharpness(a);
      if (v > bestV) {
        bestV = v;
        best = a;
      }
    }
    return best;
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

  /** The page turned upright (orient.onnx) and straightened (estimateSkew):
   *  {gray, turns, skew} — turns of quarter-turns undone, degrees of skew undone. */
  async #upright(gray, turn = true) {
    const k = turn ? await this.orientation(gray) : 0;
    if (k) gray = rotQuarterCW(gray, k);
    let a = this.#estimateSkew(gray);
    if (Math.abs(a) >= DESKEW_MIN) gray = this.#rotatePage(gray, a);
    else a = 0;
    return { gray, turns: k, skew: a };
  }

  /** Quarter turns (counter-clockwise) the page content is turned by. */
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
   *  {lines: [{box, text, conf}] in reading order, rotation, skew, markdown,
   *  blocks, text}.  turn=false: the page is known to be upright (skips the
   *  orientation and deskew passes). */
  async run(gray, turn = true) {
    const t = {};
    let t0 = Date.now();
    const up = await this.#upright(gray, turn);
    gray = up.gray;
    t.orientMs = Date.now() - t0;
    t0 = Date.now();
    const boxes = await this.detect(gray);
    t.detMs = Date.now() - t0;
    t0 = Date.now();
    const rec = await this.recognize(gray, boxes);
    t.recMs = Date.now() - t0;
    const { lines, md, blocks } = await this.#assemble([gray.width, gray.height], boxes, rec, t);
    const skew = round1(up.skew);
    if (up.skew) {
      // boxes back on the page as turned, before the deskew
      for (const ln of lines) ln.box = this.#unrotateBox(ln.box, up.skew, gray.width, gray.height);
      for (const b of blocks) b.bbox = this.#unrotateBox(b.bbox, up.skew, gray.width, gray.height);
    }
    return {
      lines,
      rotation: 90 * up.turns,
      skew,
      markdown: md,
      blocks,
      text: lines.map((l) => l.text).join("\n"),
      timings: t,
    };
  }
}
