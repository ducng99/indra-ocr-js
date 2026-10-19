// Math lines: a classifier finds lines that hold math, and MathReader reads
// them as text with inline LaTeX ("we have $x _ { i } \\leq 1$").
// Ported from indra_ocr/math.py.
//
// The CTC recognizer reads one symbol per column, so it cannot write a fraction
// or a sum with limits.  MathReader reads a taller crop of the line with an
// attention decoder, one token per step, with the decoder's keys and values
// cached.
//
// Models (all optional; the pipeline skips math without them): mathcls.onnx,
// math_enc.onnx, math_dec_init.onnx, math_dec_step.onnx and math_meta.json.
import { cropLine, padBatch, widthBatches } from "./rec.mjs";

export const MATH_MIN = 0.5; // classifier probability above which a line is read as math
// A math reading replaces a line only at this mean token probability or more: on
// print the model has not seen (old scans) it writes confident-looking LaTeX
// that is wrong, at low probability.
export const MATH_MIN_CONF = 0.9;
// A line whose mean token probability is this far below MATH_MIN_CONF after 8
// tokens stops decoding: it would be rejected anyway (garbage on unseen print).
export const EARLY_STOP_MARGIN = 0.1;
export const BOS = 1;
export const EOS = 2;
export const N_SPECIAL = 4;
const FILES = ["mathcls.onnx", "math_enc.onnx", "math_dec_init.onnx", "math_dec_step.onnx"];

// Tokens -> text; inside math (between "$" tokens) they are joined by spaces.
export function detokenize(tokens) {
  const out = [];
  const math = [];
  let inMath = false;
  for (const t of tokens) {
    if (t === "$") {
      if (inMath) {
        out.push("$" + math.join(" ") + "$");
        math.length = 0;
      }
      inMath = !inMath;
    } else if (inMath) math.push(t);
    else out.push(t);
  }
  if (inMath && math.length) out.push("$" + math.join(" ") + "$");
  return out.join("").split(/\s+/).filter(Boolean).join(" ");
}

// Skip feeds the graph does not take: exports drop unused inputs (the init
// graph does not read the mask), like Python's `{k: v for k, v in feed.items()
// if k in names}`.  A session without inputNames (a custom backend) gets
// everything.
const feed = (sess, inputs) => {
  if (!sess.inputNames) return inputs;
  const o = {};
  for (const k in inputs) if (sess.inputNames.includes(k)) o[k] = inputs[k];
  return o;
};

export class MathReader {
  /** sessions: {cls, enc, init, step} as built by a backend; meta: math_meta.json. */
  constructor(Tensor, sessions, meta) {
    this.Tensor = Tensor;
    this.cls = sessions.cls;
    this.enc = sessions.enc;
    this.init = sessions.init;
    this.step = sessions.step;
    const cfg = meta.config;
    this.height = cfg.height;
    this.maxTokens = cfg.max_tokens;
    this.layers = cfg.dec_layers;
    this.heads = cfg.dec_heads;
    this.headDim = cfg.channels / cfg.dec_heads;
    this.itos = ["", "", "", "", ...meta.tokens];
  }

  /** The MathReader for a backend's model folder, or null when the math files
   *  are missing from it (Python: MathReader.load). */
  static async load(backend) {
    let meta;
    try {
      meta = await backend.json("math_meta.json");
    } catch {
      return null;
    }
    const named = {};
    for (const f of FILES) named[f] = await backend.session(f);
    const { "mathcls.onnx": cls, "math_enc.onnx": enc, "math_dec_init.onnx": init, "math_dec_step.onnx": step } = named;
    return new MathReader(backend.Tensor, { cls, enc, init, step }, meta);
  }

  /** Per line crop (the recognizer's crop): does it hold math? */
  async flags(crops, batch = 64) {
    const out = new Array(crops.length).fill(false);
    for (const part of widthBatches(crops.map((c) => c.w), batch)) {
      const sel = part.map((i) => crops[i]);
      const { arr, w, widths } = padBatch(sel);
      const H = sel[0].ink.length / sel[0].w; // the recognizer's crop height
      const { p } = await this.cls.run(
        feed(this.cls, {
          x: new this.Tensor("float32", arr, [part.length, 1, H, w]),
          widths: new this.Tensor("int64", widths, [part.length]),
        })
      );
      part.forEach((i, k) => (out[i] = p.data[k] > MATH_MIN));
    }
    return out;
  }

  /** (text with inline LaTeX, mean token probability) for each box. */
  async read(gray, boxes, batch = 64) {
    const crops = boxes.map((b) => cropLine(gray, b, this.height));
    const out = crops.map(() => ({ text: "", conf: 0 }));
    // Few, wide batches: each batch is one decoding loop of up to max_tokens
    // steps, and a step costs about the same for 8 lines as for 40.
    for (const part of widthBatches(crops.map((c) => c.w), batch, 4.0, 64, 16 * 64 * 640)) {
      const sel = part.map((i) => crops[i]);
      const { arr, w: Wm, widths } = padBatch(sel);
      const B = part.length;
      const T = this.Tensor;
      const { memory, mask } = await this.enc.run(
        feed(this.enc, {
          x: new T("float32", arr, [B, 1, this.height, Wm]),
          widths: new T("int64", widths, [B]),
        })
      );
      const { cross_k: ck0, cross_v: cv0 } = await this.init.run(feed(this.init, { memory }));
      let sk = { data: new Float32Array(0), dims: [this.layers, B, this.heads, 0, this.headDim] };
      let sv = { data: new Float32Array(0), dims: sk.dims };
      let tok = new BigInt64Array(B).fill(BigInt(BOS));
      const seqs = Array.from({ length: B }, () => []);
      const probs = Array.from({ length: B }, () => []);
      const done = new Uint8Array(B);
      for (let t = 0; t < this.maxTokens - 1; t++) {
        const res = await this.step.run(
          feed(this.step, {
            token: new T("int64", tok, [B]),
            pos: new T("int64", BigInt64Array.from([BigInt(t)]), []),
            self_k: new T("float32", sk.data, sk.dims),
            self_v: new T("float32", sv.data, sv.dims),
            cross_k: ck0,
            cross_v: cv0,
            mask,
          })
        );
        const { logits, new_k: nk, new_v: nv } = res;
        sk = { data: nk.data, dims: nk.dims };
        sv = { data: nv.data, dims: nv.dims };
        const V = logits.dims[1];
        for (let b = 0; b < B; b++) {
          const row = b * V;
          let mx = -Infinity;
          let am = 0;
          for (let v = 0; v < V; v++) {
            const l = logits.data[row + v];
            if (l > mx) {
              mx = l;
              am = v;
            }
          }
          tok[b] = BigInt(am);
          if (done[b]) continue;
          let se = 0;
          for (let v = 0; v < V; v++) se += Math.exp(logits.data[row + v] - mx);
          const top = 1 / se; // softmax probability of the argmax token
          probs[b].push(top);
          if (am === EOS) done[b] = true;
          else if (
            probs[b].length >= 8 &&
            probs[b].reduce((s, v) => s + v, 0) / probs[b].length <
              MATH_MIN_CONF - EARLY_STOP_MARGIN
          ) {
            done[b] = true; // rejected: keeps a low mean probability
          } else if (am >= N_SPECIAL) seqs[b].push(am);
        }
        if (done.every(Boolean)) break;
      }
      part.forEach((j, k) => {
        out[j] = {
          text: detokenize(seqs[k].map((i) => this.itos[i])),
          conf: probs[k].length ? probs[k].reduce((s, v) => s + v, 0) / probs[k].length : 0,
        };
      });
    }
    return out;
  }
}
