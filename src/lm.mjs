// Beam search with a token n-gram language model, for lines the recognizer is
// unsure of.  Ported from indra_ocr/lm.py.
//
// The recognizer reads greedily: one best class per step.  On a line whose
// confidence is under the gate, a CTC prefix beam search keeps several readings
// and the language model breaks close calls between them:
// score = log P_ctc + alpha * log P_lm + beta * tokens.  The recognizer's own
// probabilities still decide, so a clearly written odd word survives.
//
// lm.npz holds the model as a backoff table (no pickle): per history length n,
// sorted keys of the (history, token) pairs seen in training with their
// log-probabilities, and sorted history keys with their backoff weights.  A key
// packs tokens `bits` bits each (0 = line start, t + 1 = token t): 11 unless
// the file says more (a larger vocab), with order x bits at most TAG_SHIFT.
export const BITS = 11; // per token, unless lm.npz holds "bits"
export const TAG_SHIFT = 61; // the history length goes in the top 3 bits
export const BOS = -1;

const pack = (tokens, bits) => {
  let k = 0n;
  for (const t of tokens) k = (k << BigInt(bits)) | BigInt(t === BOS ? 0 : t + 1);
  return k;
};

// ---------------------------------------------------------------------------
// npz / npy reading: np.savez_compressed is a zip of little-endian .npy files.
// ---------------------------------------------------------------------------

async function inflateRaw(bytes) {
  const ds = new DecompressionStream("deflate-raw");
  const stream = new Blob([bytes]).stream().pipeThrough(ds);
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

// The named arrays of an .npz file: Map(name -> Uint8Array of the .npy body).
async function readNpz(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (dv.getUint32(0, true) !== 0x04034b50)
    throw new Error("lm file is not an .npz (zip) archive");
  // End of central directory: find from the back (a comment may follow it).
  let eocd = -1;
  for (let i = bytes.length - 22; i >= Math.max(0, bytes.length - 22 - 65536); i--) {
    if (dv.getUint32(i, true) === 0x06054b50) {
      eocd = i;
      break;
    }
  }
  if (eocd < 0) throw new Error("lm file: no zip central directory");
  const count = dv.getUint16(eocd + 10, true);
  let at = dv.getUint32(eocd + 16, true);
  const out = new Map();
  for (let e = 0; e < count; e++) {
    if (dv.getUint32(at, true) !== 0x02014b50) throw new Error("lm file: bad zip entry");
    const method = dv.getUint16(at + 10, true);
    const csize = dv.getUint32(at + 20, true);
    if (csize === 0xffffffff) throw new Error("lm file: zip64 is not supported");
    const nameLen = dv.getUint16(at + 28, true);
    const extraLen = dv.getUint16(at + 30, true);
    const commentLen = dv.getUint16(at + 32, true);
    const local = dv.getUint32(at + 42, true);
    const name = new TextDecoder().decode(bytes.subarray(at + 46, at + 46 + nameLen));
    // The local header repeats the name / extra lengths: the data starts after both.
    const lNameLen = dv.getUint16(local + 26, true);
    const lExtraLen = dv.getUint16(local + 28, true);
    const start = local + 30 + lNameLen + lExtraLen;
    const raw = bytes.slice(start, start + csize);
    out.set(name.replace(/\.npy$/, ""), method === 0 ? raw : await inflateRaw(raw));
    at += 46 + nameLen + extraLen + commentLen;
  }
  return out;
}

// One .npy payload: {descr, shape} plus a view of the data (little-endian only,
// like every model file this package ships).
function readNpy(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes[0] !== 0x93 || bytes[1] !== 0x4e) throw new Error("lm table is not an .npy file");
  const major = bytes[6];
  let off = 8;
  const hdrLen = major < 3 ? dv.getUint16(off, true) : dv.getUint32(off, true);
  off += major < 3 ? 2 : 4;
  const header = new TextDecoder().decode(bytes.subarray(off, off + hdrLen));
  const descr = /'descr':\s*'([^']+)'/.exec(header)[1];
  const shape = /'shape':\s*\(([^)]*)\)/
    .exec(header)[1]
    .split(",")
    .map((s) => parseInt(s, 10))
    .filter((n) => !Number.isNaN(n));
  off += hdrLen;
  const n = shape.reduce((a, b) => a * b, 1);
  if (descr === "|u1") return { descr, shape, data: bytes.slice(off) };
  const body = bytes.slice(off); // aligned copy for typed-array views
  if (descr === "<u8") return { descr, shape, data: new BigUint64Array(body.buffer, 0, n) };
  if (descr === "<f4") return { descr, shape, data: new Float32Array(body.buffer, 0, n) };
  if (descr === "<i8") return { descr, shape, data: new BigInt64Array(body.buffer, 0, n) };
  throw new Error(`lm table: unsupported dtype ${descr}`);
}

// ---------------------------------------------------------------------------
// The model
// ---------------------------------------------------------------------------

export class NgramLM {
  /** bytes: the contents of lm.npz (as fetched). */
  static async load(bytes) {
    const z = await readNpz(bytes);
    const arr = (name) => readNpy(z.get(name)).data;
    const lm = new NgramLM();
    lm.order = Number(arr("order")[0]);
    lm.vocabSize = Number(arr("vocab_size")[0]);
    lm.bits = z.has("bits") ? Number(arr("bits")[0]) : BITS;
    if (lm.order * lm.bits > TAG_SHIFT)
      throw new Error(`order ${lm.order} x ${lm.bits} bits do not fit a key`);
    // All history lengths in one sorted table, the length in the top bits
    // (keys use at most order x bits), so a step needs one search per table.
    // Keys are split hi / lo (26 low bits) for Number-speed in the searches;
    // hi fits a double exactly (at most 2^38).
    const concat = (tables) => {
      let n = 0;
      for (const t of tables) n += t.length;
      const hi = new Float64Array(n);
      const lo = new Float64Array(n);
      const vals = new Float64Array(n);
      let at = 0;
      for (const t of tables) {
        for (let i = 0; i < t.length; i++) {
          hi[at + i] = Number(t[i] >> 26n);
          lo[at + i] = Number(t[i] & 0x3ffffffn);
        }
        at += t.length;
      }
      return { hi, lo };
    };
    const tag = (n) => BigInt(n) << BigInt(TAG_SHIFT);
    const k = [];
    const v = [];
    const hk = [];
    const hb = [];
    for (let n = 0; n < lm.order; n++) {
      k.push(arr(`k${n}`));
      v.push(arr(`v${n}`));
      hk.push(arr(`hk${n}`));
      hb.push(arr(`hb${n}`));
    }
    const merge = (keys, vals, tagN) => {
      // keys | tag, values concatenated, sorted by the tagged key
      const n = keys.reduce((a, t) => a + t.length, 0);
      const hi = new Float64Array(n);
      const lo = new Float64Array(n);
      const vs = new Float64Array(n);
      let at = 0;
      for (let t = 0; t < keys.length; t++) {
        const tagBig = tagN(t);
        const kk = keys[t];
        for (let i = 0; i < kk.length; i++) {
          const key = kk[i] | tagBig;
          hi[at] = Number(key >> 26n);
          lo[at] = Number(key & 0x3ffffffn);
          vs[at] = vals[t][i];
          at++;
        }
      }
      return { hi, lo, vs };
    };
    lm.p = merge(k, v, tag);
    lm.h = merge(hk, hb, tag);
    lm.hb = hb;
    lm.ikcache = new Map();
    return lm;
  }

  // An upper bound of every value logpKeys returns: the largest
  // log-probability plus every positive backoff weight a back-off can add.
  get maxLogp() {
    if (this._maxLogp === undefined) {
      let top = -Math.log(this.vocabSize);
      for (let i = 0; i < this.p.vs.length; i++)
        if (this.p.vs[i] > top) top = this.p.vs[i];
      let bo = 0;
      for (const h of this.hb) {
        let mx = -Infinity;
        for (let i = 0; i < h.length; i++) if (h[i] > mx) mx = h[i];
        if (mx > 0) bo += mx;
      }
      this._maxLogp = top + bo;
    }
    return this._maxLogp;
  }

  // Binary search in the hi/lo split table: first index with key >= (hi, lo).
  static #find(t, hi, lo) {
    let a = 0;
    let b = t.hi.length;
    while (a < b) {
      const mid = (a + b) >> 1;
      const mh = t.hi[mid];
      if (mh < hi || (mh === hi && t.lo[mid] < lo)) a = mid + 1;
      else b = mid;
    }
    return a;
  }

  static #hit(t, i, hi, lo) {
    return i < t.hi.length && t.hi[i] === hi && t.lo[i] === lo;
  }

  // logp for many (history, token) pairs, each given as one key: the last
  // order - 1 history tokens and then the token, packed (`pack`).  Shorter
  // histories are the key's low bits, so every history length is one search.
  // This is the vectorized `logp_keys` of the Python module, per key: the
  // value comes from the longest seen history n <= m that saw the token, plus
  // the backoff weights of the longer seen histories (n, m]; with no such n,
  // all backoffs and a uniform floor.
  logpKeys(keys) {
    const out = new Array(keys.length);
    const miss = [];
    for (let r = 0; r < keys.length; r++) {
      const v = this.ikcache.get(keys[r]);
      if (v === undefined) miss.push(r);
      else out[r] = v;
    }
    if (!miss.length) return out;
    const N = this.order;
    const bits = BigInt(this.bits);
    const one = 1n;
    const logVocab = Math.log(this.vocabSize);
    const ck = this.ikcache;
    for (const r of miss) {
      const key = keys[r];
      const h0 = key >> bits; // the (order - 1) history tokens, shortest in the low bits
      const ct = key & ((one << bits) - one); // the token
      const tag = (n) => BigInt(n) << BigInt(TAG_SHIFT);
      // m: the longest history length the model has seen, for this pair.
      let m = 0;
      for (let n = 1; n < N; n++) {
        const hq = (h0 & ((one << (bits * BigInt(n))) - one)) | tag(n);
        const hi = Number(hq >> 26n);
        const lo = Number(hq & 0x3ffffffn);
        const i = NgramLM.#find(this.h, hi, lo);
        if (!NgramLM.#hit(this.h, i, hi, lo)) break;
        m = n;
      }
      let acc = 0;
      let res = null;
      for (let n = m; n >= 0; n--) {
        const hp = h0 & ((one << (bits * BigInt(n))) - one);
        const pq = ((hp << bits) | ct) | tag(n);
        const hi = Number(pq >> 26n);
        const lo = Number(pq & 0x3ffffffn);
        const i = NgramLM.#find(this.p, hi, lo);
        if (NgramLM.#hit(this.p, i, hi, lo)) {
          res = acc + this.p.vs[i];
          break;
        }
        const hq = hp | tag(n); // seen for every n <= m
        const j = NgramLM.#find(this.h, Number(hq >> 26n), Number(hq & 0x3ffffffn));
        acc += NgramLM.#hit(this.h, j, Number(hq >> 26n), Number(hq & 0x3ffffffn))
          ? this.h.vs[j]
          : 0;
      }
      if (res === null) res = acc - logVocab;
      out[r] = res;
      if (ck.size < 1_000_000) ck.set(key, res);
    }
    return out;
  }
}

const lse = (a, b) => {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const m = a > b ? a : b;
  return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
};

// CTC prefix beam search over a line's steps.  blank: [T] blank
// log-probabilities; topv / topi: [T, k] flat arrays, the k best classes per
// step (blank included) and their log-probabilities; a class under `candFloor`
// probability at a step is not tried there.  Returns (token ids, the step
// where each token was first emitted).
export function beamSearch(
  blank,
  topv,
  topi,
  lm,
  alpha = 0.5,
  beta = 1.5,
  beam = 8,
  blankSkip = 0.999,
  candFloor = 1e-3
) {
  const bits = BigInt(lm.bits);
  const hmask = (1n << (bits * BigInt(lm.order - 1))) - 1n;
  const K = topi.length / blank.length; // classes per step
  // prefix -> (log p ending in blank, log p ending in non-blank, lm score,
  // steps, the prefix's last order - 1 tokens packed)
  let beams = new Map([["", [0.0, -Infinity, 0.0, [], 0n]]]);
  const logSkip = Math.log(blankSkip);
  const logFloor = Math.log(candFloor);
  const T = blank.length;
  for (let t = 0; t < T; t++) {
    const b0 = blank[t];
    if (b0 > logSkip) {
      beams = new Map(
        [...beams].map(([p, e]) => [p, [lse(e[0], e[1]) + b0, -Infinity, e[2], e[3], e[4]]])
      );
      continue;
    }
    const nxt = new Map();
    const add = (prefix, b, nb, s, st, hk) => {
      const old = nxt.get(prefix);
      if (old === undefined) nxt.set(prefix, [b, nb, s, st, hk]);
      else nxt.set(prefix, [lse(old[0], b), lse(old[1], nb), old[2], old[3], old[4]]);
    };
    const cands = [];
    for (let c = 0; c < K; c++) {
      const cls = Number(topi[t * K + c]);
      const lp = topv[t * K + c];
      if (cls !== 0 && lp >= logFloor) cands.push([cls, lp]);
    }
    // Every beam's own entries first (the blank, and a repeat that collapses):
    // their scores only grow below, so with a full beam the lowest of them is
    // a floor; a new prefix that cannot reach it never enters the beam, and
    // its LM score is not looked up (the result is the same).
    for (const [p, e] of beams) {
      const last = p.length ? Number(p.slice(p.lastIndexOf(",") + 1)) : null;
      add(p, lse(e[0], e[1]) + b0, -Infinity, e[2], e[3], e[4]);
      for (const [c, lp] of cands) if (c === last) add(p, -Infinity, e[1] + lp, e[2], e[3], e[4]);
    }
    const floor =
      beams.size >= beam
        ? Math.min(...[...nxt.values()].map((e) => lse(e[0], e[1]) + e[2]))
        : -Infinity;
    const reach = beta + alpha * lm.maxLogp;
    const ext = []; // (prefix, class, log p of the new token, lm score, steps, history)
    for (const [p, e] of beams) {
      const tot = lse(e[0], e[1]);
      const last = p.length ? Number(p.slice(p.lastIndexOf(",") + 1)) : null;
      for (const [c, lp] of cands) {
        // after a blank, the same class is a new token
        const newFrom = c === last ? e[0] + lp : tot + lp;
        if (newFrom === -Infinity) continue;
        const q = p ? `${p},${c}` : `${c}`;
        if (nxt.has(q)) add(q, -Infinity, newFrom, e[2], e[3], e[4]);
        else if (newFrom + e[2] + reach >= floor - 1e-9) // float slack
          ext.push([p, c, newFrom, e[2], e[3], e[4]]);
      }
    }
    const lps = ext.length
      ? lm.logpKeys(ext.map((e) => (e[5] << bits) | BigInt(e[1] + 1)))
      : [];
    ext.forEach(([p, c, newFrom, s, st, hk], i) => {
      add(p ? `${p},${c}` : `${c}`, -Infinity, newFrom, s + alpha * lps[i] + beta, [...st, t], ((hk << bits) | BigInt(c + 1)) & hmask);
    });
    beams = new Map(
      [...nxt]
        .sort((x, y) => lse(y[1][0], y[1][1]) + y[1][2] - (lse(x[1][0], x[1][1]) + x[1][2]))
        .slice(0, beam)
    );
  }
  let best = null;
  let bestV = -Infinity;
  for (const [p, e] of beams) {
    const v = lse(e[0], e[1]) + e[2];
    if (v > bestV) {
      bestV = v;
      best = p;
    }
  }
  const ids = best ? best.split(",").filter(Boolean).map(Number) : [];
  return [ids, best ? beams.get(best)[3] : []];
}
