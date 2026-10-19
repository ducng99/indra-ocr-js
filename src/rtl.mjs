// Right-to-left text for a recognizer that reads every line left to right.
// Ported from indra_ocr/rtl.py.
//
// A CTC recognizer emits characters in the order it sees them, left to right,
// so the label of a line with Arabic or Hebrew is its characters in visual
// order.  `flip` turns logical order (how text is stored and typed) into
// visual order, and the same call turns visual order back into logical order:
//
// - RTL base line (more RTL letters than LTR letters): the line reads right to
//   left, so the order of its units reverses.  A unit is a run of LTR text
//   (Latin words, digits, and the spaces and punctuation between them), which
//   keeps its own order, or one other character cluster.
// - LTR base line: the line keeps its order, and each run of RTL text reverses.
//
// A cluster is a base character with its combining marks (Arabic harakat,
// Hebrew niqqud), so marks stay after their base.  A mirrored character
// (brackets) at right-to-left level changes to its mirror image, as a renderer
// draws it.
//
// This is a small subset of the Unicode bidi algorithm (UAX #9): it has no
// embeddings or isolates, and it resolves neutrals by runs.  It is its own
// inverse, so the labels and the decoding always agree.
import { isRtlChar, isLtrChar, isLtrLetter, isCombining } from "./unicode.mjs";

const MIRROR = new Map(
  [..."()[]{}<>«»‹›"].map((c, i) => [c, ")(][}{><»«›‹"[i]])
);
// Presentation forms (glyph variants: initial, medial, final, ligatures) map to
// base letters.  Tatweel stretches a word and carries no letter.
const TATWEEL = "ـ";

export const isRtl = (c) => isRtlChar(c);

export const hasRtl = (text) => {
  for (const c of text) if (isRtlChar(c)) return true;
  return false;
};

// Whether the line reads right to left: more RTL letters than LTR letters.
// The counts do not depend on the order, so logical and visual text agree.
export const rtlBase = (text) => {
  let r = 0;
  let lt = 0;
  for (const c of text) {
    if (isRtlChar(c)) r++;
    else if (isLtrLetter(c)) lt++;
  }
  return r > lt;
};

// Presentation forms to base letters (NFKC on those blocks only), no tatweel,
// no combining mark without a base character.
export function normalize(text) {
  const out = [];
  for (const c of text) {
    const o = c.codePointAt(0);
    if ((o >= 0xfb1d && o <= 0xfdff) || (o >= 0xfe70 && o <= 0xfeff))
      out.push(c.normalize("NFKC"));
    else if (c !== TATWEEL && (out.length || !isCombining(c))) out.push(c);
  }
  return out.join("");
}

export function clusters(text) {
  const out = [];
  for (const c of text) {
    if (out.length && isCombining(c)) out[out.length - 1] += c;
    else out.push(c);
  }
  return out;
}

// A cluster's first code point (py indexes by code point, so read past a
// surrogate pair): the kind of the base character.
const kind = (cl) => (isRtlChar(cl) ? "R" : isLtrChar(cl) ? "L" : "N");

// [start, end) spans from a `strong` cluster to the last `strong` cluster
// before a cluster of a kind in `stop`.
function runs(kinds, strong, stop) {
  const spans = [];
  let start = null;
  let last = null;
  for (let i = 0; i <= kinds.length; i++) {
    const k = i < kinds.length ? kinds[i] : "#";
    if (k === strong) {
      if (start === null) start = i;
      last = i;
    } else if (stop.has(k) || k === "#") {
      if (start !== null) {
        spans.push([start, last + 1]);
        start = last = null;
      }
    }
  }
  return spans;
}

const mirror = (cl) => {
  const c = String.fromCodePoint(cl.codePointAt(0));
  return (MIRROR.get(c) ?? c) + cl.slice(c.length);
};

// Logical order to visual order, and visual order back to logical order.
// baseRtl: the paragraph direction, when known (a renderer lays out a line by
// its paragraph's direction); null decides by `rtlBase`.
export function flip(text, baseRtl = null, mirrorChars = true) {
  if (!hasRtl(text)) return text;
  const cls = clusters(text);
  const kinds = cls.map(kind);
  const rtl = baseRtl === null ? rtlBase(text) : baseRtl;
  if (rtl) {
    // Units: LTR runs kept whole, every other cluster alone; reverse the units.
    const units = [];
    const spans = [...runs(kinds, "L", new Set(["R"])), [cls.length, cls.length]];
    let i = 0;
    for (const [a, b] of spans) {
      for (; i < a; i++) units.push(mirrorChars ? mirror(cls[i]) : cls[i]);
      if (a < b) units.push(cls.slice(a, b).join(""));
      i = b;
    }
    return units.reverse().join("");
  }
  // LTR base: reverse each RTL run in place.
  const out = cls.slice();
  for (const [a, b] of runs(kinds, "R", new Set(["L"]))) {
    const seg = cls.slice(a, b).reverse();
    for (let k = 0; k < seg.length; k++)
      out[a + k] = mirrorChars ? mirror(seg[k]) : seg[k];
  }
  return out.join("");
}
