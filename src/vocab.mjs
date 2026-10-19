// CTC vocabulary (class 0 = blank), from meta.json.  Port of indra_ocr/vocab.py
// decode(): a LaTeX command token ("\frac") gets a space before a following
// letter; Hangul jamo compose back into syllables.
export class Vocab {
  constructor(chars) {
    this.chars = chars;
    this.index = new Map(chars.map((c, i) => [c, i + 1]));
    this.hasCommands = chars.some((c) => c.length > 1);
    this.jamo = this.index.has("ᄀ"); // Hangul syllables read as jamo
  }

  static fromMeta(obj) {
    if (typeof obj === "string") obj = JSON.parse(obj); // meta.json stores it double-encoded
    if (obj.blank !== 0) throw new Error("only blank=0 vocabularies are supported");
    return new Vocab(obj.chars);
  }

  decode(ids) {
    const toks = [];
    for (const i of ids) if (i > 0 && i <= this.chars.length) toks.push(this.chars[i - 1]);
    if (this.jamo) return toks.join("").normalize("NFC"); // compose the jamo back into syllables
    if (!this.hasCommands) return toks.join("");
    let out = "";
    for (let k = 0; k < toks.length; k++) {
      const t = toks[k];
      const nxt = toks[k + 1] ?? "";
      out += t;
      if (
        t.length > 2 &&
        t[0] === "\\" &&
        /^\p{L}+$/u.test(t.slice(1)) &&
        /^\p{L}/u.test(nxt)
      )
        out += " ";
    }
    return out;
  }
}
