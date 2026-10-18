// CTC vocabulary (class 0 = blank), from meta.json.  Port of indra_ocr/vocab.py
// decode(): a LaTeX command token ("\frac") gets a space before a following letter.
export class Vocab {
  constructor(chars) {
    this.chars = chars;
    this.index = new Map(chars.map((c, i) => [c, i + 1]));
    this.hasCommands = chars.some((c) => c.length > 1);
  }

  static fromMeta(obj) {
    if (typeof obj === "string") obj = JSON.parse(obj); // meta.json stores it double-encoded
    if (obj.blank !== 0) throw new Error("only blank=0 vocabularies are supported");
    return new Vocab(obj.chars);
  }

  decode(ids) {
    const toks = [];
    for (const i of ids) if (i > 0 && i <= this.chars.length) toks.push(this.chars[i - 1]);
    if (!this.hasCommands) return toks.join("");
    let out = "";
    for (let k = 0; k < toks.length; k++) {
      const t = toks[k];
      const nxt = toks[k + 1] ?? "";
      out += t;
      if (
        t.length > 2 &&
        t[0] === "\\" &&
        /^[A-Za-z]+$/.test(t.slice(1)) &&
        /^[A-Za-z]/.test(nxt)
      )
        out += " ";
    }
    return out;
  }
}
