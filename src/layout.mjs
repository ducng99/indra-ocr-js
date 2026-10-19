// Page text from the ordered lines and the OrderNet layout heads: markdown
// and layout blocks.  Ported from indra_ocr/layout_post.py.
//
// order.onnx with layout heads gives every line a block class ("cls") and
// every pair of lines a "same block" logit ("same").  Walking the reading
// order, a new block starts wherever the same-block logit between consecutive
// lines is negative.  Each block is then rendered by its class:
//
//   Section-Header        "## " + joined text
//   Text, Caption, ...    one paragraph; a word split by a hyphen at a line end is joined
//   List-Group            one line per item (a new item starts at a bullet or number)
//   Table                 an HTML table assembled from the cell lines (rows by vertical
//                         centre, columns anchored on the fullest rows)
//   Page-Header / Footer  left out unless `keepFurniture`
import { BLOCK_CLASSES, cluster } from "./order.mjs";
import { median, roundHalfEven } from "./imaging.mjs";

const HEADER = BLOCK_CLASSES.indexOf("Page-Header");
const FOOTER = BLOCK_CLASSES.indexOf("Page-Footer");
const SECTION = BLOCK_CLASSES.indexOf("Section-Header");
const LIST = BLOCK_CLASSES.indexOf("List-Group");
const TABLE = BLOCK_CLASSES.indexOf("Table");

const WHOLE_MATH = /^\$[^$]+\$$/; // a line that is one formula
// A list marker, followed by text or alone on its line (the detector often cuts
// a bullet off as a line of its own; the next line then joins its item).
const ITEM = /^\s*([•\-–▪◦*·→]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)])(\s|$)/;

const round1 = (v) => roundHalfEven(v * 10) / 10;
const escapeHtml = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

// Han, kana, CJK punctuation and fullwidth forms: written without spaces
// (Korean Hangul is not: Korean separates words with spaces).
const isCJK = (c) => {
  const o = c.codePointAt(0);
  return (
    (o >= 0x3000 && o <= 0x30ff) ||
    (o >= 0x3400 && o <= 0x9fff) ||
    (o >= 0xf900 && o <= 0xfaff) ||
    (o >= 0xff00 && o <= 0xffef) ||
    (o >= 0x20000 && o <= 0x2fa1f)
  );
};
// Python indexes strings by code point; JS strings by UTF-16 unit.
const firstChar = (s) => String.fromCodePoint(s.codePointAt(0));
const lastChar = (s) =>
  s.length > 1 && s.charCodeAt(s.length - 1) >= 0xdc00 && s.charCodeAt(s.length - 1) <= 0xdfff
    ? s.slice(-2)
    : s.slice(-1);
const isAlpha = (c) => /\p{L}/u.test(c);
const isLower = (c) => /\p{Ll}/u.test(c);

// Join the lines of one paragraph.  "tumori-" + "genesis" -> "tumorigenesis";
// a line break between two CJK characters takes no space.
export function joinLines(texts) {
  let out = "";
  for (let t of texts) {
    t = t.trim();
    if (!t) continue;
    const first = firstChar(t);
    if (
      out.length > 1 &&
      out.endsWith("-") &&
      isAlpha(out[out.length - 2]) &&
      isAlpha(first) &&
      isLower(first)
    )
      out = out.slice(0, -1) + t;
    else if (out && isCJK(lastChar(out)) && isCJK(first)) out += t;
    else out = out ? out + " " + t : t;
  }
  return out;
}

// Runs of the reading order with a positive same-block logit between
// neighbours.  same is the [n, n] row-major logit matrix.
export function splitBlocks(order, same, n) {
  const blocks = [];
  for (const i of order) {
    const last = blocks[blocks.length - 1];
    if (last && same[last[last.length - 1] * n + i] > 0) last.push(i);
    else blocks.push([i]);
  }
  return blocks;
}

const mean = (xs) => xs.reduce((s, v) => s + v, 0) / xs.length;

// Group cell lines into rows by vertical centre, top to bottom, left to right.
function rows(boxes) {
  const cy = boxes.map((b) => (b[1] + b[3]) / 2);
  const h = median(boxes.map((b) => b[3] - b[1]));
  const rows = [];
  for (const i of cy.map((v, j) => j).sort((a, b) => cy[a] - cy[b])) {
    const last = rows[rows.length - 1];
    if (last && Math.abs(cy[i] - mean(last.map((k) => cy[k]))) < 0.5 * h) last.push(i);
    else rows.push([i]);
  }
  return rows.map((r) => r.slice().sort((a, b) => boxes[a][0] - boxes[b][0]));
}

// Join the line rows of wrapped cells into one table row.
//
// Inside a wrapped cell the gap to the next line is the font's line gap; a new
// table row adds cell padding.  A row joins the row above when its gap is under
// half the table's median row gap and under 0.6 line heights, and it has fewer
// lines than the row above (a continuation fills only the wrapped columns).
function mergeWrapped(rs, boxes) {
  if (rs.length < 3) return rs;
  const h = median(boxes.map((b) => b[3] - b[1]));
  const gaps = rs.slice(0, -1).map((cur, k) => {
    const nxt = rs[k + 1];
    return Math.min(...nxt.map((i) => boxes[i][1])) - Math.max(...cur.map((i) => boxes[i][3]));
  });
  const med = median(gaps);
  const out = [rs[0].slice()];
  let prev = rs[0];
  for (let k = 0; k < gaps.length; k++) {
    const r = rs[k + 1];
    // A continuation fills only the wrapped columns: fewer lines than the row above.
    if (gaps[k] < 0.5 * med && gaps[k] < 0.6 * h && r.length < prev.length)
      out[out.length - 1].push(...r);
    else out.push(r.slice());
    prev = r;
  }
  return out;
}

// First index where edges[i] > v (np.searchsorted(edges, v, side="right")).
const searchsortedRight = (edges, v) => {
  let lo = 0;
  let hi = edges.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (edges[mid] > v) hi = mid;
    else lo = mid + 1;
  }
  return lo;
};

// An HTML table from cell lines (one text line per cell, as born-digital PDFs give).
//
// A first row that is one line across most of the table is a title: it goes
// before the table as text.  The first remaining row is the header (<th>).
export function tableHtml(boxes, texts) {
  const rs = rows(boxes);
  const spanX = Math.max(...boxes.map((b) => b[2])) - Math.min(...boxes.map((b) => b[0]));
  if (
    rs.length > 2 &&
    rs[0].length === 1 &&
    Math.max(...rs.map((r) => r.length)) > 1 &&
    boxes[rs[0][0]][2] - boxes[rs[0][0]][0] > 0.6 * spanX
  ) {
    const keep = boxes.map((_, k) => k).filter((k) => k !== rs[0][0]);
    return (
      texts[rs[0][0]] +
      "\n\n" +
      tableHtml(keep.map((k) => boxes[k]), keep.map((k) => texts[k]))
    );
  }
  const width = Math.max(...rs.map((r) => r.length));
  const full = rs.filter((r) => r.length === width);
  const merged = mergeWrapped(rs, boxes);
  // Column bands from the rows with the most cells: those rows rarely span.
  const bands = Array.from({ length: width }, (_, c) => [
    median(full.map((r) => boxes[r[c]][0])),
    median(full.map((r) => boxes[r[c]][2])),
  ]);
  // Widen each band to the midpoints between neighbours, so every x falls in one band.
  const edges = [-Infinity];
  for (let c = 0; c < width - 1; c++) edges.push((bands[c][1] + bands[c + 1][0]) / 2);
  edges.push(Infinity);
  const out = ["<table>"];
  for (let ri = 0; ri < merged.length; ri++) {
    const tag = ri === 0 && merged.length > 1 ? "th" : "td";
    const cells = new Map();
    for (const k of merged[ri].slice().sort((a, b) => boxes[a][1] - boxes[b][1] || boxes[a][0] - boxes[b][0])) {
      const [x0, , x1] = boxes[k];
      const first = searchsortedRight(edges, x0) - 1;
      const last = searchsortedRight(edges, x1) - 1;
      // A cell spans every band it covers by more than half of the band.
      let span = [];
      for (let c = Math.max(first, 0); c <= Math.min(last, width - 1); c++) {
        if (Math.min(x1, bands[c][1]) - Math.max(x0, bands[c][0]) > 0.5 * (bands[c][1] - bands[c][0]))
          span.push(c);
      }
      if (!span.length) span = [Math.min(Math.max(first, 0), width - 1)];
      const c0 = span[0];
      const prev = cells.get(c0);
      cells.set(c0, {
        n: prev ? Math.max(prev.n, span.length) : span.length,
        text: prev ? prev.text + " " + texts[k] : texts[k],
      });
    }
    const tds = [];
    let c = 0;
    while (c < width) {
      const cell = cells.get(c);
      if (cell) {
        tds.push(`<${tag}${cell.n > 1 ? ` colspan="${cell.n}"` : ""}>${escapeHtml(cell.text)}</${tag}>`);
        c += cell.n;
      } else {
        tds.push(`<${tag}></${tag}>`);
        c += 1;
      }
    }
    out.push("<tr>" + tds.join("") + "</tr>");
  }
  out.push("</table>");
  return out.join("\n");
}

// The lines of one cell are stacked (they overlap horizontally).  When the
// cell head joins side-by-side lines in more than 10% of its cells, it has
// failed on this table and the geometric assembly is used instead.
export function headsPlausible(boxes, tab, n) {
  const cells = cluster(tab.subarray(2 * n * n), n);
  let bad = 0;
  for (const c of cells) {
    if (c.length < 2) continue;
    let mn = Infinity;
    for (const a of c)
      for (const b of c)
        mn = Math.min(mn, Math.min(boxes[a][2], boxes[b][2]) - Math.max(boxes[a][0], boxes[b][0]));
    if (mn < 0) bad++;
  }
  return bad <= 0.1 * cells.length;
}

// tab [3, n, n] -> the block's own [3, b, b] submatrix for the line indices blk.
function subTab(tab, blk) {
  const n = Math.round(Math.sqrt(tab.length / 3));
  const b = blk.length;
  const out = new Float32Array(3 * b * b);
  for (let k = 0; k < 3; k++)
    for (let i = 0; i < b; i++)
      for (let j = 0; j < b; j++)
        out[k * b * b + i * b + j] = tab[k * n * n + blk[i] * n + blk[j]];
  return out;
}

// An HTML table from the OrderNet table heads.  tab [3, N, N] = logits
// same row, same column, same cell, for the N lines of one table block.
export function tableHtmlHeads(boxes, texts, tab, n) {
  const cells = cluster(tab.subarray(2 * n * n), n);
  const cb = cells.map((c) => [
    Math.min(...c.map((i) => boxes[i][0])),
    Math.min(...c.map((i) => boxes[i][1])),
    Math.max(...c.map((i) => boxes[i][2])),
    Math.max(...c.map((i) => boxes[i][3])),
  ]);
  const ctext = cells.map((c) =>
    c
      .slice()
      .sort((a, b) => boxes[a][1] - boxes[b][1] || boxes[a][0] - boxes[b][0])
      .map((i) => texts[i])
      .join(" ")
  );
  const m = cells.length;
  const cellLogits = (k) => {
    const out = new Float32Array(m * m);
    for (let a = 0; a < m; a++)
      for (let b = a + 1; b < m; b++) {
        let s = 0;
        for (const i of cells[a]) for (const j of cells[b]) s += tab[k * n * n + i * n + j];
        const v = s / (cells[a].length * cells[b].length);
        out[a * m + b] = out[b * m + a] = v;
      }
    return out;
  };
  const byMinY = (g) => Math.min(...g.map((c) => cb[c][1]));
  const byMinX = (g) => Math.min(...g.map((c) => cb[c][0]));
  const rowGroups = cluster(cellLogits(0), m).sort((a, b) => byMinY(a) - byMinY(b));
  const colGroups = cluster(cellLogits(1), m).sort((a, b) => byMinX(a) - byMinX(b));
  const rowOf = new Map();
  rowGroups.forEach((g, r) => g.forEach((c) => rowOf.set(c, r)));
  const colOf = new Map();
  colGroups.forEach((g, k) => g.forEach((c) => colOf.set(c, k)));
  const centers = colGroups.map((g) => median(g.map((c) => (cb[c][0] + cb[c][2]) / 2)));
  const h = median(boxes.map((b) => b[3] - b[1]));
  const grid = new Map();
  for (let c = 0; c < m; c++) {
    const c0 = colOf.get(c);
    let span = 1;
    while (
      c0 + span < colGroups.length &&
      cb[c][0] - h <= centers[c0 + span] &&
      centers[c0 + span] <= cb[c][2] + h
    )
      span++;
    const key = rowOf.get(c) * colGroups.length + c0;
    const prev = grid.get(key);
    grid.set(key, {
      span: prev ? Math.max(prev.span, span) : span,
      text: prev ? prev.text + " " + ctext[c] : ctext[c],
    });
  }
  const out = ["<table>"];
  for (let r = 0; r < rowGroups.length; r++) {
    const tag = r === 0 && rowGroups.length > 1 ? "th" : "td";
    const tds = [];
    let c = 0;
    while (c < colGroups.length) {
      const cell = grid.get(r * colGroups.length + c);
      if (cell) {
        tds.push(`<${tag}${cell.span > 1 ? ` colspan="${cell.span}"` : ""}>${escapeHtml(cell.text)}</${tag}>`);
        c += cell.span;
      } else {
        tds.push(`<${tag}></${tag}>`);
        c += 1;
      }
    }
    out.push("<tr>" + tds.join("") + "</tr>");
  }
  out.push("</table>");
  return out.join("\n");
}

// Block class of a block: the summed cls logits over its lines, argmax.
function blockCls(cls, blk, n) {
  const C = cls.length / n;
  let am = 0;
  let bv = -Infinity;
  for (let c = 0; c < C; c++) {
    let s = 0;
    for (const i of blk) s += cls[i * C + c];
    if (s > bv) {
      bv = s;
      am = c;
    }
  }
  return am;
}

// The layout blocks behind pageMarkdown, in reading order:
// {label (a BLOCK_CLASSES name), bbox (pixels, the union of its lines),
// text (its lines, newline-joined)}.
export function pageBlocks(boxes, texts, order, cls, same) {
  const n = boxes.length;
  return splitBlocks(order, same, n).map((blk) => {
    const b = blk.map((i) => boxes[i]);
    return {
      label: BLOCK_CLASSES[blockCls(cls, blk, n)],
      bbox: [
        round1(Math.min(...b.map((x) => x[0]))),
        round1(Math.min(...b.map((x) => x[1]))),
        round1(Math.max(...b.map((x) => x[2]))),
        round1(Math.max(...b.map((x) => x[3]))),
      ],
      text: blk.map((i) => texts[i]).join("\n"),
    };
  });
}

// Markdown for one page.  boxes [[x0, y0, x1, y1], ...], texts, order =
// reading order, cls [n, C] block logits, same [n, n] "same block" logits.
// `cells` maps a line index to its table cells [(box, text)] when the line
// holds several; `tab` [3, n, n] (same row / column / cell logits) builds
// tables from the heads.  `pictures` [P, 4]: picture regions, each written as
// `![](picture-<k>)` before the first block below it that it overlaps
// horizontally.
export function pageMarkdown(boxes, texts, order, cls, same, opts = {}) {
  const { keepFurniture = false, cells = null, tab = null, pictures = null } = opts;
  const n = boxes.length;
  const parts = [];
  const partBoxes = [];
  for (const blk of splitBlocks(order, same, n)) {
    const nParts = parts.length;
    blockPart(parts, blk, boxes, texts, cls, keepFurniture, cells, tab, n);
    if (parts.length > nParts) {
      const b = blk.map((i) => boxes[i]);
      partBoxes.push([
        Math.min(...b.map((x) => x[0])),
        Math.min(...b.map((x) => x[1])),
        Math.max(...b.map((x) => x[2])),
        Math.max(...b.map((x) => x[3])),
      ]);
    }
  }
  if (pictures && pictures.length) {
    const slots = new Map();
    pictures.forEach(([x0, y0, x1, y1], k) => {
      const cy = (y0 + y1) / 2;
      const at = partBoxes.findIndex(
        ([bx0, by0, bx1, by1]) => (by0 + by1) / 2 > cy && Math.min(x1, bx1) > Math.max(x0, bx0)
      );
      const j = at === -1 ? parts.length : at;
      if (!slots.has(j)) slots.set(j, []);
      slots.get(j).push(`![](picture-${k})`);
    });
    const merged = [];
    for (let j = 0; j <= parts.length; j++) {
      merged.push(...(slots.get(j) ?? []));
      if (j < parts.length) merged.push(parts[j]);
    }
    parts.length = 0;
    parts.push(...merged);
  }
  return parts.filter(Boolean).join("\n\n");
}

// Append one block's markdown to `parts` (nothing for dropped furniture).
function blockPart(parts, blk, boxes, texts, cls, keepFurniture, cells, tab, n) {
  const c = blockCls(cls, blk, n);
  const textsB = blk.map((i) => texts[i]);
  if ((c === HEADER || c === FOOTER) && !keepFurniture) return;
  if (textsB.every((t) => WHOLE_MATH.test(t.trim()))) {
    // display equations
    parts.push(textsB.map((t) => "$$" + t.trim().slice(1, -1) + "$$").join("\n"));
    return;
  }
  if (c === SECTION) parts.push("## " + joinLines(textsB));
  else if (c === TABLE && blk.length > 1 && tab) {
    const bBoxes = blk.map((i) => boxes[i]);
    const tSub = subTab(tab, blk);
    if (headsPlausible(bBoxes, tSub, blk.length))
      parts.push(tableHtmlHeads(bBoxes, textsB, tSub, blk.length));
    else {
      const cb = [];
      const ct = [];
      for (const i of blk)
        for (const [bx, tx] of cells?.get(i) || [[boxes[i], texts[i]]]) {
          cb.push(bx);
          ct.push(tx);
        }
      parts.push(tableHtml(cb, ct));
    }
  } else if (c === TABLE && blk.length > 1) {
    const cb = [];
    const ct = [];
    for (const i of blk)
      for (const [bx, tx] of cells?.get(i) || [[boxes[i], texts[i]]]) {
        cb.push(bx);
        ct.push(tx);
      }
    parts.push(tableHtml(cb, ct));
  } else if (c === LIST) {
    const items = [];
    for (const t of textsB) {
      if (!items.length || ITEM.test(t)) items.push([t]);
      else items[items.length - 1].push(t);
    }
    parts.push(items.map(joinLines).join("\n"));
  } else parts.push(joinLines(textsB));
}
