// indra-ocr playground — demo webapp for @ducng99/indra-ocr-js, bundled with
// bun (see ../package.json).  Imports the library from the repo source; your
// own app would install it and import the same names:
//
//   import { loadIndra, readGray } from "@ducng99/indra-ocr-js/browser";
import { loadIndra, readGray } from "../../src/browser.mjs";

const $ = (id) => document.getElementById(id);
const ui = {
  run: $("run"), gear: $("gear"), panel: $("panel"),
  models: $("models"), cjk: $("cjk"), orient: $("orient"),
  pbar: $("pbar"), pfill: $("pbar").firstElementChild,
  viewport: $("viewport"), wrap: $("pageWrap"), cv: $("cv"), overlay: $("overlay"),
  empty: $("empty"), samples: $("samples"), pick: $("pick"),
  open: $("open"), file: $("file"),
  zin: $("zin"), zout: $("zout"), zfit: $("zfit"), zpct: $("zpct"), dims: $("dims"),
  tabs: [...document.querySelectorAll(".tab")], copy: $("copy"),
  lines: $("lines"), mdview: $("mdview"), ph: $("ph"), scroll: $("scroll"),
  status: $("status"), chips: $("chips"),
};

const state = {
  gray: null,        // grayscale of the loaded image (OCR input)
  shown: null,       // grayscale actually painted (quarter-turn-corrected)
  page: null,        // last ocr.run() result
  view: "text",      // results rail tab
  zoom: "fit",       // "fit" | factor (1 = 100%)
  busy: false,
};

const status = (msg, err = false) => {
  ui.status.textContent = msg;
  ui.status.classList.toggle("err", err);
};
const mb = (n) => `${(n / 1048576).toFixed(1)} MB`;

// ── models ──────────────────────────────────────────────────────────────
// One loadIndra per (repo, options); a change reloads on the next run.
// loadIndra streams each file; progress events are aggregated into the bar.
let ocr = null, sig = null;
const dl = new Map();

function onProgress(p) {
  dl.set(p.file, { loaded: p.loaded, total: p.total });
  let loaded = 0, total = 0, known = true;
  for (const { loaded: l, total: t } of dl.values()) {
    loaded += l;
    if (t == null) known = false;
    else total += t;
  }
  ui.pbar.hidden = false;
  ui.pbar.classList.toggle("indet", !known);
  ui.pfill.style.transform = known ? `scaleX(${total ? loaded / total : 0})` : "";
  status(known
    ? `downloading models ${mb(loaded)} / ${mb(total)} — ${p.file}`
    : `downloading models — ${p.file}`);
}

async function getOcr() {
  const base = ui.models.value.trim() || "datalab-to/indra-ocr@v1.1.0";
  const opts = { cjk: ui.cjk.checked, orient: ui.orient.checked };
  const next = JSON.stringify([base, opts.cjk, opts.orient]);
  if (ocr && sig === next) return ocr;
  ocr = null;
  sig = next;
  dl.clear();
  try {
    ocr = await loadIndra(base, { ...opts, progress: onProgress });
    ui.pbar.hidden = true;
    return ocr;
  } catch (e) {
    sig = null;
    ui.pbar.hidden = true;
    throw e;
  }
}

// settings changes take effect on the next run
for (const input of [ui.models, ui.cjk, ui.orient]) {
  input.addEventListener("change", () => {
    if (sig !== null) status("settings changed — models reload on the next run");
    sig = null;
  });
}

// settings popover
ui.gear.addEventListener("click", () => {
  ui.panel.hidden = !ui.panel.hidden;
  ui.gear.setAttribute("aria-expanded", String(!ui.panel.hidden));
});
document.addEventListener("pointerdown", (e) => {
  if (!ui.panel.hidden && !e.target.closest(".settings")) {
    ui.panel.hidden = true;
    ui.gear.setAttribute("aria-expanded", "false");
  }
});

// ── samples ─────────────────────────────────────────────────────────────
// Pages drawn on canvases, so the demo needs no image assets.  The upside-
// down one shows off orientation: run() reports rotation and the boxes land
// on the corrected view.
function paper(w, h) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const x = c.getContext("2d");
  x.fillStyle = "#fdfdfb";
  x.fillRect(0, 0, w, h);
  x.fillStyle = "#1a1a1e";
  return x;
}
const serif = (px, bold) => `${bold ? "bold " : ""}${px}px Georgia, "Times New Roman", serif`;
const mono = (px) => `${px}px ui-monospace, Menlo, Consolas, monospace`;

function drawArticle({ flip = false } = {}) {
  const w = 1080, h = 560;
  const x = paper(w, h);
  x.font = serif(38, true);
  x.fillText("Notes from a Quiet Machine", 64, 104);
  x.font = serif(25);
  [
    "This page was recognized without leaving the browser:",
    "line detection, recognition, reading order and orientation",
    "all run locally on WebAssembly, via ONNX Runtime.",
    "Your pixels stay put — only the model weights travel,",
    "once, and then the browser caches them.",
    "Filed under on-device intelligence",
  ].forEach((t, i) => x.fillText(t, 64, 180 + i * 54));
  if (!flip) return x.canvas;
  const r = paper(w, h);
  r.translate(w / 2, h / 2);
  r.rotate(Math.PI);
  r.drawImage(x.canvas, -w / 2, -h / 2);
  return r.canvas;
}

function drawColumns() {
  const w = 1080, h = 620;
  const x = paper(w, h);
  x.font = serif(36, true);
  x.fillText("Reading Order, Demonstrated", 64, 96);
  x.font = serif(23);
  const left = [
    "Documents rarely read in one straight run:",
    "columns, sidebars and tables all compete",
    "for attention. Detection only sees lines,",
    "so ordering is a problem of its own.",
  ];
  const right = [
    "1. find the lines",
    "2. read every line, left to right",
    "3. decide what belongs to what block",
    "4. put the blocks back in order",
  ];
  left.forEach((t, i) => x.fillText(t, 64, 180 + i * 56));
  right.forEach((t, i) => x.fillText(t, 610, 180 + i * 56));
  return x.canvas;
}

function drawReceipt() {
  const w = 640, h = 860;
  const x = paper(w, h);
  x.font = mono(30);
  x.textAlign = "center";
  x.fillText("CORNER BOOKSTORE", w / 2, 96);
  x.font = mono(22);
  x.fillText("123 Inkwell Lane · tel 555-0110", w / 2, 136);
  x.textAlign = "left";
  const row = (l, r, y) => {
    x.fillText(l, 56, y);
    x.textAlign = "right";
    x.fillText(r, w - 56, y);
    x.textAlign = "left";
  };
  let y = 220;
  for (const [l, r] of [
    ["Map of Invisible Cities", "18.50"],
    ["The Index Card", "9.75"],
    ["Brass Bookmark", "4.00"],
    ["Absolute Python 4e", "52.00"],
  ]) {
    row(l, r, y);
    y += 52;
  }
  x.fillText("—".repeat(24), 56, y + 14);
  row("SUBTOTAL", "84.25", y + 70);
  row("TAX 8%", "6.74", y + 122);
  x.font = mono(30);
  row("TOTAL", "90.99", y + 190);
  x.font = mono(20);
  row("2026-10-19 14:32", "VISA ..0424", y + 260);
  x.textAlign = "center";
  x.fillText("thank you — come again", w / 2, y + 330);
  return x.canvas;
}

function drawHanzi() {
  const w = 1080, h = 520;
  const x = paper(w, h);
  x.font = serif(40, true);
  x.fillText("光学字符识别", 64, 100);
  x.font = serif(26);
  [
    "检测、识别与阅读顺序模型都运行在浏览器里。",
    "图片不会离开这台设备，识别结果实时生成。",
    "标点也要保留：，。、；！？——（2026）",
  ].forEach((t, i) => x.fillText(t, 64, 190 + i * 60));
  return x.canvas;
}

const SAMPLES = [
  { label: "Article", draw: drawArticle },
  { label: "Two columns", draw: drawColumns },
  { label: "Receipt", draw: drawReceipt },
  { label: "中文", draw: drawHanzi },
  { label: "Upside down", draw: () => drawArticle({ flip: true }) },
];

for (const s of SAMPLES) {
  const canvas = s.draw();
  const btn = document.createElement("button");
  const thumb = document.createElement("span");
  thumb.className = "thumb";
  thumb.appendChild(canvas);
  const lbl = document.createElement("span");
  lbl.className = "lbl";
  lbl.textContent = s.label;
  btn.append(thumb, lbl);
  btn.onclick = () => {
    canvas.toBlob(async (blob) => {
      status(`sample: ${s.label}…`);
      await showImage(blob);
    }, "image/png");
  };
  ui.samples.appendChild(btn);
}

// ── display ─────────────────────────────────────────────────────────────
// The models see the page turned upright; boxes come back in that frame, so
// once run() reports a rotation the painted view is quarter-turned to match.
function rot90cw({ data, width: w, height: h }) {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) out[x * h + (h - 1 - y)] = data[y * w + x];
  return { data: out, width: h, height: w };
}

function paintGray(g) {
  state.shown = g;
  ui.cv.width = g.width;
  ui.cv.height = g.height;
  const ctx = ui.cv.getContext("2d");
  const img = ctx.createImageData(g.width, g.height);
  for (let i = 0, p = 0; i < g.data.length; i++, p += 4) {
    img.data[p] = img.data[p + 1] = img.data[p + 2] = g.data[i];
    img.data[p + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  ui.dims.textContent = `${g.width} × ${g.height} px`;
  ui.overlay.replaceChildren();
  applyZoom();
}

// Fit is computed in JS (explicit px on the canvas), never in CSS: a
// percentage max-width against the max-content-sized pageWrap is cyclic — the
// canvas ends up 2×padding narrower than the overlay and boxes skew sideways.
function fitScale() {
  const vw = ui.viewport.clientWidth - 48, vh = ui.viewport.clientHeight - 48;
  return Math.max(0.05, Math.min(1, vw / state.shown.width, vh / state.shown.height));
}

function applyZoom() {
  const fit = state.zoom === "fit";
  const z = fit ? fitScale() : state.zoom;
  ui.cv.style.width = `${Math.round(state.shown.width * z)}px`;
  ui.zpct.textContent = fit ? "fit" : `${Math.round(z * 100)}%`;
}

ui.zin.onclick = () => setZoom((state.zoom === "fit" ? fitScale() : state.zoom) * 1.25);
ui.zout.onclick = () => setZoom(Math.min(5, Math.max(0.15, (state.zoom === "fit" ? fitScale() : state.zoom) / 1.25)));
ui.zfit.onclick = () => setZoom("fit");
function setZoom(z) {
  state.zoom = z;
  applyZoom();
}
new ResizeObserver(() => {
  if (state.zoom === "fit" && state.shown) applyZoom();
}).observe(ui.viewport);

async function showImage(src) {
  const g = await readGray(src);
  state.gray = g;
  state.page = null;
  state.zoom = "fit";
  paintGray(g);
  ui.empty.hidden = true;
  ui.wrap.hidden = false;
  ui.pick.hidden = false;
  ui.run.disabled = false;
  clearResults();
  await run();
}

// ── results rail ────────────────────────────────────────────────────────
function clearResults() {
  ui.lines.replaceChildren();
  ui.lines.hidden = true;
  ui.mdview.hidden = true;
  ui.ph.hidden = false;
  ui.chips.replaceChildren();
  ui.tabs.find((t) => t.dataset.v === "md").disabled = true;
  setView("text");
}

function setView(v) {
  state.view = v;
  for (const t of ui.tabs) t.classList.toggle("on", t.dataset.v === v);
  ui.mdview.hidden = v !== "md";
  ui.lines.hidden = v !== "text" || !state.page;
  ui.ph.hidden = !(v === "text" && !state.page);
}

for (const t of ui.tabs)
  t.onclick = () => !t.disabled && setView(t.dataset.v);

ui.copy.onclick = async () => {
  const text = state.view === "md" ? state.page?.markdown : state.page?.text;
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    status(state.view === "md" ? "markdown copied" : "text copied");
  } catch {
    status("clipboard blocked by the browser", true);
  }
};

function hover(i, on) {
  ui.overlay.children[i]?.classList.toggle("hot", on);
  ui.lines.children[i]?.classList.toggle("hot", on);
  if (on) ui.lines.children[i]?.scrollIntoView({ block: "nearest" });
}

function renderResults() {
  const page = state.page;
  ui.overlay.replaceChildren();
  ui.lines.replaceChildren();
  page.lines.forEach((l, i) => {
    const [x0, y0, x1, y1] = l.box;
    const b = document.createElement("div");
    b.className = "box";
    b.style.cssText =
      `left:${(x0 / state.shown.width) * 100}%;top:${(y0 / state.shown.height) * 100}%;` +
      `width:${((x1 - x0) / state.shown.width) * 100}%;height:${((y1 - y0) / state.shown.height) * 100}%`;
    b.onmouseenter = () => hover(i, true);
    b.onmouseleave = () => hover(i, false);
    ui.overlay.appendChild(b);

    const row = document.createElement("div");
    row.className = "row";
    const pct = Math.round((l.conf ?? 0) * 100);
    const idx = document.createElement("span");
    idx.className = "idx";
    idx.textContent = i + 1;
    const txt = document.createElement("span");
    txt.className = "txt";
    txt.textContent = l.text;
    const bar = document.createElement("span");
    bar.className = "conf";
    bar.title = `confidence ${pct}%`;
    bar.appendChild(document.createElement("i")).style.width = `${pct}%`;
    const p = document.createElement("span");
    p.className = "pct";
    p.textContent = `${pct}%`;
    row.append(idx, txt, bar, p);
    row.onmouseenter = () => hover(i, true);
    row.onmouseleave = () => hover(i, false);
    ui.lines.appendChild(row);
  });
  ui.ph.hidden = true;
  ui.lines.hidden = state.view !== "text";
  const mdTab = ui.tabs.find((t) => t.dataset.v === "md");
  mdTab.disabled = page.markdown == null;
  ui.mdview.textContent = page.markdown ?? "";
  renderChips();
}

function chip(text, cls = "") {
  const s = document.createElement("span");
  s.className = `chip ${cls}`.trim();
  s.textContent = text;
  ui.chips.appendChild(s);
}

function renderChips() {
  const { page } = state;
  ui.chips.replaceChildren();
  chip(`${page.lines.length} lines`, "good");
  if (page.rotation) chip(`↻ ${page.rotation}°`, "good");
  if (page.skew) chip(`deskew ${page.skew}°`);
  const t = page.timings ?? {};
  if (t.orientMs != null) chip(`orient ${t.orientMs}ms`);
  if (t.detMs != null) chip(`det ${t.detMs}ms`);
  if (t.recMs != null) chip(`rec ${t.recMs}ms`);
  if (t.orderMs != null) chip(`order ${t.orderMs}ms`);
}

// ── run ─────────────────────────────────────────────────────────────────
async function run() {
  if (!state.gray || state.busy) return;
  state.busy = true;
  ui.run.disabled = true;
  const t0 = performance.now();
  try {
    await getOcr();
    status("running OCR…");
    await new Promise((r) => setTimeout(r, 20)); // let the status paint
    const page = await ocr.run(state.gray);
    state.page = page;
    if (page.rotation) paintGray(rotN(state.gray, page.rotation / 90));
    setView(state.view);
    renderResults();
    const ms = Math.round(performance.now() - t0);
    status(`done — hover a line to locate it on the page` +
      (page.rotation ? ` (page was turned ${page.rotation}°)` : ""));
    chip(`${ms} ms`);
  } catch (e) {
    status("failed: " + (e?.message ?? e), true);
    console.error(e);
  }
  ui.run.disabled = false;
  state.busy = false;
}

function rotN(g, k) {
  for (; k > 0; k--) g = rot90cw(g);
  return g;
}

ui.run.onclick = () => run();

// samples overlay toggle (once an image is loaded)
ui.pick.onclick = () => {
  ui.empty.hidden = !ui.empty.hidden;
};

// ── input: open / drop / paste ──────────────────────────────────────────
ui.open.onclick = () => ui.file.click();
ui.file.onchange = () => {
  if (ui.file.files[0]) showImage(ui.file.files[0]);
  ui.file.value = "";
};

let dragCount = 0;
window.addEventListener("dragenter", (e) => {
  e.preventDefault();
  if (++dragCount === 1) ui.viewport.classList.add("drag");
});
window.addEventListener("dragleave", () => {
  if (--dragCount === 0) ui.viewport.classList.remove("drag");
});
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => {
  e.preventDefault();
  dragCount = 0;
  ui.viewport.classList.remove("drag");
  const f = e.dataTransfer.files[0];
  if (f) showImage(f);
});
window.addEventListener("paste", (e) => {
  const item = [...(e.clipboardData?.items ?? [])].find((i) => i.type.startsWith("image/"));
  const f = item?.getAsFile();
  if (f) showImage(f);
});
