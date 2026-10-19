// Headless-browser verification of the browser entry (src/browser.mjs):
// fetches the pinned ONNX model set into test/models/ (gitignored, shared
// with test/node.mjs via ./models.mjs), serves
// this package over localhost, and runs OCR in headless Chrome on the wasm
// backend.  With image paths as arguments — and INDRA_OCR_BIN pointing at the
// Python indra-ocr CLI, when available — the lines are compared against it.
// Without arguments, a synthetic page is rendered in the browser as a smoke
// test.  Usage: node test/browser.mjs [image ...]
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer";
import { ensureModels, models } from "./models.mjs";

const root = fileURLToPath(new URL("../", import.meta.url));
const PY = process.env.INDRA_OCR_BIN ?? null;
const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.bundle.min.mjs";

const MIME = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8", ".json": "application/json; charset=utf-8",
  ".png": "image/png", ".onnx": "application/octet-stream", ".wasm": "application/wasm",
};

// --- models: fetch the pinned revision once, keep it out of git ---
await ensureModels();

const server = createServer(async (req, res) => {
  try {
    const p = path.normalize(path.join(root, decodeURIComponent(new URL(req.url, "http://x").pathname)));
    if (!p.startsWith(root)) throw new Error();
    const data = await readFile(p.endsWith("/") ? p + "index.html" : p);
    res.writeHead(200, {
      "content-type": MIME[path.extname(p).toLowerCase()] ?? "application/octet-stream",
      // the host page is about:blank (origin "null"), so module imports need CORS
      "access-control-allow-origin": "*",
    });
    res.end(data);
  } catch {
    res.writeHead(404).end();
  }
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const port = server.address().port;

// Minimal host page for the library: an importmap so the bare
// "onnxruntime-web" import resolves, and a __test(img, base, opts) hook that
// loads the models from `base` (cached across calls) and runs one page.
const testPage = (libUrl) => `<!doctype html><html><head>
<script type="importmap">{"imports":{"onnxruntime-web":"${ORT_URL}"}}</script>
<script type="module">
  import { loadIndra, readGray } from ${JSON.stringify(libUrl)};
  let ocr = null, sig = null;
  window.__test = async (img, base, opts = {}) => {
    const s = JSON.stringify([base, opts.cjk ?? true, opts.order ?? true, opts.orient ?? true]);
    if (sig !== s) {
      ocr = await loadIndra(base, opts);
      sig = s;
      window.__device = ocr.device;
    }
    const gray = await readGray(img);
    const page = await ocr.run(gray);
    return { lines: page.lines, rotation: page.rotation, text: page.text, device: window.__device };
  };
</script></head><body></body></html>`;

async function openTestPage(browser, libUrl) {
  const page = await browser.newPage();
  page.on("console", (m) => m.type() === "error" && !m.text().includes("favicon") && console.error("[page error]", m.text()));
  page.on("pageerror", (e) => console.error("[pageerror]", e.message));
  page.on("requestfailed", (r) => console.error("[request failed]", r.url(), r.failure()?.errorText));
  await page.setContent(testPage(libUrl), { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => typeof window.__test === "function", { timeout: 30000 });
  return page;
}

const browser = await puppeteer.launch({
  headless: "new",
  args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
});
let failed = 0;
try {
  const images = process.argv.slice(2);

  const page = await openTestPage(browser, `http://127.0.0.1:${port}/src/browser.mjs`);
  for (const img of images.length ? images : [null]) {
    // Real images may sit outside the served root — hand them to the page as
    // data URLs; the Python CLI still gets the real path.
    const dataUrl = img
      ? `data:${MIME[path.extname(img).toLowerCase()] ?? "image/png"};base64,${await readFile(img, "base64")}`
      : await page.evaluate(() => {
          const c = document.createElement("canvas");
          c.width = 1024;
          c.height = 512;
          const x = c.getContext("2d");
          x.fillStyle = "#fff";
          x.fillRect(0, 0, c.width, c.height);
          x.fillStyle = "#111";
          x.font = "32px serif";
          ["Hello indra-ocr, running in the browser.", "Second line with numbers 1234567890.",
           "A third line, slightly longer, to split.", "And a fourth one for good measure."].forEach(
            (t, i) => x.fillText(t, 40, 80 + i * 60)
          );
          return c.toDataURL("image/png");
        });
    const t0 = Date.now();
    const result = await page.evaluate(
      async (u, base) => await window.__test(u, base),
      dataUrl, `http://127.0.0.1:${port}/test/models/`
    );
    const ms = Date.now() - t0;
    const jsLines = result.lines.map((l) => l.text);
    let report = `${img ? path.basename(img) : "(synthetic)"} [${result.device ?? "wasm"}]: ${jsLines.length} lines in ${ms} ms, rotation ${result.rotation}`;
    if (!img) {
      if (!jsLines.length) failed++;
    } else if (PY) {
      try {
        const py = await new Promise((resolve, reject) => {
          const p = spawn(PY, [img, "--format", "json", "--model-dir", models, "--cores", "1"]);
          let out = "";
          p.stdout.on("data", (d) => (out += d));
          p.on("close", () => resolve(out));
          p.on("error", reject);
        });
        const pyLines = JSON.parse(py).lines.map((l) => l.text);
        const same = jsLines.filter((t, i) => t === pyLines[i]).length;
        report += ` | python: ${pyLines.length} lines, identical ${same} (${((100 * same) / Math.max(pyLines.length, 1)).toFixed(1)}%)`;
        if (jsLines.length !== pyLines.length || result.device !== "wasm") failed++;
      } catch {
        report += " | (python comparison failed)";
      }
    }
    console.log(report);
    console.log("  first lines:", JSON.stringify(jsLines.slice(0, 3)));
  }
  await page.close();
  if (!images.length)
    console.log("(no images passed — smoke-tested a synthetic page; pass paths to test real ones)");
} finally {
  await browser.close();
}
server.close();
process.exit(failed ? 1 : 0);
