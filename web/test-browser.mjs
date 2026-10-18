// Headless-browser verification: serves this package, and in Chrome runs OCR
// through the library (web/indra-ocr-web.mjs + src/) on the wasm backend,
// comparing with the Python package's output when the indra-ocr CLI is
// available (set INDRA_OCR_BIN, e.g. to a venv's indra-ocr).  Also runs one
// page through the staged hf-repo layout exactly as an app would.
//
// Images default to ../assets/examples/* of a surrounding indra checkout when
// present; otherwise a synthetic page is rendered in the browser as a smoke
// test.  Usage: node web/test-browser.mjs [image ...]
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer";

const root = fileURLToPath(new URL("../", import.meta.url)); // this package
const def = (p) => path.join(root, p);
const PY = process.env.INDRA_OCR_BIN ?? "/tmp/opencode/venv/bin/indra-ocr";
const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.bundle.min.mjs";

let images = process.argv.slice(2);
if (!images.length) {
  const guess = ["table_pmc", "multi_column", "chinese_essay", "rotated_letter"].map(
    (n) => def(`../assets/examples/${n}.png`)
  );
  images = guess.filter((p) => existsSync(p));
}

const MIME = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8", ".json": "application/json; charset=utf-8",
  ".png": "image/png", ".onnx": "application/octet-stream", ".wasm": "application/wasm",
};

const server = createServer(async (req, res) => {
  try {
    const p = path.normalize(path.join(root, decodeURIComponent(new URL(req.url, "http://x").pathname)));
    if (!p.startsWith(root)) throw new Error();
    const data = await readFile(p.endsWith("/") ? p + "index.html" : p);
    res.writeHead(200, { "content-type": MIME[path.extname(p).toLowerCase()] ?? "application/octet-stream", "access-control-allow-origin": "*" });
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
  import { loadIndra, readGrayBrowser } from ${JSON.stringify(libUrl)};
  let ocr = null, sig = null;
  window.__test = async (img, base, opts = {}) => {
    const s = JSON.stringify([base, opts.cjk ?? true, opts.order ?? true, opts.orient ?? true]);
    if (sig !== s) {
      ocr = await loadIndra(base, opts);
      sig = s;
      window.__device = ocr.device;
    }
    const gray = await readGrayBrowser(img);
    const page = await ocr.__call__(gray);
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
  if (!images.length) {
    // No example images around: draw a synthetic page and smoke-test it.
    images = [
      await openTestPage(browser, `http://127.0.0.1:${port}/web/indra-ocr-web.mjs`).then(async (page) =>
        page.evaluate(() => {
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
        }).finally(() => page.close())
      ),
    ];
    console.log("(no example images found — smoke-testing a synthetic page; pass paths to test real ones)");
  }

  // Library as it lives in this package (../src/ import), models from /models.
  {
    const page = await openTestPage(browser, `http://127.0.0.1:${port}/web/indra-ocr-web.mjs`);
    for (const img of images) {
      // Images may sit outside the served root (e.g. ../assets/examples/) — hand
      // them to the page as data URLs; the Python CLI still gets the real path.
      const dataUrl = img.startsWith("data:")
        ? img
        : `data:${MIME[path.extname(img).toLowerCase()] ?? "image/png"};base64,${await readFile(img, "base64")}`;
      const t0 = Date.now();
      const result = await page.evaluate(
        async (u, base) => await window.__test(u, base),
        dataUrl, `http://127.0.0.1:${port}/models/`
      );
      const ms = Date.now() - t0;
      const jsLines = result.lines.map((l) => l.text);
      let report = `${img.startsWith("data:") ? "(synthetic)" : path.basename(img)} [${result.device ?? "wasm"}]: ${jsLines.length} lines in ${ms} ms, rotation ${result.rotation}`;
      if (img.startsWith("data:")) {
        if (!jsLines.length) failed++;
      } else {
        try {
          const py = await new Promise((resolve, reject) => {
            const p = spawn(PY, [img, "--format", "json", "--model-dir", def("models"), "--cores", "1"]);
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
          report += " | (python indra-ocr not available for comparison)";
        }
      }
      console.log(report);
      console.log("  first lines:", JSON.stringify(jsLines.slice(0, 3)));
    }
    await page.close();
  }
} finally {
  await browser.close();
}

// The staged HF repo layout (hf-repo/, from prepare-repo.mjs): serve it and
// run one page through the library exactly as an app would that imports it
// from https://huggingface.co/<repo>/resolve/main/.
const staged = def("hf-repo");
if (existsSync(path.join(staged, "det.onnx"))) {
  const server2 = createServer(async (req, res) => {
    try {
      const p = path.normalize(path.join(staged, decodeURIComponent(new URL(req.url, "http://x").pathname)));
      if (!p.startsWith(staged)) throw new Error();
      const data = await readFile(p);
      res.writeHead(200, { "content-type": MIME[path.extname(p).toLowerCase()] ?? "application/octet-stream", "access-control-allow-origin": "*" });
      res.end(data);
    } catch {
      res.writeHead(404).end();
    }
  });
  await new Promise((r) => server2.listen(0, "127.0.0.1", r));
  const port2 = server2.address().port;
  const browser2 = await puppeteer.launch({
    headless: "new",
    args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
  });
  try {
    const page = await openTestPage(browser2, `http://127.0.0.1:${port2}/indra-ocr-web.mjs`);
    const result = await page.evaluate(async (img, base) => {
      const r = await window.__test(img, base);
      return { n: r.lines.length, first: r.lines.slice(0, 2).map((l) => l.text) };
    }, images[0].startsWith("data:") ? images[0] : `data:${MIME[path.extname(images[0]).toLowerCase()] ?? "image/png"};base64,${await readFile(images[0], "base64")}`, `http://127.0.0.1:${port2}/`);
    console.log(`staged hf-repo layout: ${result.n} lines, first: ${JSON.stringify(result.first)}`);
    if (!result.n) failed++;
    await page.close();
  } finally {
    await browser2.close();
    server2.close();
  }
}
server.close();
process.exit(failed ? 1 : 0);
