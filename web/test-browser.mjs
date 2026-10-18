// Headless-browser verification: serves this package, loads web/demo.html in
// Chrome, runs OCR through the same code the demo uses on wasm AND on WebGPU
// (SwiftShader software GPU in headless Chrome), and compares each with the
// Python package's output when the indra-ocr CLI is available (set
// INDRA_OCR_BIN, e.g. to a venv's indra-ocr).  Also runs one page through the
// staged hf-repo layout exactly as an app would.
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
    res.writeHead(200, { "content-type": MIME[path.extname(p).toLowerCase()] ?? "application/octet-stream" });
    res.end(data);
  } catch {
    res.writeHead(404).end();
  }
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const port = server.address().port;

const browser = await puppeteer.launch({
  headless: "new",
  args: [
    "--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage",
    "--enable-unsafe-swiftshader", // WebGPU via the software adapter
  ],
});
let failed = 0;
try {
  const page = await browser.newPage();
  page.on("console", (m) => m.type() === "error" && !m.text().includes("favicon") && console.error("[page error]", m.text()));
  page.on("requestfailed", (r) => console.error("[request failed]", r.url(), r.failure()?.errorText));
  await page.goto(`http://127.0.0.1:${port}/web/demo.html`, { waitUntil: "networkidle0" });

  const hasWebGPU = await page.evaluate(async () => {
    if (!navigator.gpu) return false;
    try { return !!(await navigator.gpu.requestAdapter()); } catch { return false; }
  });
  console.log(`WebGPU adapter in headless Chrome: ${hasWebGPU ? "yes (SwiftShader)" : "no"}`);

  if (!images.length) {
    // No example images around: draw a synthetic page and smoke-test it.
    images = [
      await page.evaluate(() => {
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
      }),
    ];
    console.log("(no example images found — smoke-testing a synthetic page; pass paths to test real ones)");
  }

  for (const img of images) {
    // Images may sit outside the served root (e.g. ../assets/examples/) — hand
    // them to the page as data URLs; the Python CLI still gets the real path.
    const dataUrl = img.startsWith("data:")
      ? img
      : `data:${MIME[path.extname(img).toLowerCase()] ?? "image/png"};base64,${await readFile(img, "base64")}`;
    for (const device of ["wasm", ...(hasWebGPU ? ["webgpu"] : [])]) {
      const t0 = Date.now();
      const result = await page.evaluate(
        async (u, base, dev) => await window.__test(u, base, dev),
        dataUrl, "/models", device
      );
      const ms = Date.now() - t0;
      const jsLines = result.lines.map((l) => l.text);
      let report = `${img.startsWith("data:") ? "(synthetic)" : path.basename(img)} [${result.device ?? device}]: ${jsLines.length} lines in ${ms} ms, rotation ${result.rotation}`;
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
          if (jsLines.length !== pyLines.length || result.device !== device) failed++;
        } catch {
          report += " | (python indra-ocr not available for comparison)";
        }
      }
      console.log(report);
      if (device === "wasm") console.log("  first lines:", JSON.stringify(jsLines.slice(0, 3)));
    }
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
    args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage", "--enable-unsafe-swiftshader"],
  });
  try {
    const page = await browser2.newPage();
    page.on("requestfailed", (r) => console.error("[request failed]", r.url(), r.failure()?.errorText));
    await page.goto(`http://127.0.0.1:${port}/web/demo.html`, { waitUntil: "domcontentloaded" });
    const result = await page.evaluate(async (img, base) => {
      const { loadIndra, readGrayBrowser } = await import(base + "indra-ocr-web.mjs");
      const ocr = await loadIndra(base, { device: "wasm" });
      const gray = await readGrayBrowser(img);
      const page = await ocr.__call__(gray);
      return { n: page.lines.length, first: page.lines.slice(0, 2).map((l) => l.text) };
    }, images[0].startsWith("data:") ? images[0] : `data:${MIME[path.extname(images[0]).toLowerCase()] ?? "image/png"};base64,${await readFile(images[0], "base64")}`, `http://127.0.0.1:${port2}/`);
    console.log(`staged hf-repo layout: ${result.n} lines, first: ${JSON.stringify(result.first)}`);
    if (!result.n) failed++;
  } finally {
    await browser2.close();
    server2.close();
  }
}
server.close();
process.exit(failed ? 1 : 0);
