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

const notFound = new Set(); // optional model files the loader skips by design
const server = createServer(async (req, res) => {
  try {
    const p = path.normalize(path.join(root, decodeURIComponent(new URL(req.url, "http://x").pathname)));
    if (!p.startsWith(root)) throw new Error();
    let data = await readFile(p.endsWith("/") ? p + "index.html" : p);
    // Pages/workers without the importmap: for ?worker, point the bare import at the CDN.
    if (new URL(req.url, "http://x").searchParams.has("worker"))
      data = data.toString().replace(/from "onnxruntime-web"/g, `from "${ORT_URL}"`);
    res.writeHead(200, {
      "content-type": MIME[path.extname(p).toLowerCase()] ?? "application/octet-stream",
      // the host page is about:blank (origin "null"), so module imports need CORS
      "access-control-allow-origin": "*",
    });
    res.end(data);
  } catch {
    notFound.add(decodeURIComponent(new URL(req.url, "http://x").pathname));
    res.writeHead(404, {
      // as above: the host page is about:blank, so even its 404s need CORS
      "access-control-allow-origin": "*",
    }).end();
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
    const s = JSON.stringify([
      base,
      opts.cjk ?? true,
      opts.order ?? true,
      opts.orient ?? true,
      opts.math ?? false,
      opts.tableHeads ?? false,
    ]);
    if (sig !== s) {
      ocr = await loadIndra(base, opts);
      sig = s;
      window.__device = ocr.device;
    }
    const gray = await readGray(img);
    const page = await ocr.run(gray);
    return {
      lines: page.lines,
      rotation: page.rotation,
      skew: page.skew,
      markdown: page.markdown,
      text: page.text,
      device: window.__device,
    };
  };
</script></head><body></body></html>`;

async function openTestPage(browser, libUrl) {
  const page = await browser.newPage();
  // A 404 the server itself produced is an optional model file the loader
  // skips by design (lm_<group>.npz when a group has none) — informational.
  page.on("console", (m) => m.type() === "error" && !m.text().includes("favicon") && !(notFound.size && m.text().includes("Failed to load resource")) && console.error("[page error]", m.text()));
  page.on("pageerror", (e) => console.error("[pageerror]", e.message));
  page.on("requestfailed", (r) => {
    const u = new URL(r.url(), "http://x").pathname;
    if (notFound.has(u)) return console.error("[optional model 404]", u);
    console.error("[request failed]", r.url(), r.failure()?.errorText);
  });
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
    if (img === images[0]) {
      // One extra pass with the math models and the table heads on: the
      // markdown path must produce output in the browser too.
      const r2 = await page.evaluate(
        async (u, base) => await window.__test(u, base, { math: true, tableHeads: true }),
        dataUrl, `http://127.0.0.1:${port}/test/models/`
      );
      const md = r2.markdown ?? "";
      const ok = md.length > 0;
      console.log(`  math+heads pass: markdown ${md.length} chars, "${md.split("\n")[0].slice(0, 60)}"`);
      if (!ok) failed++;
    }
  }
  await page.close();

  // readGray: transparency over white, maxSide downscaling, and in a worker.
  // Workers need a real origin (the OCR host page is about:blank), so this
  // runs on a page served from the test server.
  const rgPage = await browser.newPage();
  await rgPage.goto(`http://127.0.0.1:${port}/package.json`);
  const rg = await rgPage.evaluate(async (workerUrl) => {
    const c = document.createElement("canvas");
    c.width = 400;
    c.height = 200;
    const x = c.getContext("2d"); // transparent background, black text
    x.fillStyle = "#000";
    x.font = "40px serif";
    x.fillText("Transparent", 20, 100);
    const blob = await new Promise((r) => c.toBlob(r, "image/png"));
    const { readGray } = await import(workerUrl);
    const full = await readGray(blob);
    const small = await readGray(blob, { maxSide: 100 });
    const big = await readGray(blob, { maxSide: 4000 });
    // The worker imports the library itself (no DOM, no importmap — the server
    // rewrites the bare onnxruntime-web import for ?worker), then reads in place.
    const src = `import(${JSON.stringify(workerUrl)}).then(({ readGray }) => {
      onmessage = async (e) => {
        try {
          const g = await readGray(e.data, { maxSide: 100 });
          postMessage({ w: g.width, h: g.height, corner: g.data[0] });
        } catch (err) { postMessage({ err: String(err) }); }
      };
      postMessage({ ready: true });
    }, (err) => postMessage({ err: "import: " + err }));`;
    const w = new Worker(URL.createObjectURL(new Blob([src], { type: "text/javascript" })), { type: "module" });
    const inWorker = await new Promise((r) => {
      w.onerror = (e) => r({ err: `worker error: ${e.message}` });
      w.onmessage = (e) => {
        if (e.data.ready) w.postMessage(blob);
        else r(e.data);
      };
    });
    w.terminate();
    return {
      full: [full.width, full.height, full.data[0], Math.min(...full.data)],
      small: [small.width, small.height],
      big: [big.width, big.height],
      inWorker,
    };
  }, `http://127.0.0.1:${port}/src/browser.mjs?worker`);
  const rgOk = rg.full.slice(0, 3).join() === "400,200,255" && rg.full[3] < 64 && rg.small.join() === "100,50" &&
    rg.big.join() === "400,200" && rg.inWorker.w === 100 && rg.inWorker.h === 50 && rg.inWorker.corner === 255;
  console.log(`readGray: ${rgOk ? "ok" : "FAILED"} ${JSON.stringify(rg)}`);
  if (!rgOk) failed++;
  await rgPage.close();
  if (!images.length)
    console.log("(no images passed — smoke-tested a synthetic page; pass paths to test real ones)");
} finally {
  await browser.close();
}
server.close();
process.exit(failed ? 1 : 0);
