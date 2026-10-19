// Node verification of the Node entry (src/index.mjs): fetches the pinned
// ONNX model set into test/models/ (gitignored, shared with test/browser.mjs)
// and runs OCR on onnxruntime-node's CPU EP.  With image paths as arguments —
// and INDRA_OCR_BIN pointing at the Python indra-ocr CLI, when available —
// the lines are compared against it.  Without arguments, a synthetic page is
// rasterized from SVG with sharp (the Node stand-in for the canvas render in
// test/browser.mjs).  Usage: node test/node.mjs [image ...]
import { spawn } from "node:child_process";
import path from "node:path";
import sharp from "sharp";
import { loadIndra, readGray } from "../src/index.mjs";
import { ensureModels, models } from "./models.mjs";

const PY = process.env.INDRA_OCR_BIN ?? null;

await ensureModels();

const ocr = await loadIndra(models);
let failed = 0;
{
  const images = process.argv.slice(2);

  for (const img of images.length ? images : [null]) {
    // Real images go straight to sharp by path; without one, render the same
    // synthetic page test/browser.mjs draws on its canvas (SVG text y is the
    // baseline, like canvas fillText).
    const source = img ?? Buffer.from(
      `<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="512">
  <rect width="1024" height="512" fill="#fff"/>
  ${["Hello indra-ocr, running in Node.", "Second line with numbers 1234567890.",
     "A third line, slightly longer, to split.", "And a fourth one for good measure."].map(
        (t, i) => `<text x="40" y="${80 + i * 60}" font-family="serif" font-size="32" fill="#111">${t}</text>`
      ).join("\n  ")}
</svg>`);
    const t0 = Date.now();
    const page = await ocr.run(await readGray(source));
    const ms = Date.now() - t0;
    const jsLines = page.lines.map((l) => l.text);
    let report = `${img ? path.basename(img) : "(synthetic)"} [${ocr.device}]: ${jsLines.length} lines in ${ms} ms, rotation ${page.rotation}`;
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
        if (jsLines.length !== pyLines.length || ocr.device !== "cpu") failed++;
      } catch {
        report += " | (python comparison failed)";
      }
    }
    console.log(report);
    console.log("  first lines:", JSON.stringify(jsLines.slice(0, 3)));
  }

  if (!images.length)
    console.log("(no images passed — smoke-tested a synthetic page; pass paths to test real ones)");
}
process.exit(failed ? 1 : 0);
