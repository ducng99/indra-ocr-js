// Shared by the test entries: the pinned ONNX model set and a once-per-run
// fetch into test/models/ (gitignored).  Both test/browser.mjs and
// test/node.mjs import this so they exercise identical model files.
import { mkdir, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const models = fileURLToPath(new URL("./models/", import.meta.url));
export const MODEL_BASE = "https://huggingface.co/datalab-to/indra-ocr/resolve/v1.1.0";
export const MODEL_FILES = [
  "det.onnx",
  "rec.onnx",
  "order.onnx",
  "script.onnx",
  "rec_cjk.onnx",
  "rec_rtl.onnx",
  "rec_world.onnx",
  "orient.onnx",
  "lm.npz",
  "lm_cjk.npz",
  "lm_world.npz",
  "mathcls.onnx",
  "math_enc.onnx",
  "math_dec_init.onnx",
  "math_dec_step.onnx",
  "math_meta.json",
  "meta.json",
];

export async function ensureModels() {
  await mkdir(models, { recursive: true });
  for (const f of MODEL_FILES) {
    const dest = path.join(models, f);
    if (existsSync(dest)) {
      console.log(`have   ${f}`);
      continue;
    }
    const url = `${MODEL_BASE}/${f}`;
    console.log(`get    ${url}`);
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${url}`);
    const buf = Buffer.from(await res.arrayBuffer());
    await writeFile(dest, buf);
    console.log(`saved  ${f} (${(buf.length / 1e6).toFixed(1)} MB)`);
  }
}
