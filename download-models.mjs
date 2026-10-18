// Downloads the indra-ocr ONNX models from the Hugging Face Hub.
import { mkdir, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";

const REPO = "datalab-to/indra-ocr";
const REVISION = "v3.1.1"; // the model set the Python package is built against
const OUT = path.join(import.meta.dirname, "models");
const FILES = ["det.onnx", "rec.onnx", "order.onnx", "script.onnx", "rec_cjk.onnx", "orient.onnx", "meta.json", "config.json"];

await mkdir(OUT, { recursive: true });
for (const f of FILES) {
  const dest = path.join(OUT, f);
  if (existsSync(dest)) {
    console.log(`have   ${f}`);
    continue;
  }
  const url = `https://huggingface.co/${REPO}/resolve/${REVISION}/${f}`;
  console.log(`get    ${url}`);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${url}`);
  const buf = Buffer.from(await res.arrayBuffer());
  await writeFile(dest, buf);
  console.log(`saved  ${f} (${(buf.length / 1e6).toFixed(1)} MB)`);
}
