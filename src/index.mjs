// @ducng99/indra-ocr-js: the indra-ocr ONNX models in Node, on onnxruntime-node.
// The pipeline in ./pipeline.mjs is pure ESM; this entry wires it to
// ort-node and adds sharp-based image decoding.  For the browser entry
// (onnxruntime-web), import "@ducng99/indra-ocr-js/browser".
//
//   import { loadIndra, readGray } from "@ducng99/indra-ocr-js";
//   const ocr = await loadIndra(MODEL_BASE);   // see README: hosting the models
//   const page = await ocr.run(await readGray("page.png"));
//
// loadIndra(base, opts):
//   base    where the model files live (det/rec/order/script/rec_cjk/orient
//           .onnx + meta.json): a local directory, an HF ".../resolve/<rev>"
//           URL, or a HF repo as "user/repo" (revision defaults to "main") or
//           "user/repo@rev" to pin one
//   opts.cjk / opts.order / opts.orient   skip those models (no load)
//   opts.providers     onnxruntime execution providers (default ["cpu"])
import { readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { InferenceSession, Tensor } from "onnxruntime-node";
import sharp from "sharp";
import { IndraOCR } from "./pipeline.mjs";

export { IndraOCR };

async function fetchBuf(url) {
  if (/^https?:$/.test(url.protocol)) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${url}`);
    return new Uint8Array(await res.arrayBuffer());
  }
  return new Uint8Array(await readFile(url));
}

async function makeSession(base, name, providers) {
  const buf = await fetchBuf(new URL(name, base));
  const sess = await InferenceSession.create(buf, {
    executionProviders: providers,
    graphOptimizationLevel: "all",
  });
  return { run: (inputs) => sess.run(inputs) };
}

// "user/repo" or "user/repo@rev" → the HF resolve URL for the repo.  Anything
// else (a full URL with a scheme, a path with extra segments, ...) is returned
// unchanged.
function resolveHf(base) {
  const m = /^([^/@:\s]+)\/([^/@:\s]+)(?:@(.+))?$/.exec(base);
  return m
    ? `https://huggingface.co/${m[1]}/${m[2]}/resolve/${m[3] || "main"}`
    : base;
}

// base → a directory URL to load model files from.  http/https/file URLs pass
// through; otherwise it is a local directory (those win over the HF shorthand,
// so "test/models" stays a folder), falling back to the HF "user/repo[@rev]"
// expansion.
function resolveBaseDir(base) {
  if (/^(https?|file):/i.test(base)) base = new URL(base).href;
  else if (existsSync(base)) base = pathToFileURL(base).href;
  else base = resolveHf(base);
  return base.endsWith("/") ? base : base + "/";
}

export async function loadIndra(base, opts = {}) {
  const providers = opts.providers ?? ["cpu"];
  const dir = resolveBaseDir(base);
  const backend = {
    Tensor,
    meta: async () => JSON.parse(new TextDecoder().decode(await fetchBuf(new URL("meta.json", dir)))),
    session: (name) => makeSession(dir, name, providers),
  };
  const ocr = await IndraOCR.create(backend, opts);
  ocr.device = providers.join("+");
  return ocr;
}

// Decode an image file path or Buffer to the grayscale {data, width, height}
// the pipeline takes, with cv2's Rec.601 luma (see the luma note in
// src/imaging.mjs).  sharp decodes; the pixel loop mirrors the browser entry.
export async function readGray(source) {
  const { data, info } = await sharp(source).raw().toBuffer({ resolveWithObject: true });
  const { width, height, channels } = info;
  const out = new Uint8Array(width * height);
  for (let i = 0, p = 0; i < out.length; i++, p += channels) {
    out[i] = channels >= 3
      ? Math.round(0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2])
      : data[p];
  }
  return { data: out, width, height };
}
