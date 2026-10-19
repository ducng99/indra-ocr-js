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
//           .onnx + meta.json; + the math models with opts.math): a local
//           directory, an HF ".../resolve/<rev>" URL, or a HF repo as
//           "user/repo" (revision defaults to "main") or "user/repo@rev" to
//           pin one
//   opts.cjk / opts.order / opts.orient   skip those models (no load)
//   opts.math             read math lines as LaTeX (loads the math models)
//   opts.progress         callback for download progress: called with
//                         {file, loaded, total} per chunk (total may be null)
//                         and {file, loaded, total, done: true} per finished file
//   opts.tableHeads       build tables from the order model's table heads
//   opts.furniture        keep page headers and footers in the markdown
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

async function makeSession(base, name, providers, getBuf) {
  const buf = await getBuf(name);
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
  const onProgress = typeof opts.progress === "function" ? opts.progress : null;

  // Download cache: one in-flight promise per file, all candidates started
  // concurrently.  Missing optional files reject inside the cache; the
  // pipeline's try/catch around the optional sessions handles that.
  const cache = new Map();
  const get = (name) => {
    if (!cache.has(name)) {
      const buf = fetchBuf(new URL(name, dir), onProgress
        ? (loaded, total) => onProgress({ file: name, loaded, total })
        : null).then((b) => {
          onProgress?.({ file: name, loaded: b.byteLength, total: b.byteLength, done: true });
          return b;
        });
      cache.set(name, buf);
    }
    return cache.get(name);
  };
  const names = ["meta.json", "det.onnx", "rec.onnx"];
  if (opts.order !== false) names.push("order.onnx");
  if (opts.orient !== false) names.push("orient.onnx");
  if (opts.cjk !== false) names.push("script.onnx", "rec_cjk.onnx");
  if (opts.math) {
    names.push("mathcls.onnx", "math_enc.onnx", "math_dec_init.onnx", "math_dec_step.onnx", "math_meta.json");
  }
  await Promise.allSettled(names.map(get));

  const decode = async (name) => JSON.parse(new TextDecoder().decode(await get(name)));
  const backend = {
    Tensor,
    meta: () => decode("meta.json"),
    json: decode,
    session: (name) => makeSession(dir, name, providers, get),
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
