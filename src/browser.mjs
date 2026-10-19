// @ducng99/indra-ocr-js: the indra-ocr ONNX models in the browser, on onnxruntime-web
// (wasm backend).  The pipeline in ./pipeline.mjs is pure ESM; this entry
// wires it to ort-web and adds the browser image decoding.  (The default
// package entry, src/index.mjs, is the Node build on onnxruntime-node.)
//
//   import { loadIndra, readGray } from "@ducng99/indra-ocr-js/browser";
//   const ocr = await loadIndra(MODEL_BASE);   // see README: hosting the models
//   const page = await ocr.run(await readGray("page.png"));
//
// loadIndra(base, opts):
//   base    URL serving the model files (det/rec/order/orient/script/rec_<group>
//           .onnx + meta.json; + the math models with opts.math; + the n-gram
//           language models lm*.npz), e.g. an HF ".../resolve/<rev>" URL, or a
//           HF repo as "user/repo" (revision defaults to the release the
//           models were exported at) or "user/repo@rev" to pin one
//   opts.cjk / opts.order / opts.orient / opts.lm   skip those models (no
//           download); cjk covers the script classifier and the group recognizers
//   opts.math             read math lines as LaTeX (downloads the math models)
//   opts.recInt8          load rec.int8.onnx instead of rec.onnx
//   opts.threads          wasm worker threads (default 4; needs cross-origin
//                         isolation, otherwise onnxruntime-web falls back to 1)
//   opts.progress         callback for download progress: called with
//                         {file, loaded, total} per chunk (total may be null)
//                         and {file, loaded, total, done: true} per finished file
//   opts.tableHeads       build tables from the order model's table heads
//   opts.furniture        keep page headers and footers in the markdown
//   opts.wasmPaths     ort.env.wasm.wasmPaths override
import { InferenceSession, Tensor, env as ortEnv } from "onnxruntime-web";
import { IndraOCR } from "./pipeline.mjs";

export { IndraOCR };

// The release tag the published model files were exported at (py models.py
// HF_REVISION): the default revision of the "user/repo" shorthand.
const HF_REVISION = "v1.1.0";

// Stream the file so callers can track download progress.  `onChunk` gets
// (loaded, total) per chunk; total is null when there is no Content-Length.
async function fetchBuf(url, onChunk) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${url}`);
  if (!res.body || !onChunk) return new Uint8Array(await res.arrayBuffer());
  const total = res.headers.has("content-length") ? +res.headers.get("content-length") : null;
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onChunk(loaded, total);
  }
  const buf = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) {
    buf.set(c, off);
    off += c.byteLength;
  }
  return buf;
}

async function makeSession(base, name, getBuf) {
  const buf = await getBuf(name);
  const sess = await InferenceSession.create(buf, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  return {
    run: (inputs) => sess.run(inputs),
    inputNames: sess.inputNames,
    outputNames: sess.outputNames,
  };
}

// "user/repo" or "user/repo@rev" → the HF resolve URL for the repo.  Anything
// else (a full URL with a scheme, a path with extra segments, ...) is returned
// unchanged.
function resolveBase(base) {
  const m = /^([^/@:\s]+)\/([^/@:\s]+)(?:@(.+))?$/.exec(base);
  return m
    ? `https://huggingface.co/${m[1]}/${m[2]}/resolve/${m[3] || HF_REVISION}`
    : base;
}

export async function loadIndra(base, opts = {}) {
  const threads = opts.threads ?? 4;
  base = resolveBase(base);
  if (!/\/$/.test(base)) base += "/";
  if (opts.wasmPaths) ortEnv.wasm.wasmPaths = opts.wasmPaths;
  // Needs cross-origin isolation; onnxruntime-web falls back to 1 without it.
  ortEnv.wasm.numThreads = threads;
  const onProgress = typeof opts.progress === "function" ? opts.progress : null;

  // Download cache: one in-flight promise per file, all candidates started
  // concurrently.  Missing optional files reject inside the cache; the
  // pipeline's try/catch around the optional sessions handles that.
  const cache = new Map();
  const get = (name) => {
    if (!cache.has(name)) {
      const buf = fetchBuf(base + name, onProgress
        ? (loaded, total) => onProgress({ file: name, loaded, total })
        : null).then((b) => {
          onProgress?.({ file: name, loaded: b.byteLength, total: b.byteLength, done: true });
          return b;
        });
      cache.set(name, buf);
    }
    return cache.get(name);
  };
  const names = ["meta.json", "det.onnx", opts.recInt8 ? "rec.int8.onnx" : "rec.onnx"];
  if (opts.order !== false) names.push("order.onnx");
  if (opts.orient !== false) names.push("orient.onnx");
  if (opts.math) {
    names.push("mathcls.onnx", "math_enc.onnx", "math_dec_init.onnx", "math_dec_step.onnx", "math_meta.json");
  }
  await Promise.allSettled(names.map(get));

  // meta.json names the recognizer groups (script.onnx's classes): the group
  // recognizers rec_<g>.onnx and the language models lm*.npz follow from it.
  const decode = async (name) => JSON.parse(new TextDecoder().decode(await get(name)));
  let meta = null;
  try {
    meta = await decode("meta.json");
  } catch {
    /* IndraOCR.create will surface the missing meta.json */
  }
  const groups = (meta?.script_groups ?? ["base", "cjk", "rtl", "other"]).filter((g) => g !== "base");
  const more = [];
  if (opts.cjk !== false) more.push("script.onnx", ...groups.map((g) => `rec_${g}.onnx`));
  // The language models ship per group (lm.npz for the base recognizer,
  // lm_<group>.npz when a group has one).
  if (opts.lm !== false) {
    more.push("lm.npz");
    if (opts.cjk !== false) more.push(...groups.map((g) => `lm_${g}.npz`));
  }
  await Promise.allSettled(more.map(get));

  const backend = {
    Tensor,
    meta: () => decode("meta.json"),
    json: decode,
    bytes: async (name) => new Uint8Array(await get(name)),
    session: (name) => makeSession(base, name, get),
  };
  const ocr = await IndraOCR.create(backend, opts);
  ocr.device = "wasm";
  return ocr;
}

// Decode an image URL/Blob to the grayscale {data, width, height} the pipeline
// takes, with cv2's Rec.601 luma (see the luma note in src/imaging.mjs).
export async function readGray(source) {
  const blob = source instanceof Blob ? source : await (await fetch(source)).blob();
  const bmp = await createImageBitmap(blob);
  const c = document.createElement("canvas");
  c.width = bmp.width;
  c.height = bmp.height;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bmp, 0, 0);
  const { data, width, height } = ctx.getImageData(0, 0, bmp.width, bmp.height);
  const out = new Uint8Array(width * height);
  for (let i = 0, p = 0; i < out.length; i++, p += 4) {
    out[i] = (data[p] * 9797 + data[p + 1] * 19234 + data[p + 2] * 3737) >> 15;
  }
  bmp.close();
  return { data: out, width, height };
}
