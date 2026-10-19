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
//   base    URL serving the model files (det/rec/order/script/rec_cjk/orient
//           .onnx + meta.json), e.g. an HF ".../resolve/<rev>" URL, or a HF
//           repo as "user/repo" (revision defaults to "main") or
//           "user/repo@rev" to pin one
//   opts.cjk / opts.order / opts.orient   skip those models (no download)
//   opts.wasmPaths     ort.env.wasm.wasmPaths override
import { InferenceSession, Tensor, env as ortEnv } from "onnxruntime-web";
import { IndraOCR } from "./pipeline.mjs";

export { IndraOCR };

async function fetchBuf(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${url}`);
  return new Uint8Array(await res.arrayBuffer());
}

async function makeSession(base, name) {
  const buf = await fetchBuf(base + name);
  const sess = await InferenceSession.create(buf, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  return { run: (inputs) => sess.run(inputs) };
}

// "user/repo" or "user/repo@rev" → the HF resolve URL for the repo.  Anything
// else (a full URL with a scheme, a path with extra segments, ...) is returned
// unchanged.
function resolveBase(base) {
  const m = /^([^/@:\s]+)\/([^/@:\s]+)(?:@(.+))?$/.exec(base);
  return m
    ? `https://huggingface.co/${m[1]}/${m[2]}/resolve/${m[3] || "main"}`
    : base;
}

export async function loadIndra(base, opts = {}) {
  base = resolveBase(base);
  if (!/\/$/.test(base)) base += "/";
  if (opts.wasmPaths) ortEnv.wasm.wasmPaths = opts.wasmPaths;

  const backend = {
    Tensor,
    meta: async () => await (await fetch(base + "meta.json")).json(),
    session: (name) => makeSession(base, name),
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
    out[i] = Math.round(0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2]);
  }
  bmp.close();
  return { data: out, width, height };
}
