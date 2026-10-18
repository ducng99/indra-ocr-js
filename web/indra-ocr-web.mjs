// indra-ocr in the browser, on onnxruntime-web (the same ONNX Runtime that
// transformers.js uses in the browser; here used directly, since raw custom
// sessions are outside transformers.js's API).  Runs on the wasm backend.
//
//   import { loadIndra } from "./indra-ocr-web.mjs";
//   const ocr = await loadIndra("https://huggingface.co/USER/REPO/resolve/main");
//   const page = await ocr.__call__(gray);   // gray: {data, width, height}
//
// loadIndra(base, opts):
//   base    model file base URL: an HF ".../resolve/main" URL, or a same-origin path
//   opts.cjk / opts.order / opts.orient   skip those models (no download)
//   opts.wasmPaths     ort.env.wasm.wasmPaths override
import { InferenceSession, Tensor, env as ortEnv } from "onnxruntime-web";
import { IndraOCR } from "../src/pipeline.mjs"; // staged as ./src/ by prepare-repo.mjs

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

export async function loadIndra(base, opts = {}) {
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
// takes, with cv2's Rec.601 luma (see the Rec.601 note in src/imaging.mjs).
export async function readGrayBrowser(source) {
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
