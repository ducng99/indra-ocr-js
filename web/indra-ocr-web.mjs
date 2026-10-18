// indra-ocr in the browser, on onnxruntime-web (the same ONNX Runtime that
// transformers.js uses in the browser; here used directly, since raw custom
// sessions are outside transformers.js's API).  Runs on WebGPU or wasm.
//
//   import { loadIndra } from "./indra-ocr-web.mjs";
//   const ocr = await loadIndra("https://huggingface.co/USER/REPO/resolve/main");
//   const page = await ocr.__call__(gray);   // gray: {data, width, height}
//
// loadIndra(base, opts):
//   base    model file base URL: an HF ".../resolve/main" URL, or a same-origin path
//   opts.device        "auto" (default) | "webgpu" | "wasm"
//   opts.cjk / opts.order / opts.orient   skip those models (no download)
//   opts.wasmPaths     ort.env.wasm.wasmPaths override
//
// "auto" picks WebGPU when navigator.gpu reports an adapter, else wasm.  A
// "webgpu" session that fails to initialize or run (e.g. an op the WebGPU EP
// does not implement without its wasm fallback) is transparently rebuilt on
// wasm, so WebGPU never breaks a page that wasm can do.
import { InferenceSession, Tensor, env as ortEnv } from "onnxruntime-web";
import { IndraOCR } from "../src/pipeline.mjs"; // staged as ./src/ by prepare-repo.mjs

async function fetchBuf(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${url}`);
  return new Uint8Array(await res.arrayBuffer());
}

async function makeSession(base, name, eps) {
  const buf = await fetchBuf(base + name);
  const create = (providers) =>
    InferenceSession.create(buf, { executionProviders: providers, graphOptimizationLevel: "all" });
  let gpu = eps.includes("webgpu");
  let sess;
  try {
    sess = await create(eps);
  } catch (e) {
    if (!gpu) throw e;
    console.warn(`indra-ocr: ${name}: WebGPU init failed (${e?.message ?? e}); using wasm`);
    gpu = false;
    sess = await create(["wasm"]);
  }
  return {
    async run(inputs) {
      try {
        return await sess.run(inputs);
      } catch (e) {
        if (!gpu) throw e;
        console.warn(`indra-ocr: ${name}: WebGPU run failed (${e?.message ?? e}); switching to wasm`);
        gpu = false;
        sess = await create(["wasm"]);
        return await sess.run(inputs);
      }
    },
  };
}

export async function loadIndra(base, opts = {}) {
  if (!/\/$/.test(base)) base += "/";
  if (opts.wasmPaths) ortEnv.wasm.wasmPaths = opts.wasmPaths;

  // Which device?  "auto" probes for a WebGPU adapter first.
  let useGpu = false;
  const want = opts.device ?? "auto";
  if (want === "webgpu") {
    useGpu = true;
  } else if (want === "auto" && typeof navigator !== "undefined" && navigator.gpu) {
    try {
      useGpu = !!(await navigator.gpu.requestAdapter());
    } catch {
      /* no adapter -> wasm */
    }
  }
  const eps = useGpu ? ["webgpu", "wasm"] : ["wasm"]; // wasm kept for op-level fallback

  const backend = {
    Tensor,
    meta: async () => await (await fetch(base + "meta.json")).json(),
    session: (name) => makeSession(base, name, eps),
  };
  const ocr = await IndraOCR.create(backend, opts);
  ocr.device = useGpu ? "webgpu" : "wasm";
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
