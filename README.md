# @ducng99/indra-ocr-js

OCR in JavaScript with the [indra-ocr](https://huggingface.co/datalab-to/indra-ocr) ONNX models: line detection, recognition (Latin + CJK), reading order, quarter-turn orientation, deskew, math-as-LaTeX, and markdown/layout blocks. Runs in Node on [onnxruntime-node](https://onnxruntime.ai/docs/get-started/node) and in the browser on [onnxruntime-web](https://onnxruntime.ai/docs/get-started/web)'s wasm backend — all locally, no server.

The pre- and post-processing is ported 1:1 from the [indra-ocr Python package](https://github.com/datalab-to/indra), and every module mirrors its Python source: `det.mjs` ← `det_post.py`, `rec.mjs` ← `rec_preproc.py`, `order.mjs` ← `order_post.py`, `math.mjs` ← `math.py`, `layout.mjs` ← `layout_post.py`, `vocab.mjs` ← `vocab.py`, `pipeline.mjs` ← `pipeline.py`.

`src/pipeline.mjs` is pure ESM (no Node or browser APIs); the ONNX sessions and `Tensor` are injected by a backend, which `src/index.mjs` builds on onnxruntime-node (the default entry) and `src/browser.mjs` on onnxruntime-web (`@ducng99/indra-ocr-js/browser`).

## Install

```sh
npm install @ducng99/indra-ocr-js
```

The six `.onnx` files plus `meta.json` (~18 MB) are not part of the package. Turning `math: true` adds five more files (~30 MB) that are fetched only then.

`loadIndra(base)` loads them at runtime from a base URL — or, in Node, a local directory — you choose.

Hugging Face `resolve` endpoints send permissive CORS headers, so the pinned upstream revision works out of the box.

`base` also accepts a HF repo shorthand: `"user/repo"` expands to `https://huggingface.co/user/repo/resolve/main`, and `"user/repo@rev"` pins a revision — `loadIndra("datalab-to/indra-ocr@v3.1.1")` is the URL above.

For production, upload the model files (they are the only thing you need to host — no code) to your own repo or any CORS-enabled static host, so upstream changes can never break your app.

The Node entry's native dependencies (`onnxruntime-node`, `sharp`) are optional: browser-only installs can skip them with `npm install --omit=optional`.

## Usage

Node (the default import; `base` may be a local directory):

```js
import { loadIndra, readGray } from "@ducng99/indra-ocr-js";

const ocr = await loadIndra(
  "https://huggingface.co/datalab-to/indra-ocr/resolve/v3.1.1"
);
const page = await ocr.run(await readGray("page.png"));
console.log(page.markdown);
// page: {
//   lines: [{box: [x0, y0, x1, y1], text, conf}],  // reading order
//   rotation,      // quarter turns undone (90/180/270), 0 when upright
//   skew,          // degrees of tilt undone, 0.0 when straight
//   markdown,      // page markdown (needs order.onnx's layout heads)
//   blocks,        // [{label, bbox, text}] in reading order
//   text,          // the lines, newline-joined
//   timings,       // {orientMs, detMs, recMs, orderMs}
// }
```

Browser (`@ducng99/indra-ocr-js/browser`; without a bundler, add an importmap that resolves the bare `onnxruntime-web` import to the jsdelivr `ort.wasm.bundle.min.mjs`):

```js
import { loadIndra, readGray } from "@ducng99/indra-ocr-js/browser";
// same API as above
```

`loadIndra(base, opts)` takes these options:

- `cjk: false` skips `script.onnx` and `rec_cjk.onnx` (about 9.5 MB smaller, Latin-only output).
- `order: false` skips `order.onnx` and falls back to geometric reading order (no markdown).
- `orient: false` skips `orient.onnx` and disables rotation detection and deskew.
- `math: true` reads math lines as inline LaTeX (`$x _ { i } \leq 1$`; needs the math models in the model folder, and throws when they are missing).
- `tableHeads: true` builds tables from the order model's row / column / cell heads instead of from geometry.
- `furniture: true` keeps page headers and footers in the markdown (dropped by default).
- `providers` (Node) overrides the onnxruntime execution providers (default `["cpu"]`).
- `wasmPaths` (browser) overrides where onnxruntime-web loads its `.wasm` binaries from.

Missing optional model files degrade silently instead of throwing.

Without COOP/COEP headers the wasm backend runs single-threaded.

## API

Both entries export the same `{ loadIndra, readGray, IndraOCR }`. `ocr.detect(gray)` returns line boxes, `ocr.recognize(gray, boxes)` returns per-line `{text, conf, toks}`, `ocr.readingOrder(boxes, texts, [w, h])` returns `{order, layout}` (the layout heads' logits when the model has them), and `ocr.orientation(gray)` returns detected quarter turns. `readGray` takes a URL or Blob in the browser, a file path or Buffer in Node.

`IndraOCR.create({ Tensor, meta, json, session }, opts)` builds the pipeline on a custom backend, for example sharing a transformers.js `Tensor`. (`json` is only needed for `math: true`, to read `math_meta.json`.)

## Scope

CenterMask line masking (`mask.onnx`, not in the current model revision) and PDF input are not ported from Python; PDF pages stay the caller's job (render one to a canvas or a Buffer and hand the image to `readGray`).

Output text matches the Python CLI except for occasional boundary glyphs (`1`/`l`, `0`/`O`), because the JS preprocessing is not bit-identical to cv2.

## Development

```sh
npm install
npm test                   # synthetic-page smoke test in headless Chrome
npm test page.png ...      # real images; INDRA_OCR_BIN=/path/to/indra-ocr for parity check
```

Node >= 24 is required; the models, the optional native dependencies, and Puppeteer's Chrome download on first run.
