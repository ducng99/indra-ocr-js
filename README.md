# indra-ocr-js

The [indra-ocr](https://github.com/datalab-to/indra) ONNX models in JavaScript — line detection, recognition (Latin + CJK), reading order, orientation, math — in the browser on onnxruntime-web (wasm), in Node on onnxruntime-node. Models stream at runtime from the [Hugging Face Hub](https://huggingface.co/datalab-to/indra-ocr) (cached) or any URL / local folder.

## Install

```sh
npm install @ducng99/indra-ocr-js
```

ESM only, Node ≥ 24.

## Usage

Node — `import "@ducng99/indra-ocr-js"`; browser — `import "@ducng99/indra-ocr-js/browser"`:

```js
import { loadIndra, readGray } from "@ducng99/indra-ocr-js";

const ocr = await loadIndra("datalab-to/indra-ocr"); // HF repo, URL or local dir
const page = await ocr.run(await readGray("page.png"));

page.text;      // every line, in reading order
page.markdown;  // headers, lists, tables, math
page.lines;     // [{ box: [x0, y0, x1, y1], text, conf }]
page.blocks;    // [{ label, bbox, text }] — Text, Section-Header, Table, ...
```

`readGray(source, { maxSide })` takes a path or Buffer in Node, a URL, Blob or `ImageBitmap` (anything `createImageBitmap` accepts) in the browser, and returns the grayscale `{ data, width, height }` the pipeline takes. Transparent pixels are composited over white. `maxSide` optionally downscales (aspect kept, never upscaled) so the longer side is at most that many pixels — e.g. `{ maxSide: 2048 }` for large phone photos. In the browser it uses `OffscreenCanvas`, so it also works inside a Web Worker.

`loadIndra(base, opts)` options:

- `{order, orient, cjk}: false` — skip those models (a missing file also degrades gracefully: XY-cut reading order, no rotation, Latin-only)
- `math: true` — read math lines as inline LaTeX (extra models)
- `tableHeads`, `furniture`, `adaptive` — table building from the layout heads, keep headers/footers, second detection pass
- `progress` — download callback, `{file, loaded, total}` per chunk
- `providers` (Node, default `["cpu"]`), `wasmPaths` (browser) — backend tuning

`ocr.run(gray, turn = true)` — pass `turn = false` to skip orientation/deskew. Lower-level: `ocr.detect(gray)`, `ocr.recognize(gray, boxes)`, `ocr.orientation(gray)`; custom runtimes can call `IndraOCR.create(backend)` with an injected `{Tensor, meta, session}`.

## Tests

```sh
npm test                # browser entry, headless Chrome
npm run test:node       # Node entry
npm test page.png ...   # real images; + INDRA_OCR_BIN=/path/to/indra-ocr for a parity check vs the Python CLI
```

First run downloads the pinned models into `test/models/` and puppeteer's Chrome.

## Porting

A 1:1 port of the Python `indra-ocr` package (`det.mjs` ← `det_post.py`, `rec.mjs` ← `rec_preproc.py` + `line_mask.py`, `order.mjs` ← `order_post.py`, `math.mjs` ← `math.py`, `layout.mjs` ← `layout_post.py`, `vocab.mjs` ← `vocab.py`, `rtl.mjs` ← `rtl.py` with `unicode.mjs` ← Python's `unicodedata`, `lm.mjs` ← `lm.py`, `pipeline.mjs` ← `pipeline.py`, `imaging.mjs` ← cv2), down to half-even rounding, cv2's resize taps and grayscale conversion, so model inputs stay numerically identical.

[`demo/`](demo/) is a client-side playground (`bun run build`), deployed to GitHub Pages and a HF Space by the **Deploy demo** workflow.

## License

[Apache-2.0](LICENSE)
