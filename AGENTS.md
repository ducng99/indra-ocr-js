# AGENTS.md

## What this is

A JavaScript library (browser + Node) running the [indra-ocr](https://github.com/datalab-to/indra) ONNX models (line detection, recognition Latin + CJK, reading order, quarter-turn orientation) — in the browser on onnxruntime-web's wasm backend, in Node on onnxruntime-node. **There is no build, lint, or typecheck step** — `src/*.mjs` is plain ESM published as-is (`exports`: `.` → `src/index.mjs`, `./browser` → `src/browser.mjs`; `sideEffects: false`). Don't invent `npm run build`/`lint`; don't introduce environment-specific APIs into the shared `src/` modules.

## Commands

```sh
npm install
npm test                # smoke test: renders a synthetic page, OCRs it in headless Chrome
npm run test:node       # same smoke test for the Node entry, on onnxruntime-node
npm test page.png ...   # run on real image files instead
INDRA_OCR_BIN=/path/to/indra-ocr npm test page.png ...   # + parity check vs the Python CLI
```

- The test entries are `test/browser.mjs` (puppeteer) and `test/node.mjs` (onnxruntime-node), sharing the model fetch in `test/models.mjs`. There is no unit-test framework and no way to run a subset.
- Requires network on first run: it downloads six `.onnx` files + `meta.json` (~18 MB) into `test/models/` (gitignored) and puppeteer's Chrome.
- Exit code is non-zero on failure; with `INDRA_OCR_BIN` set it also fails if the JS line count differs from Python's (the Python CLI runs with the same models via `--model-dir test/models`).
- How the test works (matters when editing it): it serves the repo root over localhost, builds a host page whose importmap resolves the bare `"onnxruntime-web"` import to the jsdelivr CDN bundle, imports `src/browser.mjs` straight from source, and drives `window.__test(img, base, opts)` in the page. Real image paths outside the repo root are passed as data URLs.

## Architecture

### Backend injection

`src/pipeline.mjs` is pure ESM: no Node or browser APIs, no onnxruntime import. `IndraOCR.create(backend, opts)` receives a backend `{ Tensor, meta: async () => metaJson, session: async (name) => { run(inputs) } }`. Two entries build backends on it: `src/index.mjs` (the default export, Node — onnxruntime-node, `readGray()` file/Buffer → grayscale via sharp, `base` is a local directory or URL, `opts.providers` the EPs) and `src/browser.mjs` (`@ducng99/indra-ocr-js/browser` — onnxruntime-web/wasm, `readGray()` URL/Blob → grayscale via canvas, `base` is a URL, `opts.wasmPaths`). They export the same `{ loadIndra, readGray, IndraOCR }`; keep `readGray`'s Rec.601 luma loop identical in both. Keep this split: anything environment-specific belongs in an entry (`index.mjs`/`browser.mjs`) or an injected backend, never in `pipeline.mjs`.

### Parity with the Python package (the core constraint)

All pre/post-processing is ported 1:1 from the Python `indra-ocr` package; the mirrored Python file is the spec for each module (see the table in README.md):

- `det.mjs` ← `det_post.py`, `rec.mjs` ← `rec_preproc.py`, `order.mjs` ← `order_post.py`, `vocab.mjs` ← `vocab.py`, `pipeline.mjs` ← `pipeline.py`, `imaging.mjs` ← cv2 equivalents.

Consequences for code changes:

- Use `roundHalfEven` (not `Math.round`) wherever Python rounds; `resizeGray` reimplements cv2's separable INTER_AREA/INTER_LINEAR taps; grayscale uses Rec.601 luma. These exist purely to keep model inputs numerically close to Python/cv2.
- Text vs Python should differ only on boundary glyphs (`1`/`l`, `0`/`O`, …); anything beyond that means a porting regression.

### Pipeline flow (`IndraOCR.run`)

1. `orientation()` — `orient.onnx` predicts quarter-turns CCW; the page is corrected with `rotQuarterCW` (skippable, and returns 0 when the model is absent).
2. `detect()` — `det.onnx` → `decodeBoxes`: connected components (4-connectivity) of the core-logit map at half resolution, line top/bottom from distance maps. Optional adaptive second pass at higher resolution when the median line height is small (≥ 8 boxes found and `medH < ADAPTIVE_MIN_H`).
3. `recognize()` — crops each line (padded by `PAD_X`/`PAD_Y` × line height), resizes to fixed height, groups crops into width-similar `widthBatches`, CTC-decodes (`blank = 0`, collapse repeats). With CJK enabled, `script.onnx` first classifies lines; hanzi lines go to `rec_cjk.onnx` with its own vocab and produce char tokens instead of ids. Low-confidence lines (< `MIN_CONF`) come back empty.
4. `#assemble()` — splits lines at wide token gaps or column gutters using the token x-positions (`toks`), then applies reading order.

Reading order: `order.onnx` outputs pairwise ordering logits plus a same-block head and block classes. `decodeBlocksOrder` clusters lines into blocks (average linkage over same-block logits), orders lines within a block geometrically (XY-cut; top-down for Table/Form/TOC classes), and orders blocks by mean pairwise logits. Without the model it degrades to pure geometric `xyCut`.

### Data shapes

- Images flow through the pipeline as grayscale `{data: Uint8Array (row-major), width, height}`; model inputs are float32 ink maps (`1 - gray/255`).
- Detection boxes are `[x0, y0, x1, y1, score]` in page pixels.
- Recognition returns `{text, conf, toks}` per box, where `toks` is `[vocab id | char, page x]` at the CTC peaks — these x-positions drive line splitting and gutter cuts.

### Models are external

The six `.onnx` files + `meta.json` are never in the package or repo; they are loaded at runtime from a base URL (or local directory in Node). `opts.{order, orient, cjk}: false` skips those downloads, and `IndraOCR.create` also silently degrades when optional model files are missing from the base URL (order → geometric fallback, orient → no rotation, no CJK → Latin-only). `meta.json` provides the vocabularies and det/rec geometry (`det_long_side`, `rec_height`).
