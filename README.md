# indra-ocr in the browser (ONNX Runtime: wasm)

Can the [datalab-to/indra-ocr](https://huggingface.co/datalab-to/indra-ocr) `.onnx`
files be used with [transformers.js](https://huggingface.co/docs/transformers.js)?

**Through the standard transformers.js API — no.** The repo ships custom
architectures (`model_type: "indra-ocr"`), not transformers models: there is no
transformers `config.json`/weights layout and no `onnx/model.onnx`, so
`pipeline()` and `AutoModel.from_pretrained()` both fail.

**On transformers.js's ONNX Runtime backend — yes.** transformers.js is built
on onnxruntime-web in the browser; this directory runs the raw
`det.onnx` / `rec.onnx` / `order.onnx` / `script.onnx` + `rec_cjk.onnx` /
`orient.onnx` sessions on that runtime directly (wasm), with the
pre- and post-processing ported 1:1 from the Python package:

| Python | here |
| --- | --- |
| `det_post.py` (`page_to_input`, `decode`) | `src/det.mjs` |
| `rec_preproc.py` (`crop_line`, `pad_batch`, `width_batches`) | `src/rec.mjs` |
| `order_post.py` (`line_features`, `decode_order`, `decode_blocks_order`, `_cluster`, XY-cut) | `src/order.mjs` |
| `vocab.py` (`Vocab.decode`) | `src/vocab.mjs` |
| `pipeline.py` (`OrtOCR`: detect / recognize / gutter cuts / line split / assemble / orientation) | `src/pipeline.mjs` |
| cv2 resize / grayscale / rot90 | `src/imaging.mjs` (separable INTER_AREA/INTER_LINEAR taps, Rec.601 luma) |
| `cv2.imread(..., IMREAD_GRAYSCALE)` | `readGrayBrowser()` in `web/indra-ocr-web.mjs` |

`src/` is pure ESM (no Node/browser APIs); the sessions and `Tensor` are
injected by a backend — `web/indra-ocr-web.mjs` builds one on
onnxruntime-web. (In a bundler app that also uses transformers.js you can
inject its `Tensor` the same way via `IndraOCR.create(backend, opts)`.)

## Try it

```sh
npm install
npm run download      # det/rec/order/script/rec_cjk/orient + meta.json from the Hub (v3.1.1)
npm run serve         # -> http://127.0.0.1:8080/web/demo.html
```

The demo runs everything on the wasm backend. `npm run test:browser` runs the
demo headless in Chrome and compares against the Python CLI (when the dev
venv's `indra-ocr` is present).

## Use from your own Hugging Face repo

```sh
node web/prepare-repo.mjs YOUR-USERNAME/indra-ocr-web   # -> js/hf-repo/
huggingface-cli upload YOUR-USERNAME/indra-ocr-web js/hf-repo/* . --repo-type model
```

`js/hf-repo/` gets the six `.onnx` files + `meta.json`, the pure ESM library
(`indra-ocr-web.mjs` + `src/`), and a README. Your app then needs only:

```html
<script type="importmap">
{ "imports": {
    "onnxruntime-web": "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0-dev.20250409-89f8206ba4/dist/ort.wasm.bundle.min.mjs"
} }
</script>
<script type="module">
  import { loadIndra, readGrayBrowser } from
    "https://huggingface.co/YOUR-USERNAME/indra-ocr-web/resolve/main/indra-ocr-web.mjs";
  const ocr = await loadIndra("https://huggingface.co/YOUR-USERNAME/indra-ocr-web/resolve/main");
  const page = await ocr.__call__(await readGrayBrowser("page.png"));
  console.log(page.text);   // page.lines: [{box, text, conf}], page.rotation
</script>
```

API: `loadIndra(base, opts)` with
`opts.cjk / opts.order / opts.orient` (skip optional models),
`opts.wasmPaths` (ort wasm location override);
`ocr.device` is always `"wasm"`.
In a bundler/npm app, skip the importmap and `npm i onnxruntime-web`.

## Parity with the Python package

`npm run test:browser` (headless Chrome, wasm): line counts, line splitting
and reading order match the Python CLI exactly; residual text differences are
boundary-case glyphs (`1`/`l`, `|`/`I`, `0`/`O`, `,`/`.`) — the JS
preprocessing is not bit-identical to cv2 (independent float accumulation,
±2/255 on the ink maps), enough to flip an occasional low-margin CTC peak.

```
table_pmc.png:     188 lines, identical 160 (85.1%)
chinese_essay.png:  44 lines, identical  35 (79.5%)   (rec_cjk path)
multi_column.png:  138 lines, identical 124 (89.9%)*   (*measured earlier via the Node runner, same pure src/)
rotated_letter.png: 98 lines, identical  89 (90.8%)*
```

## Not ported

- `estimate_skew` / deskew (small tilts are left alone; quarter-turn
  orientation **is** ported via `orient.onnx`)
- `mask.onnx` CenterMask (not in this model revision)
- the math models (`math_*.onnx`)
- markdown / block assembly from the order model's layout heads (lines and
  reading order are; `BLOCK_CLASSES` is exported in `src/order.mjs`)
- PDF input, the text layer, `parallel.py`, the viewer

## Notes

- **Standalone.** This directory is a self-contained repo root: the library
  and the staged `hf-repo/` depend on nothing outside it (runtime deps come
  from the CDN/npm; models are fetched from the Hub by `npm run download`).
  Dev-only extras: `test:browser` uses `../assets/examples/*` of a
  surrounding indra checkout when present (otherwise it smoke-tests a
  synthetic page; pass your own images as arguments) and compares against the
  Python CLI when `INDRA_OCR_BIN` points at an `indra-ocr` executable.
- The Node runner (`ocr.mjs`/`compare.mjs`, `onnxruntime-node` + `sharp`) was
  removed — this is browser-only now. The remaining Node scripts
  (`download-models.mjs`, `web/serve.mjs`, `web/prepare-repo.mjs`,
  `web/test-browser.mjs`) are dev tooling only.
- HF `resolve` endpoints send `Access-Control-Allow-Origin: *`, so the models
  (LFS/CDN redirects included) fetch fine from any page.
- Without cross-origin isolation (COOP/COEP headers) ort-web runs the wasm
  backend single-threaded. `web/serve.mjs` sets the headers; browsers only
  honor them over HTTPS.
- ~18 MB of models download on first load (`cjk: false` saves ~9.5 MB;
  `orient: false`, `order: false` a bit more).
