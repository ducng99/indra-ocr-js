// Stages the files to push to your own Hugging Face model repo: the ONNX
// models + meta.json + the browser library (pure ESM, no build step).
//
//   node js/web/prepare-repo.mjs [target-repo-id]
//   # -> js/hf-repo/  (gitignored), then push it:
//   huggingface-cli upload <target-repo-id> js/hf-repo/* . --repo-type model
//
// Your app then loads the whole thing straight from the Hub (CORS is open on
// the resolve endpoints):
//
//   import { loadIndra } from "https://huggingface.co/<repo>/resolve/main/indra-ocr-web.mjs";
//   const ocr = await loadIndra("https://huggingface.co/<repo>/resolve/main");
//
// (onnxruntime-web comes from your importmap or npm — see demo.html.  Caveat:
// if the Hub ever serves .mjs without a JavaScript MIME type, import the
// library from npm/your own server instead and only fetch the models from
// the Hub.)
import { mkdir, copyFile, writeFile, readFile, readdir, rm } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const jsDir = path.join(here, "..");
const models = path.join(jsDir, "models");
const out = path.join(jsDir, "hf-repo");
const repoId = process.argv[2] ?? "YOUR-USERNAME/indra-ocr-web";

if (!existsSync(path.join(models, "det.onnx"))) {
  console.error("js/models/ is empty — run `npm run download` first");
  process.exit(1);
}

await rm(out, { recursive: true, force: true });
await mkdir(out, { recursive: true });
await mkdir(path.join(out, "src"), { recursive: true });

// models + meta (skip the CJK pair only if you want a smaller Latin-only repo)
const files = ["det.onnx", "rec.onnx", "order.onnx", "script.onnx", "rec_cjk.onnx", "orient.onnx", "meta.json", "config.json"];
for (const f of files) {
  if (!existsSync(path.join(models, f))) continue;
  await copyFile(path.join(models, f), path.join(out, f));
  console.log(`+ ${f}`);
}

// the browser library (its "../src/" import becomes "./src/" in the flat repo layout)
{
  const code = await readFile(path.join(here, "indra-ocr-web.mjs"), "utf8");
  await writeFile(
    path.join(out, "indra-ocr-web.mjs"),
    code.replace('../src/pipeline.mjs', './src/pipeline.mjs')
  );
  console.log("+ indra-ocr-web.mjs");
}
for (const f of await readdir(path.join(jsDir, "src"))) {
  if (!f.endsWith(".mjs")) continue;
  await copyFile(path.join(jsDir, "src", f), path.join(out, "src", f));
  console.log(`+ src/${f}`);
}

await writeFile(
  path.join(out, "README.md"),
  `---
license: apache-2.0
---

# indra-ocr (browser)

The [indra-ocr](https://huggingface.co/datalab-to/indra-ocr) ONNX models
(tag v3.1.1) plus a browser loader that runs them on
[onnxruntime-web](https://onnxruntime.ai/docs/get-started/web) (wasm).  The
pre/post-processing is ported from the
[indra-ocr Python package](https://github.com/datalab-to/indra).

\`\`\`html
<script type="importmap">
{ "imports": {
    "onnxruntime-web": "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0-dev.20250409-89f8206ba4/dist/ort.wasm.bundle.min.mjs"
} }
</script>
<script type="module">
  import { loadIndra, readGrayBrowser } from
    "https://huggingface.co/${repoId}/resolve/main/indra-ocr-web.mjs";

  const ocr = await loadIndra("https://huggingface.co/${repoId}/resolve/main");
  const gray = await readGrayBrowser("page.png");        // -> {data, width, height}
  const page = await ocr.__call__(gray);                 // {lines: [{box, text, conf}], text, rotation}
  console.log(page.text);
</script>
\`\`\`

In a bundler/npm app, skip the importmap and install \`onnxruntime-web\`.

Options: \`loadIndra(base, { cjk: false, orient: false, order: false })\` skips
optional models.  Without cross-origin isolation (COOP/COEP) the wasm backend
runs single-threaded.
`
);
console.log("+ README.md");

console.log(`
staged in ${out}

push it with:
  huggingface-cli upload ${repoId} "${out}" . --repo-type model
`);
