// Tiny static server for the demo: serves this package's root, so
//   http://127.0.0.1:8080/web/demo.html   (models at /models)
// Usage: node web/serve.mjs [port]
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
const port = Number(process.argv[2] ?? 8080);
const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".wasm": "application/wasm",
  ".onnx": "application/octet-stream",
};

createServer(async (req, res) => {
  try {
    const url = new URL(req.url, "http://x");
    let p = path.normalize(path.join(root, decodeURIComponent(url.pathname)));
    if (!p.startsWith(root)) throw new Error("outside root");
    if (p.endsWith("/")) p += "index.html";
    const data = await readFile(p);
    res.writeHead(200, {
      "content-type": MIME[path.extname(p).toLowerCase()] ?? "application/octet-stream",
      "cross-origin-opener-policy": "same-origin",
      "cross-origin-embedder-policy": "require-corp",
    });
    res.end(data);
  } catch {
    res.writeHead(404).end("not found");
  }
}).listen(port, "127.0.0.1", () => {
  console.log(`serving ${root} at http://127.0.0.1:${port}/web/demo.html`);
});
