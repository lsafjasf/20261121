import { createReadStream, statSync } from "node:fs";
import { createServer } from "node:http";
import { extname, resolve, sep } from "node:path";

const root = resolve(process.cwd());
const port = Number(process.env.PORT || 4173);
const types = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".wav": "audio/wav",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
};

createServer((request, response) => {
  const pathname = decodeURIComponent(new URL(request.url, "http://localhost").pathname);
  let file = resolve(root, `.${pathname === "/" ? "/index.html" : pathname}`);
  if (!file.startsWith(`${root}${sep}`) && file !== root) {
    response.writeHead(403).end("Forbidden");
    return;
  }
  try {
    if (statSync(file).isDirectory()) file = resolve(file, "index.html");
    const size = statSync(file).size;
    response.writeHead(200, {
      "Content-Type": types[extname(file)] || "application/octet-stream",
      "Content-Length": size,
      "Cache-Control": "no-cache",
    });
    createReadStream(file).pipe(response);
  } catch {
    response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" }).end("Not found");
  }
}).listen(port, "127.0.0.1", () => {
  console.log(`Looproom running at http://127.0.0.1:${port}`);
});
