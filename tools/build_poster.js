// Build the A0 poster PDF (and a preview PNG) from poster/poster.html.
//   node tools/build_poster.js            (needs Playwright with Chromium)
// Serves the repository root on a local port so the poster can load ../docs/static assets.
const http = require("http");
const fs = require("fs");
const path = require("path");
let chromium;
try { ({ chromium } = require("playwright")); }
catch { ({ chromium } = require(require("child_process").execSync("npm root -g").toString().trim() + "/playwright")); }

const ROOT = path.resolve(__dirname, "..");
const TYPES = { ".html": "text/html", ".css": "text/css", ".js": "text/javascript", ".json": "application/json", ".png": "image/png",
  ".jpg": "image/jpeg", ".svg": "image/svg+xml", ".woff2": "font/woff2" };
const server = http.createServer((req, res) => {
  const file = path.join(ROOT, decodeURIComponent(req.url.split("?")[0]));
  if (!file.startsWith(ROOT) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { "Content-Type": TYPES[path.extname(file)] || "application/octet-stream" });
  fs.createReadStream(file).pipe(res);
});

(async () => {
  await new Promise((r) => server.listen(0, r));
  const port = server.address().port;
  const MM = 96 / 25.4;
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: Math.round(1189 * MM), height: Math.round(841 * MM) } });
  page.on("pageerror", (e) => console.error("page error:", e.message));
  await page.goto(`http://localhost:${port}/poster/poster.html`, { waitUntil: "networkidle" });
  await page.waitForSelector("body[data-ready='1']", { timeout: 30000 });
  await page.evaluate(() => document.fonts.ready);
  const overflow = await page.evaluate(() => [...document.querySelectorAll(".col")].map((c) => Math.round(c.scrollHeight - c.clientHeight)));
  console.log("column overflow (px, should be <= 0):", overflow);
  const out = path.join(ROOT, "poster", "PEACH_poster_NeurIPS2026_A0.pdf");
  await page.pdf({ path: out, width: "1189mm", height: "841mm", printBackground: true, preferCSSPageSize: true });
  await page.setViewportSize({ width: Math.round(1189 * MM), height: Math.round(841 * MM) });
  const prev = await browser.newPage({ viewport: { width: Math.round(1189 * MM), height: Math.round(841 * MM) }, deviceScaleFactor: 0.5 });
  await prev.goto(`http://localhost:${port}/poster/poster.html`, { waitUntil: "networkidle" });
  await prev.waitForSelector("body[data-ready='1']", { timeout: 30000 });
  await prev.screenshot({ path: path.join(ROOT, "poster", "preview.png") });
  console.log("wrote", out);
  await browser.close();
  server.close();
})();
