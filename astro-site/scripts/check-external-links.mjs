import { globSync, readFileSync, writeFileSync } from 'node:fs';
import { parseHTML } from 'linkedom';
const urls = new Set();
for (const file of globSync('dist/**/*.html')) {
  const { document } = parseHTML(readFileSync(file, 'utf8'));
  for (const anchor of document.querySelectorAll('main a[href]')) {
    const href = anchor.getAttribute('href');
    if (!/^https?:/.test(href)) continue;
    const url = new URL(href);
    if (url.hostname === 'ambiqai.github.io' && url.pathname.startsWith('/compressionkit/')) continue;
    url.hash = '';
    urls.add(url.href);
  }
}
const queue = [...urls].sort();
const results = [];
async function worker() {
  while (queue.length) {
    const url = queue.shift();
    try {
      const response = await fetch(url, { signal: AbortSignal.timeout(15000), redirect: 'follow' });
      await response.body?.cancel();
      results.push({ url, status: response.status, final: response.url });
    } catch (error) {
      results.push({ url, error: error.message });
    }
  }
}
await Promise.all(Array.from({ length: 6 }, worker));
results.sort((a, b) => a.url.localeCompare(b.url));
const report = { checkedAt: new Date().toISOString(), count: results.length, results };
writeFileSync('external-link-audit.json', JSON.stringify(report, null, 2) + '\n');
const unresolved = results.filter(r => !r.status || r.status >= 400);
console.log(JSON.stringify({ count: results.length, unresolved }, null, 2));
if (unresolved.length) process.exitCode = 1;
