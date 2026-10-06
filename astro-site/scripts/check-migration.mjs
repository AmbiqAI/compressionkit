import { createHash } from 'node:crypto';
import { readFileSync, existsSync, globSync } from 'node:fs';
import { basename } from 'node:path';
const manifest = JSON.parse(readFileSync('migration-manifest.json', 'utf8'));
const failures = [];
const hash = path => createHash('sha256').update(readFileSync(path)).digest('hex');
const sources = new Set(manifest.items.map(item => item.source));
for (const path of globSync('../examples/*.ipynb')) {
  if (!sources.has(path.slice(3))) failures.push(`Uninventoried source: ${path}`);
}
for (const item of manifest.items) {
  const source = '../' + item.source;
  if (item.retiredSource) {
    if (existsSync(source)) failures.push(`Retired source restored: ${item.source}`);
    if (item.retainedAt && (!existsSync('../' + item.retainedAt) || hash('../' + item.retainedAt) !== item.sha256)) failures.push(`Retained note differs: ${item.retainedAt}`);
  } else if (!existsSync(source) || hash(source) !== item.sha256) failures.push(`Source changed; review migration disposition: ${item.source}`);
  if (item.disposition === 'excluded') continue;
  if (item.source.endsWith('.ipynb')) {
    const name = basename(item.source);
    if (!existsSync(`public/notebooks/${name}`) || hash(source) !== hash(`public/notebooks/${name}`)) failures.push(`Notebook download differs: ${name}`);
    if (!existsSync(`dist/guides/${name.replace('.ipynb', '')}/index.html`)) failures.push(`Notebook route missing: ${name}`);
    continue;
  }
  if (!existsSync('../' + item.destination)) failures.push(`Destination missing: ${item.destination}`);
  if (item.source.endsWith('.md')) {
    const route = item.source.slice(5).replace(/(?:\/)?index\.md$/, '').replace(/\.md$/, '');
    if (!existsSync(`dist/${route}/index.html`)) failures.push(`Original route missing: ${route}`);
  }
}
for (const [path, sha] of Object.entries(manifest.watchedSources ?? {})) {
  if (!existsSync('../' + path) || hash('../' + path) !== sha) failures.push(`Generator input changed; review affected Astro pages: ${path}`);
}
console.log(JSON.stringify({ inventoried: manifest.items.length, failures }, null, 2));
if (failures.length) process.exitCode = 1;

const { parseHTML } = await import('linkedom');
const dispositions = JSON.parse(readFileSync('legacy-anchor-dispositions.json', 'utf8'));
const legacy = JSON.parse(readFileSync('anchor-review.json', 'utf8')).results.flatMap(page => page.missing.map(anchor => ({page: page.route, id: anchor.id})));
const anchorFailures = [];
for (const anchor of legacy) {
  const matches = dispositions.filter(item => item.page === anchor.page && item.id === anchor.id);
  if (matches.length !== 1) { anchorFailures.push(`Missing/duplicate disposition: ${anchor.page}#${anchor.id}`); continue; }
  const item = matches[0];
  if (item.disposition === 'retired' && item.reason) continue;
  const { document } = parseHTML(readFileSync(`dist/${item.page}/index.html`, 'utf8'));
  if (!document.getElementById(item.id)) anchorFailures.push(`Missing legacy alias: ${item.page}#${item.id}`);
  if (item.target) {
    const url = new URL(item.target, 'https://local');
    const path = 'dist/' + url.pathname.replace(/^\/compressionkit\//, '') + 'index.html';
    if (!existsSync(path)) {anchorFailures.push(`Missing legacy target: ${item.target}`);continue;}
    const { document: target } = parseHTML(readFileSync(path, 'utf8'));
    if (!target.getElementById(decodeURIComponent(url.hash.slice(1)))) anchorFailures.push(`Missing legacy target fragment: ${item.target}`);
  }
}
console.log(JSON.stringify({legacyAnchors: legacy.length, anchorFailures}, null, 2));
if (anchorFailures.length) process.exitCode = 1;
