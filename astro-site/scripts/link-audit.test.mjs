import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
const script = new URL('./check-links.mjs', import.meta.url);
function audit(href) {
  const dir = mkdtempSync(join(tmpdir(), 'ck-link-audit-'));
  try {
    mkdirSync(join(dir, 'dist'));
    writeFileSync(join(dir, 'dist/index.html'), `<main><h1 id="intro">Intro</h1><a href="${href}">Link</a></main>`);
    const result = spawnSync(process.execPath, [script.pathname], { cwd: dir, encoding: 'utf8' });
    assert.ok(result.stdout, result.stderr);
    return { status: result.status, ...JSON.parse(result.stdout) };
  } finally { rmSync(dir, { recursive: true, force: true }); }
}
test('same-site absolute links validate their local fragments', () => {
  assert.equal(audit('https://ambiqai.github.io/compressionkit/#intro').status, 0);
  const missing = audit('https://ambiqai.github.io/compressionkit/#missing');
  assert.equal(missing.status, 1);
  assert.equal(missing.failures[0][2], 'anchor');
});
test('same-site absolute links validate local destination pages', () => {
  const result = audit('https://ambiqai.github.io/compressionkit/missing/');
  assert.equal(result.status, 1);
  assert.equal(result.failures[0][2], 'missing');
});
test('other product and external URLs stay outside the local audit', () => {
  assert.equal(audit('https://ambiqai.github.io/heartkit/').count, 0);
  assert.equal(audit('https://example.org/compressionkit/').count, 0);
});
