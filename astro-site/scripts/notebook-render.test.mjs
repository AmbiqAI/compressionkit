import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
const script = new URL('./build-notebooks.py', import.meta.url);
function render(metadata) {
  const dir = mkdtempSync(join(tmpdir(), 'ck-notebook-'));
  try {
    mkdirSync(join(dir, 'examples'));
    mkdirSync(join(dir, 'site'));
    const source = JSON.stringify({ cells: [{cell_type: 'code', source: ['plot()'], outputs: [{data: {'image/png': ['AA==']}, metadata}]}] });
    writeFileSync(join(dir, 'examples/example.ipynb'), source);
    const result = spawnSync('python3', [script.pathname], {cwd: join(dir, 'site'), encoding: 'utf8'});
    return { ...result, page: result.status === 0 ? readFileSync(join(dir, 'site/src/content/docs/guides/example.md'), 'utf8') : '', download: result.status === 0 ? readFileSync(join(dir, 'site/public/notebooks/example.ipynb'), 'utf8') : '', source };
  } finally { rmSync(dir, {recursive: true, force: true}); }
}
test('notebook plot descriptions are escaped and downloads preserve source bytes', () => {
  const result = render({alt: 'Original "ECG" & reconstructed <signal> over time'});
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.page, /alt="Original &quot;ECG&quot; &amp; reconstructed &lt;signal&gt; over time"/);
  assert.equal(result.download, result.source);
});
test('notebook plots without descriptions fail the build', () => {
  const result = render({});
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Missing plot alternative text/);
});
