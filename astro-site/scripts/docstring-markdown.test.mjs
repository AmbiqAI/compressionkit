import test from 'node:test';
import assert from 'node:assert/strict';
import { docstringMarkdown } from './docstring-markdown.mjs';

test('fences reStructuredText examples so Python comments are not headings in MDX', () => {
  const input = 'Example::\n\n    codec = load_codec(path)\n\n    # Encode a frame\n    result = codec.compress(frame)\n\nNext paragraph.';
  assert.equal(docstringMarkdown(input), 'Example:\n\n```python\ncodec = load_codec(path)\n\n# Encode a frame\nresult = codec.compress(frame)\n```\n\nNext paragraph.');
});
test('preserves existing fences and prose without a literal block', () => {
  const input = 'Example::\n\nNo code here.\n\n```text\nExample::\n\n    keep this indentation\n```';
  assert.equal(docstringMarkdown(input), input);
});
