export function docstringMarkdown(value) {
  const lines = value.replace(/:(?:material|simple|fontawesome|octicons)-[\w-]+:/g, '').split('\n');
  const output = [];
  let fenced = false;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (/^\s*```/.test(line)) fenced = !fenced;
    if (!fenced && line.endsWith('::') && lines[i + 1]?.trim() === '') {
      let start = i + 1;
      while (start < lines.length && !lines[start].trim()) start++;
      if (/^ {4}\S/.test(lines[start] ?? '')) {
        let end = start;
        while (end < lines.length && (!lines[end].trim() || /^ {4}/.test(lines[end]))) end++;
        const code = lines.slice(start, end).map(line => line.slice(4)).join('\n').trimEnd();
        output.push(line.slice(0, -1), '', '```python', code, '```', '');
        i = end - 1;
        continue;
      }
    }
    output.push(line);
  }
  return output.join('\n');
}
