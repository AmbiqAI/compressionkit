"""Render saved notebook outputs without executing training code."""

import base64
import json
import re
import shutil
from pathlib import Path

def action_icon(paths: str) -> str:
    return (
        '<svg aria-hidden="true" focusable="false" viewBox="0 0 24 24" '
        'fill="none" stroke="currentColor" stroke-width="1.75" '
        'stroke-linecap="round" stroke-linejoin="round">'
        + paths + '</svg>'
    )


download_icon = action_icon('<path d="M12 3v12m-5-5 5 5 5-5M5 16v5h14v-5"/>')
source_icon = action_icon('<path d="m8 7-5 5 5 5m8-10 5 5-5 5m-3-14-2 18"/>')
launch_icon = action_icon('<path d="M14 3h7v7m0-7L10 14M10 3H3v18h18v-7"/>')

marker = "<!-- notebook-generated-page -->"
pages = Path("src/content/docs/guides")
pages.mkdir(parents=True, exist_ok=True)
for page in pages.glob("*.md"):
    if marker in page.read_text()[:500]:
        page.unlink()

assets = Path("public/notebooks")
shutil.rmtree(assets, ignore_errors=True)
assets.mkdir(parents=True)
for source in sorted(Path("../examples").glob("*.ipynb")):
    notebook = json.loads(source.read_text())
    shutil.copyfile(source, assets / source.name)
    title = next(
        (
            re.search(r"^# (.+)", "".join(c.get("source", [])), re.MULTILINE).group(1)
            for c in notebook["cells"]
            if c["cell_type"] == "markdown" and re.search(r"^# (.+)", "".join(c.get("source", [])), re.MULTILINE)
        ),
        source.stem.replace("-", " ").title(),
    )
    parts = [
        f"---\ntitle: {json.dumps(title)}\ndescription: Saved compressionKIT notebook example with code and outputs.\n---",
        marker,
        f'<div class="compressionkit-actions notebook-actions" role="group" aria-label="Notebook actions"><a href="/compressionkit/notebooks/{source.name}" download>{download_icon}<span>Download notebook</span></a><a href="https://github.com/AmbiqAI/compressionkit/blob/main/examples/{source.name}">{source_icon}<span>View source</span></a><a href="https://colab.research.google.com/github/AmbiqAI/compressionkit/blob/main/examples/{source.name}">{launch_icon}<span>Open in Colab</span></a></div>',
        "Open in Colab opens the source notebook only. Before running cells, use a Python 3.12 runtime and install compressionKIT into that runtime. See [notebook environment setup](/compressionkit/examples/#notebook-environment-setup). Hosted execution has not been validated by this documentation build.",
        "This example displays saved outputs. Building the documentation does not run training. Check dataset paths for your notebook working directory before running.",
    ]
    for ci, cell in enumerate(notebook["cells"]):
        text = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown":
            text = re.sub(
                r'<div\b[^>]*class="grid cards"[^>]*>.*?</div>',
                lambda match: "" if "View in Colab" in match[0] or "colab-badge.svg" in match[0] else match[0],
                text,
                flags=re.DOTALL,
            )
            text = re.sub(r"^# .+\n?", "", text, flags=re.MULTILINE)
            text = re.sub(r":(?:material|simple|fontawesome|octicons)-[\w-]+:", "", text)
            text = re.sub(r"\{\s*\.[^}]+\}", "", text)
            parts.append(text)
        elif cell["cell_type"] == "code":
            parts.append('<div class="notebook-input">\n\n```python title="Python"\n' + text.rstrip() + "\n```\n\n</div>")
            for oi, output in enumerate(cell.get("outputs", [])):
                data = output.get("data", {})
                if "image/png" in data:
                    name = f"{source.stem}-{ci}-{oi}.png"
                    (assets / name).write_bytes(base64.b64decode("".join(data["image/png"])))
                    parts.append(
                        f'<figure class="notebook-output-figure"><img src="/compressionkit/notebooks/{name}" alt="Saved figure from cell {ci + 1}" loading="lazy" /><figcaption>Saved output · cell {ci + 1}</figcaption></figure>'
                    )
                else:
                    raw = "".join(output.get("text", data.get("text/plain", [])))
                    raw = re.sub(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)", "", raw)
                    raw = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", raw)
                    if raw.strip():
                        if len(raw.splitlines()) > 12:
                            parts.append(
                                '<details class="notebook-output"><summary>Saved output</summary>\n\n'
                                + "```text\n" + raw.rstrip() + "\n```\n\n</details>"
                            )
                        else:
                            parts.append(
                                '<div class="notebook-output">\n\n```text title="Saved output"\n'
                                + raw.rstrip() + "\n```\n\n</div>"
                            )

    Path(f"src/content/docs/guides/{source.stem}.md").write_text("\n\n".join(parts) + "\n")
