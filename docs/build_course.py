"""Builds docs/course.html: the project explained end to end. Code blocks are cut from the real files
({{py:file:function}}, {{file:path:first-last}}, {{whole:path}}) so the page never drifts from the code."""
import ast
import html
import re
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
TEMPLATE = (Path(__file__).with_name("course_template.html")).read_text()


def py_def(file, name):
    src = (HERE / file).read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return ast.get_source_segment(src, node)
    raise KeyError(f"{file}:{name}")


def block(code, lang, path, label):
    return (f'<figure class="code"><figcaption><span class="path">{html.escape(path)}</span>'
            f'<span class="what">{html.escape(label)}</span></figcaption>'
            f'<pre><code class="language-{lang}">{html.escape(code.rstrip())}</code></pre></figure>')


def sub(m):
    kind, rest = m.group(1), m.group(2)
    if kind == "py":
        file, name = rest.split(":")
        return block(py_def(file, name), "python", file, f"{name}()" if not name[0].isupper() else f"class {name}")
    if kind == "lines":
        file, rng = rest.split(":")
        a, b = map(int, rng.split("-"))
        lines = (HERE / file).read_text().splitlines()[a - 1:b]
        lang = "bash" if file.endswith((".sh", ".sbatch", ".cmds")) else "python"
        return block("\n".join(lines), lang, file, f"lines {a}–{b}")
    if kind == "whole":
        file = rest
        lang = "bash" if file.endswith((".sh", ".sbatch", ".cmds")) else "python"
        return block((HERE / file).read_text(), lang, file, "whole file")
    raise ValueError(kind)


out = re.sub(r"\{\{(py|lines|whole):([^}]+)\}\}", sub, TEMPLATE)
assert "31dfa2" not in out                      # never ship the leaderboard token
(Path(__file__).with_name("course.html")).write_text(out)
print("wrote docs/course.html", len(out) // 1024, "KB")
