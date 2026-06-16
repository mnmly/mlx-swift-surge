#!/usr/bin/env python3
"""Emit an llms.txt-style Markdown index from a transformed DocC static site.

The experimental `--enable-experimental-markdown-output` docc flags aren't
available on this machine (SwiftPM-CLI can't locate the Metal toolchain that
mlx-swift needs, and the snapshot `docc` is x86-only), so we derive the export
from the render index + per-symbol JSON that `transform-for-static-hosting`
already produced. Pure stdlib; no dependencies.

Usage: generate_llms_txt.py <docs-dir> <Target> [<Target> ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def inline_text(nodes) -> str:
    """Flatten DocC inline-content nodes to plain text."""
    out = []
    if isinstance(nodes, list):
        for n in nodes:
            if isinstance(n, dict):
                if "text" in n:
                    out.append(n["text"])
                elif "code" in n:
                    out.append(n["code"])
                elif "inlineContent" in n:
                    out.append(inline_text(n["inlineContent"]))
    return "".join(out)


def abstract_for(docs_dir: Path, target: str, path: str) -> str:
    """Read a symbol's abstract from its render JSON, if present."""
    if not path:
        return ""
    rel = path.removeprefix("/")  # documentation/mlxsurge/...
    jf = docs_dir / target / "data" / (rel + ".json")
    if not jf.exists():
        return ""
    try:
        d = json.loads(jf.read_text())
    except Exception:
        return ""
    return inline_text(d.get("abstract", [])).strip()


def walk(node, depth, lines, docs_dir, target):
    if not isinstance(node, dict):
        return
    title = node.get("title", "")
    ntype = node.get("type", "")
    path = node.get("path", "")
    if ntype == "groupMarker":
        lines.append("")
        lines.append(f"{'#' * min(depth + 2, 6)} {title}")
    elif title:
        bullet = "  " * max(0, depth - 1) + f"- `{title}`"
        abstract = abstract_for(docs_dir, target, path)
        lines.append(bullet + (f" — {abstract}" if abstract else ""))
    for child in node.get("children", []) or []:
        walk(child, depth + 1, lines, docs_dir, target)


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: generate_llms_txt.py <docs-dir> <Target> [...]", file=sys.stderr)
        sys.exit(2)
    docs_dir = Path(sys.argv[1])
    targets = sys.argv[2:]

    lines = [
        "# mlx-swift-surge — API reference (llms.txt)",
        "",
        "Auto-generated from the DocC render index. Each entry is a public "
        "symbol with its one-line abstract.",
        "",
    ]
    for target in targets:
        idx = docs_dir / target / "index" / "index.json"
        if not idx.exists():
            print(f"warning: no index for {target} at {idx}", file=sys.stderr)
            continue
        data = json.loads(idx.read_text())
        for root in data.get("interfaceLanguages", {}).get("swift", []):
            walk(root, 0, lines, docs_dir, target)

    out = docs_dir / "llms.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out} ({len(lines)} lines).")


if __name__ == "__main__":
    main()
