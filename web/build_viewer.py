#!/usr/bin/env python3
"""Injeta os dados do manifold no template WebGPU/TSL e gera o viewer final.

Uso:
    python build_viewer.py --data ../data/manifold_data.json --out ../data/viewer_webgpu.html

O template (`template_webgpu.html`) contém o placeholder `__DATA__`, substituído
pelo JSON produzido por run_pipeline.py (vdlp_manifold.export.export_web_json).
"""
from __future__ import annotations

import argparse
import os


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(here, "..", "data", "manifold_data.json"))
    ap.add_argument("--template", default=os.path.join(here, "template_webgpu.html"))
    ap.add_argument("--out", default=os.path.join(here, "..", "data", "viewer_webgpu.html"))
    args = ap.parse_args()

    with open(args.data) as f:
        data = f.read()
    with open(args.template) as f:
        tpl = f.read()

    if "__DATA__" not in tpl:
        raise SystemExit("template sem placeholder __DATA__")

    html = tpl.replace("__DATA__", data)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"viewer gerado: {args.out} ({len(html)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
