#!/usr/bin/env python3
"""CLI do pipeline VDLP Manifold.

Uso:
    python run_pipeline.py --sessions 180 --grid 192 --out ../data

Etapas: síntese (ou --input .npy) -> embedding -> KDE -> grafo GEM ->
export (.npz + preview.png + manifold_data.json p/ o viewer web).

Resultados são HIPÓTESES DE DESIGN, jamais diagnóstico clínico.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# permite rodar sem instalar o pacote (python run_pipeline.py)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vdlp_manifold import (  # noqa: E402
    VDLP_DIMS,
    embed_vdlp,
    gem_graph,
    geodesic_on_surface,
    kde_heightfield,
    synth_vdlp_sessions,
)
from vdlp_manifold.export import export_web_json  # noqa: E402
from vdlp_manifold.preview import render_preview  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="VDLP Manifold pipeline")
    ap.add_argument("--input", help="arquivo .npy com X (N,15). Se ausente, gera sintético")
    ap.add_argument("--sessions", type=int, default=180, help="N de sessões sintéticas")
    ap.add_argument("--states", type=int, default=3, help="N de estados latentes (sintético)")
    ap.add_argument("--grid", type=int, default=192, help="resolução da grade KDE")
    ap.add_argument("--web-grid", type=int, default=128, help="grade reamostrada p/ web")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data", help="diretório de saída")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(">> 1. Dados VDLP (15-D)...")
    if args.input:
        X = np.load(args.input).astype(np.float32)
        t = np.linspace(0, 1, len(X))
        state = np.zeros(len(X), dtype=int)
        print(f"   carregado {args.input}: X={X.shape}")
    else:
        X, t, state = synth_vdlp_sessions(args.sessions, args.states, seed=args.seed)
        print(f"   sintético: X={X.shape} | estados={state.max()+1}")
    assert X.shape[1] == 15, f"VDLP precisa de 15 dims, recebi {X.shape[1]}"

    print(">> 2. Embedding (decisão automática de robustez)...")
    er = embed_vdlp(X, seed=args.seed)
    print(f"   método={er.method} | incerteza={er.uncertainty}")
    print(f"   nota  ={er.note}")

    print(f">> 3. Height-field KDE (grade {args.grid}²)...")
    z, (gx, gy), extent = kde_heightfield(er.emb, grid=args.grid)
    print(f"   Z={z.shape} range=[{z.min():.3f},{z.max():.3f}]")

    print(">> 4. Grafo GEM (bacias + geodésica)...")
    basins, peaks, nbas = gem_graph(z)
    print(f"   bacias={nbas}")
    geo = None
    if len(peaks) >= 2:
        p0 = tuple(np.round(peaks[0]).astype(int))
        p1 = tuple(np.round(peaks[-1]).astype(int))
        geo = geodesic_on_surface(z, p0, p1)
        print(f"   geodésica: {len(geo)} passos (bacia0 -> bacia{len(peaks)-1})")

    print(">> 5. Export...")
    npz = os.path.join(args.out, "vdlp_manifold.npz")
    np.savez_compressed(
        npz, X=X, emb=er.emb, t=t, state=state, Z=z, basins=basins, peaks=peaks,
        geodesic=(geo if geo is not None else np.empty((0, 2))),
        extent=np.array(extent), dims=np.array(VDLP_DIMS),
        method=er.method, uncertainty=er.uncertainty,
    )
    png = os.path.join(args.out, "vdlp_manifold_preview.png")
    render_preview(er.emb, er.method, er.uncertainty, z, extent, basins, peaks, geo, png)
    js = os.path.join(args.out, "manifold_data.json")
    nbytes = export_web_json(z, extent, peaks, geo, er.method, er.uncertainty,
                             js, grid=args.web_grid)

    print(f"   {npz}")
    print(f"   {png}")
    print(f"   {js} ({nbytes} bytes)")
    print(">> concluído. Lembrete: hipótese de design, NÃO diagnóstico clínico.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
