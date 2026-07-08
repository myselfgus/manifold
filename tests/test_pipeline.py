"""Testes de fumaça do pipeline VDLP (rápidos, sem depender de PHATE/UMAP)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from vdlp_manifold import (  # noqa: E402
    VDLP_DIMS,
    embed_vdlp,
    gem_graph,
    geodesic_on_surface,
    kde_heightfield,
    synth_vdlp_sessions,
)
from vdlp_manifold.embed import MIN_SESSIONS_FOR_MANIFOLD  # noqa: E402


def test_dims_len():
    assert len(VDLP_DIMS) == 15


def test_synth_shapes():
    X, t, state = synth_vdlp_sessions(n_sessions=50)
    assert X.shape == (50, 15)
    assert t.shape == (50,)
    assert state.shape == (50,)
    assert X.dtype == np.float32


def test_low_n_falls_back_to_pca_with_high_uncertainty():
    X, _, _ = synth_vdlp_sessions(n_sessions=MIN_SESSIONS_FOR_MANIFOLD - 3)
    r = embed_vdlp(X)
    assert r.emb.shape[1] == 2
    assert r.uncertainty == "alta"
    assert "PCA" in r.method


def test_healthy_n_uses_manifold_or_pca_fallback():
    X, _, _ = synth_vdlp_sessions(n_sessions=120)
    r = embed_vdlp(X)
    assert r.emb.shape == (120, 2)
    # com N saudável não deve sinalizar incerteza alta por causa de N
    assert r.uncertainty in ("baixa", "media")


def test_heightfield_normalized():
    X, _, _ = synth_vdlp_sessions(n_sessions=80)
    r = embed_vdlp(X)
    z, axes, extent = kde_heightfield(r.emb, grid=96)
    assert z.shape == (96, 96)
    assert abs(z.max() - 1.0) < 1e-6
    assert z.min() >= 0.0
    assert len(extent) == 4


def test_gem_and_geodesic():
    X, _, _ = synth_vdlp_sessions(n_sessions=100)
    r = embed_vdlp(X)
    z, _, _ = kde_heightfield(r.emb, grid=96)
    basins, peaks, n = gem_graph(z)
    assert n >= 1
    assert basins.shape == z.shape
    if len(peaks) >= 2:
        p0 = tuple(np.round(peaks[0]).astype(int))
        p1 = tuple(np.round(peaks[-1]).astype(int))
        path = geodesic_on_surface(z, p0, p1)
        assert path.ndim == 2 and path.shape[1] == 2
        assert tuple(path[0]) == p0
        assert tuple(path[-1]) == p1
