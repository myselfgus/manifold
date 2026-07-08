"""Geração de sessões VDLP sintéticas com estrutura de TRAJETÓRIA.

O paciente atravessa `n_states` estados latentes ao longo do tempo com
transições suaves (smoothstep) + deriva longitudinal (random walk) + ruído.
Isso produz uma curva contínua em 15-D — exatamente o caso onde
PHATE/diffusion maps preservam a estrutura e t-SNE a fragmentaria.
"""
from __future__ import annotations

import numpy as np


def synth_vdlp_sessions(
    n_sessions: int = 180,
    n_states: int = 3,
    noise: float = 0.35,
    drift: float = 0.02,
    seed: int = 42,
):
    """Retorna (X, t, state).

    X     : (n_sessions, 15) float32 — vetores VDLP por sessão
    t     : (n_sessions,)    float   — tempo normalizado 0..1
    state : (n_sessions,)    int     — estado latente dominante (validação/cor)
    """
    rng = np.random.default_rng(seed)

    # centros latentes (bacias) em 15-D
    centers = rng.normal(0, 1.0, size=(n_states, 15)) * 1.6

    t = np.linspace(0, 1, n_sessions)

    # serpenteio entre estados: interpolação com easing smoothstep
    phase = t * (n_states - 1)
    lo = np.floor(phase).astype(int)
    hi = np.clip(lo + 1, 0, n_states - 1)
    frac = phase - lo
    frac = frac * frac * (3 - 2 * frac)  # smoothstep

    base = (1 - frac)[:, None] * centers[lo] + frac[:, None] * centers[hi]

    # não-estacionariedade longitudinal + ruído idiossincrático por sessão
    drift_walk = np.cumsum(rng.normal(0, drift, size=(n_sessions, 15)), axis=0)
    X = base + drift_walk + rng.normal(0, noise, size=(n_sessions, 15))

    state = np.where(frac < 0.5, lo, hi)
    return X.astype(np.float32), t, state
