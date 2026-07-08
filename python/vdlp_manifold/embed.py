"""Embedding VDLP com decisão automática de robustez.

Ordem de preferência (trade-off de fidelidade de trajetória):
    PHATE  >  UMAP(PCA-init)  >  PCA

Guard-rail: se N < MIN_SESSIONS_FOR_MANIFOLD sessões, NÃO confie em manifold
learning — cai para PCA simples e sinaliza incerteza ALTA.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_SESSIONS_FOR_MANIFOLD = 10


@dataclass
class EmbedResult:
    emb: np.ndarray       # (N, 2)
    method: str           # rótulo do método efetivamente usado
    uncertainty: str      # "baixa" | "media" | "alta"
    note: str             # explicação legível da decisão


def _pca2(X: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA

    return PCA(n_components=2).fit_transform(X)


def embed_vdlp(X: np.ndarray, seed: int = 42) -> EmbedResult:
    """Projeta X (N,15) em 2-D escolhendo o método mais confiável para o N."""
    n = X.shape[0]

    # Guard-rail de robustez para poucas sessões
    if n < MIN_SESSIONS_FOR_MANIFOLD:
        return EmbedResult(
            _pca2(X),
            "PCA(simples)",
            "alta",
            f"N={n}<{MIN_SESSIONS_FOR_MANIFOLD}: PHATE/UMAP não confiáveis; "
            "usando PCA e sinalizando incerteza ALTA.",
        )

    # Preferência 1: PHATE (diffusion map preserva trajetória contínua)
    try:
        import phate

        knn = int(np.clip(n // 12, 5, 30))
        op = phate.PHATE(
            n_components=2, knn=knn, decay=40, t="auto",
            random_state=seed, verbose=0, n_jobs=-1,
        )
        emb = op.fit_transform(X)
        return EmbedResult(
            emb, f"PHATE(knn={knn})", "baixa",
            "PHATE: diffusion map preserva a trajetória contínua entre estados.",
        )
    except Exception as e:  # noqa: BLE001
        # Preferência 2: UMAP SEMPRE com PCA-init (estabilidade/reprodutibilidade)
        try:
            import umap

            init = _pca2(X)
            reducer = umap.UMAP(
                n_components=2, init=init, n_neighbors=15,
                min_dist=0.1, random_state=seed,
            )
            emb = reducer.fit_transform(X)
            return EmbedResult(
                emb, "UMAP(PCA-init)", "media",
                f"PHATE indisponível ({e}); UMAP com PCA-init como fallback.",
            )
        except Exception as e2:  # noqa: BLE001
            return EmbedResult(
                _pca2(X), "PCA(fallback)", "alta",
                f"PHATE e UMAP indisponíveis ({e2}); PCA simples.",
            )
