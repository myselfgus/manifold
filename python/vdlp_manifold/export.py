"""Exportação dos artefatos do manifold para o viewer web (JSON compacto)."""
from __future__ import annotations

import json

import numpy as np


def _downsample(a: np.ndarray, n: int) -> np.ndarray:
    from scipy.ndimage import zoom

    return zoom(a, (n / a.shape[0], n / a.shape[1]), order=1)


def export_web_json(z, extent, peaks, geo, method, uncertainty,
                    out_path, grid: int = 128):
    """Escreve JSON consumido por web/build_viewer.py.

    O height-field é reamostrado para `grid` (checklist de performance 128-256²).
    Bacias e geodésica são mantidas em coords da grade-fonte + o tamanho da fonte,
    para o front-end remapear corretamente.
    """
    src = int(z.shape[0])
    z_ds = _downsample(z, grid)

    payload = {
        "grid": int(grid),
        "z": [round(float(v), 4) for v in z_ds.flatten()],
        "extent": [float(v) for v in extent],
        "peaks": [[int(round(p[0])), int(round(p[1]))] for p in peaks],
        "peaks_src_grid": src,
        "geodesic": [[int(g[0]), int(g[1])] for g in geo] if geo is not None else [],
        "geo_src_grid": src,
        "method": str(method),
        "uncertainty": str(uncertainty),
    }
    with open(out_path, "w") as f:
        f.write(json.dumps(payload))
    return len(json.dumps(payload))
