"""Height-field via KDE gaussiano separável sobre o embedding 2-D.

A "paisagem" é a densidade de sessões no plano do embedding. Aqui usamos
convolução gaussiana separável em NumPy (barata, O(N*k)); on-device isto deve
rodar em MPS/vDSP. Bandwidth por regra de Scott adaptada ao passo da grade.
"""
from __future__ import annotations

import numpy as np


def _gauss_kernel1d(sigma: float) -> np.ndarray:
    r = int(max(1, round(3 * sigma)))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return (k / k.sum()).astype(np.float64)


def _sep_conv2d(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Convolução separável com padding refletido (equivalente ao vDSP)."""
    pad = len(k) // 2
    ap = np.pad(a, ((0, 0), (pad, pad)), mode="reflect")
    b = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 1, ap)
    bp = np.pad(b, ((pad, pad), (0, 0)), mode="reflect")
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 0, bp)


def kde_heightfield(emb: np.ndarray, grid: int = 192, bw: float | None = None,
                    pad: float = 0.12):
    """Retorna (Z, (gx, gy), extent).

    Z      : (grid, grid) densidade normalizada em [0,1], row=y, col=x
    gx, gy : coordenadas dos eixos da grade
    extent : (xmin, xmax, ymin, ymax)
    """
    x, y = emb[:, 0], emb[:, 1]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    dx, dy = (xmax - xmin), (ymax - ymin)
    xmin, xmax = xmin - pad * dx, xmax + pad * dx
    ymin, ymax = ymin - pad * dy, ymax + pad * dy

    gx = np.linspace(xmin, xmax, grid)
    gy = np.linspace(ymin, ymax, grid)

    hist, _, _ = np.histogram2d(
        x, y, bins=[grid, grid], range=[[xmin, xmax], [ymin, ymax]]
    )
    hist = hist.T  # (row=y, col=x)

    if bw is None:
        step = ((xmax - xmin) / grid + (ymax - ymin) / grid) / 2
        sigma_data = 0.06 * max(dx, dy)
        bw = max(sigma_data / step, 1.2)

    k = _gauss_kernel1d(bw)
    z = _sep_conv2d(hist, k)
    z /= z.max() + 1e-9
    return z, (gx, gy), (xmin, xmax, ymin, ymax)
