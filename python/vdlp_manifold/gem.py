"""Grafo GEM sobre a paisagem do manifold.

Mapeamento:
    eventos          -> pontos (o embedding em si)
    clusters/bacias  -> watershed dos picos de densidade
    fluxos           -> descida de gradiente (implícita no watershed)
    caminhos         -> geodésica aproximada (Dijkstra preferindo vales densos)
"""
from __future__ import annotations

import heapq

import numpy as np


def gem_graph(z: np.ndarray):
    """Retorna (basins, peaks, n_basins).

    basins : (H,W) rótulo de bacia por célula
    peaks  : (n,2) centros de bacia em coords de grade (row, col)
    """
    from scipy import ndimage

    energy = -z  # bacias = vales de energia = picos de densidade

    mx = ndimage.maximum_filter(z, size=9)
    peaks_mask = (z == mx) & (z > 0.25 * z.max())
    seeds, n = ndimage.label(peaks_mask)
    if n == 0:
        seeds, n = ndimage.label(z > 0.6 * z.max())

    try:
        from skimage.segmentation import watershed

        basins = watershed(energy, markers=seeds)
    except Exception:  # noqa: BLE001
        idx = ndimage.distance_transform_edt(
            seeds == 0, return_distances=False, return_indices=True
        )
        basins = seeds[tuple(idx)]

    peak_coords = ndimage.center_of_mass(z, labels=seeds, index=np.arange(1, n + 1))
    return basins, np.array(peak_coords), n


def geodesic_on_surface(z: np.ndarray, start_rc, end_rc) -> np.ndarray:
    """Caminho de menor custo na grade (Dijkstra 8-conexo).

    Custo = (1 - densidade): barato nos vales densos, caro nos cumes.
    Aproxima uma geodésica que prefere seguir a "calha" do manifold.
    """
    h, w = z.shape
    cost = (1.0 - z) + 0.02
    dist = np.full((h, w), np.inf)
    prev = -np.ones((h, w, 2), dtype=int)

    sr, sc = int(start_rc[0]), int(start_rc[1])
    er, ec = int(end_rc[0]), int(end_rc[1])
    dist[sr, sc] = 0.0
    pq = [(0.0, sr, sc)]
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while pq:
        d, r, c = heapq.heappop(pq)
        if (r, c) == (er, ec):
            break
        if d > dist[r, c]:
            continue
        for dr, dc in nbrs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                step = cost[nr, nc] * (1.41421 if dr and dc else 1.0)
                nd = d + step
                if nd < dist[nr, nc]:
                    dist[nr, nc] = nd
                    prev[nr, nc] = (r, c)
                    heapq.heappush(pq, (nd, nr, nc))

    path = []
    r, c = er, ec
    while (r, c) != (-1, -1) and not (r == sr and c == sc):
        path.append((r, c))
        r, c = prev[r, c]
        if r == -1:
            break
    path.append((sr, sc))
    return np.array(path[::-1])
