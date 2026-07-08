"""Preview estático (matplotlib) do pipeline: embedding, paisagem GEM, surface 3D."""
from __future__ import annotations

import numpy as np


def _style(ax):
    ax.tick_params(colors="w")
    for s in ax.spines.values():
        s.set_color("#30363d")
    ax.xaxis.label.set_color("w")
    ax.yaxis.label.set_color("w")


def render_preview(emb, method, uncertainty, z, extent, basins, peaks, geo, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xmin, xmax, ymin, ymax = extent
    fig = plt.figure(figsize=(15, 6), facecolor="#0d1117")

    # (a) embedding cru colorido por tempo/sessão
    ax1 = fig.add_subplot(1, 3, 1, facecolor="#0d1117")
    sc = ax1.scatter(emb[:, 0], emb[:, 1], c=np.arange(len(emb)),
                     cmap="magma", s=14, edgecolor="none")
    ax1.set_title(f"Embedding ({method})\ncor = tempo/sessão", color="w", fontsize=10)
    _style(ax1)
    cb = fig.colorbar(sc, ax=ax1, fraction=0.046)
    plt.setp(plt.getp(cb.ax, "yticklabels"), color="w")
    cb.ax.yaxis.set_tick_params(color="w")

    # (b) paisagem KDE + bacias + geodésica
    ax2 = fig.add_subplot(1, 3, 2, facecolor="#0d1117")
    ax2.imshow(z, origin="lower", extent=extent, cmap="viridis", aspect="auto")
    ax2.contour(np.linspace(xmin, xmax, z.shape[1]),
                np.linspace(ymin, ymax, z.shape[0]), z, levels=8,
                colors="white", linewidths=0.4, alpha=0.5)
    if len(peaks):
        px = xmin + (peaks[:, 1] / z.shape[1]) * (xmax - xmin)
        py = ymin + (peaks[:, 0] / z.shape[0]) * (ymax - ymin)
        ax2.scatter(px, py, c="red", s=80, marker="*", edgecolor="w", label="bacias GEM")
    if geo is not None and len(geo):
        gx = xmin + (geo[:, 1] / z.shape[1]) * (xmax - xmin)
        gy = ymin + (geo[:, 0] / z.shape[0]) * (ymax - ymin)
        ax2.plot(gx, gy, "-", color="cyan", lw=2.2, label="geodésica")
    ax2.set_title("Paisagem KDE + Grafo GEM\n(bacias, geodésica)", color="w", fontsize=10)
    ax2.legend(loc="upper right", fontsize=7, framealpha=0.3)
    _style(ax2)

    # (c) surface 3D
    ax3 = fig.add_subplot(1, 3, 3, projection="3d", facecolor="#0d1117")
    step = 2
    gxm, gym = np.meshgrid(np.linspace(xmin, xmax, z.shape[1])[::step],
                           np.linspace(ymin, ymax, z.shape[0])[::step])
    ax3.plot_surface(gxm, gym, z[::step, ::step], cmap="viridis",
                     linewidth=0, antialiased=True, alpha=0.95)
    ax3.set_title("Manifold height-field 3D", color="w", fontsize=10)
    ax3.set_facecolor("#0d1117")
    for pane in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        pane.set_pane_color((0.05, 0.06, 0.09, 1.0))
        pane.label.set_color("w")
    ax3.tick_params(colors="w")

    fig.suptitle(f"VDLP Manifold — incerteza: {uncertainty.upper()}  "
                 f"(hipótese de design, NÃO diagnóstico)", color="w", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=130, facecolor="#0d1117")
    plt.close(fig)
