"""VDLP Manifold — pipeline de visualização de manifolds para vetores VDLP (15-D).

Resultados são HIPÓTESES DE DESIGN, jamais diagnóstico clínico.
Todo drill-down deve reancorar na fala literal do paciente (ASL).
"""
from .dims import VDLP_DIMS, VDLP_DIMENSIONS, VDLP_LABELS, Dimension
from .synth import synth_vdlp_sessions
from .embed import embed_vdlp, EmbedResult
from .heightfield import kde_heightfield
from .gem import gem_graph, geodesic_on_surface

__version__ = "0.1.0"

__all__ = [
    "VDLP_DIMS",
    "VDLP_DIMENSIONS",
    "VDLP_LABELS",
    "Dimension",
    "synth_vdlp_sessions",
    "embed_vdlp",
    "EmbedResult",
    "kde_heightfield",
    "gem_graph",
    "geodesic_on_surface",
]
