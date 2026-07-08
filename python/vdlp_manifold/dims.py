"""As 15 dimensões do Espaço Mental ℳ — o vetor VDLP.

Cada dimensão é um marcador linguístico/prosódico **extraível de fala natural**
e **ancorado em frameworks clínicos e computacionais validados** (RDoC, HiTOP,
Big5, PANAS, PERMA, WHODAS, modelo Circumplex de Russell). A fundamentação
científica completa — frameworks que validam o construto, métodos de extração de
linguagem natural, métricas matemáticas, literatura e precisão computacional por
dimensão — está em:

    docs/VALIDACAO_DIMENSOES_ESPACO_MENTAL.md

Construto: Gustavo Mendes e Silva, M.D.

As dimensões organizam-se em três meta-dimensões (afetiva, cognitiva, agência).
`VDLP_DIMS` mantém os identificadores na ordem do vetor; `VDLP_DIMENSIONS` traz
os metadados legíveis de cada uma.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Dimension:
    id: str                      # identificador canônico v1…v15
    slug: str                    # nome curto (usado como rótulo)
    label: str                   # nome legível
    meta: str                    # meta-dimensão: afetiva | cognitiva | agencia
    frameworks: tuple[str, ...]  # frameworks que validam o construto
    status: str = "validado"     # "validado" | "parcial"


VDLP_DIMENSIONS: list[Dimension] = [
    # ---- META-DIMENSÃO AFETIVA ----
    Dimension("v1", "valencia_emocional", "Valência Emocional", "afetiva",
              ("RDoC", "Circumplex", "PANAS", "Big5")),
    Dimension("v2", "arousal_ativacao", "Arousal / Ativação", "afetiva",
              ("RDoC", "Circumplex", "PANAS", "HiTOP")),
    Dimension("v3", "coerencia_narrativa", "Coerência Narrativa", "afetiva",
              ("RDoC", "HiTOP", "DSM-5")),
    Dimension("v4", "complexidade_sintatica", "Complexidade Sintática", "afetiva",
              ("RDoC", "Big5", "WHODAS")),
    # ---- META-DIMENSÃO COGNITIVA ----
    Dimension("v5", "orientacao_temporal", "Orientação Temporal", "cognitiva",
              ("HiTOP", "CBT", "PERMA")),
    Dimension("v6", "densidade_autoreferencia", "Densidade de Autoreferência", "cognitiva",
              ("RDoC", "Big5", "HiTOP", "DSM-5")),
    Dimension("v7", "linguagem_social", "Linguagem Social", "cognitiva",
              ("RDoC", "PERMA", "WHODAS", "Big5")),
    Dimension("v8", "flexibilidade_discursiva", "Flexibilidade Discursiva", "cognitiva",
              ("RDoC", "Big5", "HiTOP")),
    # ---- META-DIMENSÃO DE AGÊNCIA ----
    Dimension("v9", "dominancia_agencia", "Dominância / Agência", "agencia",
              ("RDoC", "Circumplex", "PERMA", "WHODAS", "Locus of Control")),
    Dimension("v10", "fragmentacao_discurso", "Fragmentação do Discurso", "agencia",
              ("HiTOP", "DSM-5")),
    Dimension("v11", "densidade_semantica", "Densidade Semântica", "agencia",
              ("RDoC", "Big5", "Dementia Assessment")),
    Dimension("v12", "certeza_incerteza", "Marcadores de Certeza/Incerteza", "agencia",
              ("Big5", "HiTOP", "Metacognition")),
    Dimension("v13", "conectividade", "Padrões de Conectividade", "agencia",
              ("RDoC", "Big5", "WHODAS")),
    Dimension("v14", "pragmatica", "Comunicação Pragmática", "agencia",
              ("RDoC", "DSM-5", "Big5"), status="parcial"),
    Dimension("v15", "prosodia_emocional", "Prosódia Emocional", "agencia",
              ("RDoC", "Circumplex", "Affective Computing")),
]

# Identificadores (slugs) na ordem do vetor VDLP (v1…v15).
VDLP_DIMS = [d.slug for d in VDLP_DIMENSIONS]

# Rótulos legíveis, na mesma ordem.
VDLP_LABELS = [d.label for d in VDLP_DIMENSIONS]

assert len(VDLP_DIMS) == 15, "O Espaço Mental ℳ / VDLP tem exatamente 15 dimensões"
assert len(VDLP_DIMENSIONS) == 15
