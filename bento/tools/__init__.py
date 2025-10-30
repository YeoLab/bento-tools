from ._colocation import coloc_quotient, colocation
from ._composition import comp, comp_diff
from ._flux import flux, fluxmap
from ._flux_enrichment import fe, fe_fazal2019, fe_xia2019, gene_sets, load_gene_sets
from ._decomposition import decompose

# Explicit export surface
__all__ = [
    # Composition/colocation/flux
    "coloc_quotient",
    "colocation",
    "comp",
    "comp_diff",
    "flux",
    "fluxmap",
    "fe",
    "fe_fazal2019",
    "fe_xia2019",
    "gene_sets",
    "load_gene_sets",
    # Other
    "decompose",
]