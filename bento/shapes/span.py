import numpy as np
from typing import Union
from scipy.spatial import distance_matrix
from shapely.geometry import Polygon, MultiPolygon
from spatialdata import SpatialData

from ._measure import measure


def _span(shape: Union[Polygon, MultiPolygon]) -> float:
    if not shape:
        return np.nan
    shape_coo = np.array(shape.coords.xy).T
    return int(distance_matrix(shape_coo, shape_coo).max())


def span(
    sdata: SpatialData,
    shape_key: str = "cell_boundaries",
    result_key: str = "span",
    n_jobs: int = 1,
    recompute: bool = True,
    progress: bool = True,
    leave: bool = True,
) -> None:
    measure(sdata=sdata, func=_span, shape_key=shape_key, result_keys=result_key, n_jobs=n_jobs, recompute=recompute, progress=progress, leave=leave)


