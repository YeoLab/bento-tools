import numpy as np
from typing import Union, Tuple, List
from shapely.geometry import Polygon, MultiPolygon
from spatialdata import SpatialData

from ._measure import measure


def _bounds(shape: Union[Polygon, MultiPolygon]) -> Tuple[float, float, float, float]:
    if not shape:
        return (np.nan, np.nan, np.nan, np.nan)
    return shape.bounds


def bounds(
    sdata: SpatialData,
    shape_key: str = "cell_boundaries",
    result_keys: List[str] = ["xmin", "ymin", "xmax", "ymax"],
    n_jobs: int = 1,
    recompute: bool = True,
    progress: bool = True,
    leave: bool = True,
) -> None:
    measure(sdata=sdata, func=_bounds, shape_key=shape_key, result_keys=result_keys, n_jobs=n_jobs, recompute=recompute, progress=progress, leave=leave)


