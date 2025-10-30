import numpy as np
from typing import Union
from shapely.geometry import Polygon, MultiPolygon
from spatialdata import SpatialData

from ._measure import measure


def _area(shape: Union[Polygon, MultiPolygon]) -> float:
    if not shape:
        return np.nan
    return shape.area


def area(
    sdata: SpatialData,
    shape_key: str = "cell_boundaries",
    result_key: str = "area",
    n_jobs: int = 1,
    recompute: bool = True,
    progress: bool = True,
    leave: bool = True,
) -> None:
    measure(sdata=sdata, func=_area, shape_key=shape_key, result_keys=result_key, n_jobs=n_jobs, recompute=recompute, progress=progress, leave=leave)


