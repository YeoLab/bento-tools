import numpy as np
from typing import Union, Tuple, List
from shapely.geometry import Polygon, MultiPolygon
from spatialdata import SpatialData

from ._measure import measure


def _centroid(shape: Union[Polygon, MultiPolygon]) -> Tuple[float, float]:
    if not shape:
        return (np.nan, np.nan)
    centroid_x = np.mean(shape.exterior.xy[0])
    centroid_y = np.mean(shape.exterior.xy[1])
    return (centroid_x, centroid_y)


def centroid(
    sdata: SpatialData,
    shape_key: str = "cell_boundaries",
    result_keys: List[str] = ["x", "y"],
    n_jobs: int = 1,
    recompute: bool = True,
    progress: bool = True,
    leave: bool = True,
) -> None:
    measure(sdata=sdata, func=_centroid, shape_key=shape_key, result_keys=result_keys, n_jobs=n_jobs, recompute=recompute, progress=progress, leave=leave)


