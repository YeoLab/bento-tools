import numpy as np
from typing import Union
from shapely.geometry import Polygon, MultiPolygon, Point
from spatialdata import SpatialData

from ._measure import measure


def _aspect_ratio(shape: Union[Polygon, MultiPolygon]) -> float:
    if not shape:
        return np.nan
    x, y = shape.minimum_rotated_rectangle.exterior.coords.xy
    edge_length = (Point(x[0], y[0]).distance(Point(x[1], y[1])), Point(x[1], y[1]).distance(Point(x[2], y[2])))
    length, width = max(edge_length), min(edge_length)
    return length / width


def aspect_ratio(
    sdata: SpatialData,
    shape_key: str = "cell_boundaries",
    result_key: str = "aspect_ratio",
    n_jobs: int = 1,
    recompute: bool = True,
    progress: bool = True,
    leave: bool = True,
) -> None:
    measure(sdata=sdata, func=_aspect_ratio, shape_key=shape_key, result_keys=result_key, n_jobs=n_jobs, recompute=recompute, progress=progress, leave=leave)


