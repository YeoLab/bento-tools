import numpy as np
from typing import Union
from scipy.spatial import distance
from shapely.geometry import Polygon, MultiPolygon
from spatialdata import SpatialData

from ._measure import measure


def _second_moment(shape: Union[Polygon, MultiPolygon]) -> float:
    if not shape:
        return np.nan
    centroid = np.array(shape.centroid.coords).reshape(1, 2)
    pts = np.array(shape.exterior.xy).T
    radii = distance.cdist(centroid, pts)
    return np.sum(radii * radii / len(pts))


def second_moment(
    sdata: SpatialData,
    shape_key: str = "cell_boundaries",
    result_key: str = "second_moment",
    n_jobs: int = 1,
    recompute: bool = True,
    progress: bool = True,
    leave: bool = True,
) -> None:
    measure(sdata=sdata, func=_second_moment, shape_key=shape_key, result_keys=result_key, n_jobs=n_jobs, recompute=recompute, progress=progress, leave=leave)


