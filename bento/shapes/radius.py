import numpy as np
from typing import Union
from scipy.spatial import distance
from shapely.geometry import Polygon, MultiPolygon
from spatialdata import SpatialData

from ._measure import measure


def _radius(shape: Union[Polygon, MultiPolygon]) -> float:
    if not shape:
        return np.nan
    return distance.cdist(
        np.array(shape.centroid.coords).reshape(1, 2),
        np.array(shape.exterior.xy).T,
    ).mean()


def radius(
    sdata: SpatialData,
    shape_key: str = "cell_boundaries",
    result_key: str = "radius",
    n_jobs: int = 1,
    recompute: bool = True,
    progress: bool = True,
    leave: bool = True,
) -> None:
    measure(sdata=sdata, func=_radius, shape_key=shape_key, result_keys=result_key, n_jobs=n_jobs, recompute=recompute, progress=progress, leave=leave)


