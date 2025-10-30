from typing import Union, Optional
from shapely.geometry import Polygon, MultiPolygon
from spatialdata import SpatialData

from ._measure import measure
from .radius import _radius


def _opening(shape: Union[Polygon, MultiPolygon], proportion: float) -> Optional[Union[Polygon, MultiPolygon]]:
    if not shape:
        return None
    d = proportion * _radius(shape)
    return shape.buffer(-d).buffer(d)


def opening(
    sdata: SpatialData,
    shape_key: str = "cell_boundaries",
    proportion: float = 0.1,
    result_key: str = "opened_shape",
    n_jobs: int = 1,
    recompute: bool = True,
    progress: bool = True,
    leave: bool = True,
) -> None:
    measure(sdata=sdata, func=lambda s: _opening(s, proportion), shape_key=shape_key, result_keys=result_key, n_jobs=n_jobs, recompute=recompute, progress=progress, leave=leave)


