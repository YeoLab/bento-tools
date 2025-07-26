# Shape measurements are categorized into several categories depending on how they are calculated:
# - already implemented in GeoPandas: these will be simple wrappers around their existing functions since they are vectorized
# - not existing: these will be applied to each shape individually; parallelizable

from typing import Callable, List, Union

import dask.bag as db
import dask.config
import emoji
import numpy as np
import pandas as pd
from scipy.spatial import distance, distance_matrix
from shapely.geometry import Polygon, MultiPolygon, Point
from spatialdata import SpatialData
from tqdm.dask import TqdmCallback
from tqdm.auto import tqdm
from numba import njit
from bento._utils import get_shape
from bento._logging import logger


def measure(
    sdata: SpatialData,
    func: Callable,
    shape_key: str,
    result_keys: Union[str, List[str]],
    n_jobs: int = 1,
    recompute: bool = True,
    progress: bool = True,
    leave: bool = True,
) -> None:
    """Process shapes with parallel processing.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object
    func : Callable
        Function to apply to each shape
    shape_key : str
        Key to use for shapes
    result_keys : str or list of str
        Key(s) to use for results
    n_jobs : int, optional
        Number of workers to use for parallel processing

    Modifies
    -------
    sdata : SpatialData
        SpatialData object with results added to shapes at:
        - `shapes[shape_key][result_key]`
    """
    if not recompute and all(k in sdata.shapes[shape_key].columns for k in result_keys):
        logger.info(f"Skipping, recompute is False. {result_keys} exists in sdata.shapes[{shape_key}]")
        return

    # Process calculation
    shapes = get_shape(sdata, shape_key, sync=False).geometry
    shape_names = shapes.index.tolist()

    if n_jobs == 1:
        # No parallelism, no dask
        if progress:
            desc = emoji.emojize(":hourglass_not_done:")
            result = [func(shape) for shape in tqdm(shapes, desc=desc, leave=leave, mininterval=0.5)]
        else:
            result = [func(shape) for shape in shapes]
    else:
        shape_coords = np.array(shapes.apply(lambda x: np.array(x.exterior.xy)))
        # Use dask for parallel processing
        bags = db.from_sequence(shape_coords).map(func)
        dask.config.set(num_workers=n_jobs)
        if progress:
            desc = emoji.emojize(":hourglass_not_done:")
            with TqdmCallback(desc=desc, leave=leave):
                result = bags.compute()
        else:
            result = bags.compute()

    result = pd.DataFrame(result, index=shape_names)
    # Save results
    sdata.shapes[shape_key][result_keys] = result
    logger.info(f"`{result_keys}` saved to: sdata['{shape_key}']")


# ============================ FEATURE FUNCTIONS ============================


def _aspect_ratio(shape: Union[Polygon, MultiPolygon]) -> float:
    """Calculate aspect ratio of minimum rotated rectangle containing shape."""
    if not shape:
        return np.nan

    # Get coordinates of min bounding box vertices
    x, y = shape.minimum_rotated_rectangle.exterior.coords.xy

    # Get length of bound box sides
    edge_length = (
        Point(x[0], y[0]).distance(Point(x[1], y[1])),
        Point(x[1], y[1]).distance(Point(x[2], y[2])),
    )

    # length = longest side, width = shortest side
    length, width = max(edge_length), min(edge_length)

    return length / width


def _radius(shape: Union[Polygon, MultiPolygon]) -> float:
    """Calculate average radius of shape."""
    if not shape:
        return np.nan

    coords = np.array(shape.exterior.xy).T

    # Calculate centroid from coordinates
    centroid_x = np.mean(coords[:, 0])
    centroid_y = np.mean(coords[:, 1])

    # Calculate distances manually for numba compatibility
    distances = np.sqrt((coords[:, 0] - centroid_x) ** 2 + (coords[:, 1] - centroid_y) ** 2)
    return np.mean(distances)


def _span(shape: Union[Polygon, MultiPolygon]) -> float:
    """Calculate maximum diameter of shape."""
    if not shape:
        return np.nan

    shape_coo = np.array(shape.coords.xy).T
    return int(distance_matrix(shape_coo, shape_coo).max())


def _second_moment(shape: Union[Polygon, MultiPolygon]) -> float:
    """Calculate second moment of shape relative to its centroid."""
    if not shape:
        return np.nan

    centroid = np.array(shape.centroid.coords).reshape(1, 2)
    pts = np.array(shape.exterior.xy).T
    radii = distance.cdist(centroid, pts)
    return np.sum(radii * radii / len(pts))


def _opening(shape: Union[Polygon, MultiPolygon], proportion: float) -> Union[Polygon, MultiPolygon]:
    """Compute morphological opening of shape."""
    if not shape:
        return None

    # Calculate opening distance from shape radius
    d = proportion * _radius(shape)
    return shape.buffer(-d).buffer(d)


# ============================ GEOPANDAS FEATURE FUNCTIONS ============================


def _area(shape: Union[Polygon, MultiPolygon]) -> float:
    """Calculate area of shape."""
    if not shape:
        return np.nan
    return shape.area


def _perimeter(shape: Union[Polygon, MultiPolygon]) -> float:
    """Calculate perimeter length of shape."""
    if not shape:
        return np.nan
    return shape.length


def _bounds(shape: Union[Polygon, MultiPolygon]) -> tuple[float, float, float, float]:
    """Calculate bounding box coordinates of shape."""
    if not shape:
        return (np.nan, np.nan, np.nan, np.nan)
    return shape.bounds


def _centroid(shape: Union[Polygon, MultiPolygon]) -> tuple[float, float]:
    """Calculate centroid of shape from coordinate array."""
    if not shape:
        return (np.nan, np.nan)

    # Calculate centroid from coordinates
    centroid_x = np.mean(shape.exterior.xy[0])
    centroid_y = np.mean(shape.exterior.xy[1])

    return (centroid_x, centroid_y)


# ============================ PUBLIC API WRAPPERS ============================


def aspect_ratio(
    sdata: SpatialData,
    shape_key: str = "cell_boundaries",
    result_key: str = "aspect_ratio",
    n_jobs: int = 1,
    recompute: bool = True,
    progress: bool = True,
    leave: bool = True,
) -> None:
    """Calculate aspect ratio of minimum rotated rectangle containing each shape."""
    measure(
        sdata=sdata,
        func=_aspect_ratio,
        shape_key=shape_key,
        result_keys=result_key,
        n_jobs=n_jobs,
        recompute=recompute,
        progress=progress,
        leave=leave,
    )


def radius(
    sdata: SpatialData,
    shape_key: str = "cell_boundaries",
    result_key: str = "radius",
    n_jobs: int = 1,
    recompute: bool = True,
    progress: bool = True,
    leave: bool = True,
) -> None:
    """Calculate average radius of each shape."""
    measure(
        sdata=sdata,
        func=_radius,
        shape_key=shape_key,
        result_keys=result_key,
        n_jobs=n_jobs,
        recompute=recompute,
        progress=progress,
        leave=leave,
    )


def span(
    sdata: SpatialData,
    shape_key: str = "cell_boundaries",
    result_key: str = "span",
    n_jobs: int = 1,
    recompute: bool = True,
    progress: bool = True,
    leave: bool = True,
) -> None:
    """Calculate maximum diameter of each shape."""
    measure(
        sdata=sdata,
        func=_span,
        shape_key=shape_key,
        result_keys=result_key,
        n_jobs=n_jobs,
        recompute=recompute,
        progress=progress,
        leave=leave,
    )


def second_moment(
    sdata: SpatialData,
    shape_key: str = "cell_boundaries",
    result_key: str = "second_moment",
    n_jobs: int = 1,
    recompute: bool = True,
    progress: bool = True,
    leave: bool = True,
) -> None:
    """Calculate second moment of each shape relative to its centroid."""
    measure(
        sdata=sdata,
        func=_second_moment,
        shape_key=shape_key,
        result_keys=result_key,
        n_jobs=n_jobs,
        recompute=recompute,
        progress=progress,
        leave=leave,
    )


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
    """Compute morphological opening of each shape."""
    measure(
        sdata=sdata,
        func=lambda s: _opening(s, proportion),
        shape_key=shape_key,
        result_keys=result_key,
        n_jobs=n_jobs,
        recompute=recompute,
        progress=progress,
        leave=leave,
    )


def area(
    sdata: SpatialData,
    shape_key: str = "cell_boundaries",
    result_key: str = "area",
    n_jobs: int = 1,
    recompute: bool = True,
    progress: bool = True,
    leave: bool = True,
) -> None:
    """Calculate area of each shape."""
    measure(
        sdata=sdata,
        func=_area,
        shape_key=shape_key,
        result_keys=result_key,
        n_jobs=n_jobs,
        recompute=recompute,
        progress=progress,
        leave=leave,
    )


def perimeter(
    sdata: SpatialData,
    shape_key: str = "cell_boundaries",
    result_key: str = "perimeter",
    n_jobs: int = 1,
    recompute: bool = True,
    progress: bool = True,
    leave: bool = True,
) -> None:
    """Calculate perimeter length of each shape."""
    measure(
        sdata=sdata,
        func=_perimeter,
        shape_key=shape_key,
        result_keys=result_key,
        n_jobs=n_jobs,
        recompute=recompute,
        progress=progress,
        leave=leave,
    )


def bounds(
    sdata: SpatialData,
    shape_key: str = "cell_boundaries",
    result_keys: List[str] = ["xmin", "ymin", "xmax", "ymax"],
    n_jobs: int = 1,
    recompute: bool = True,
    progress: bool = True,
    leave: bool = True,
) -> None:
    """Calculate bounding box coordinates of each shape."""
    measure(
        sdata=sdata,
        func=_bounds,
        shape_key=shape_key,
        result_keys=result_keys,
        n_jobs=n_jobs,
        recompute=recompute,
        progress=progress,
        leave=leave,
    )


def centroid(
    sdata: SpatialData,
    shape_key: str = "cell_boundaries",
    result_keys: List[str] = ["x", "y"],
    n_jobs: int = 1,
    recompute: bool = True,
    progress: bool = True,
    leave: bool = True,
) -> None:
    """Calculate centroid of each shape."""
    measure(
        sdata=sdata,
        func=_centroid,
        shape_key=shape_key,
        result_keys=result_keys,
        n_jobs=n_jobs,
        recompute=recompute,
        progress=progress,
        leave=leave,
    )
