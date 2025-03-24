# Shape measurements are categorized into several categories depending on how they are calculated:
# - already implemented in GeoPandas: these will be simple wrappers around their existing functions since they are vectorized
# - not existing: these will be applied to each shape individually; parallelizable

from typing import Callable, List, Union

import dask.bag as db
import dask.config
import numpy as np
import pandas as pd
from scipy.spatial import distance, distance_matrix
from shapely.geometry import Polygon, MultiPolygon, Point
from spatialdata import SpatialData
from tqdm.dask import TqdmCallback

from bento._utils import get_shape


def _apply_func(
    sdata: SpatialData,
    func: Callable,
    shape_key: str,
    num_workers: int = 1,
) -> pd.Series:
    """Internal function to process shape calculations with parallel processing."""
    # Get shape data
    shapes = get_shape(sdata, shape_key, sync=False).geometry
    shape_names = shapes.index.tolist()

    # Create processing bags
    bags = db.from_sequence(shapes).map(func)

    # Compute results with progress bar
    with TqdmCallback(desc="Processing"), dask.config.set(num_workers=num_workers):
        result = bags.compute()

    return pd.Series(result, index=shape_names)


def measure(
    sdata: SpatialData,
    func: Callable,
    shape_key: str,
    result_keys: Union[str, List[str]],
    num_workers: int = 1,
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
    num_workers : int, optional
        Number of workers to use for parallel processing

    Modifies
    -------
    sdata : SpatialData
        SpatialData object with results added to shapes at:
        - `shapes[shape_key][result_key]`
    """
    # Process calculation
    result = _apply_func(
        sdata=sdata,
        func=func,
        shape_key=shape_key,
        num_workers=num_workers,
    )

    # Save results
    sdata.shapes[shape_key][result_keys] = result
    print(f"""Saved to: sdata['{shape_key}']
          column(s): {result_keys}
          """)


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

    return distance.cdist(
        np.array(shape.centroid.coords).reshape(1, 2), np.array(shape.exterior.xy).T
    ).mean()


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


def _opening(
    shape: Union[Polygon, MultiPolygon], proportion: float
) -> Union[Polygon, MultiPolygon]:
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


def _centroid(shape: Union[Polygon, MultiPolygon]) -> Point:
    """Calculate centroid of shape."""
    if not shape:
        return None
    return shape.centroid


# ============================ PUBLIC API WRAPPERS ============================


def aspect_ratio(
    sdata: SpatialData,
    shape_key: str = "cell_boundaries",
    result_key: str = "aspect_ratio",
    num_workers: int = 1,
) -> None:
    """Calculate aspect ratio of minimum rotated rectangle containing each shape."""
    measure(
        sdata=sdata,
        func=_aspect_ratio,
        shape_key=shape_key,
        result_keys=result_key,
        num_workers=num_workers,
    )


def radius(
    sdata: SpatialData,
    shape_key: str = "cell_boundaries",
    result_key: str = "radius",
    num_workers: int = 1,
) -> None:
    """Calculate average radius of each shape."""
    measure(
        sdata=sdata,
        func=_radius,
        shape_key=shape_key,
        result_keys=result_key,
        num_workers=num_workers,
    )


def span(
    sdata: SpatialData,
    shape_key: str = "cell_boundaries",
    result_key: str = "span",
    num_workers: int = 1,
) -> None:
    """Calculate maximum diameter of each shape."""
    measure(
        sdata=sdata,
        func=_span,
        shape_key=shape_key,
        result_keys=result_key,
        num_workers=num_workers,
    )


def second_moment(
    sdata: SpatialData,
    shape_key: str = "cell_boundaries",
    result_key: str = "second_moment",
    num_workers: int = 1,
) -> None:
    """Calculate second moment of each shape relative to its centroid."""
    measure(
        sdata=sdata,
        func=_second_moment,
        shape_key=shape_key,
        result_keys=result_key,
        num_workers=num_workers,
    )


def opening(
    sdata: SpatialData,
    shape_key: str = "cell_boundaries",
    proportion: float = 0.1,
    result_key: str = "opened_shape",
    num_workers: int = 1,
) -> None:
    """Compute morphological opening of each shape."""
    measure(
        sdata=sdata,
        func=lambda s: _opening(s, proportion),
        shape_key=shape_key,
        result_keys=result_key,
        num_workers=num_workers,
    )


def area(
    sdata: SpatialData,
    shape_key: str = "cell_boundaries",
    result_key: str = "area",
    num_workers: int = 1,
) -> None:
    """Calculate area of each shape."""
    measure(
        sdata=sdata,
        func=_area,
        shape_key=shape_key,
        result_keys=result_key,
        num_workers=num_workers,
    )


def perimeter(
    sdata: SpatialData,
    shape_key: str = "cell_boundaries",
    result_key: str = "perimeter",
    num_workers: int = 1,
) -> None:
    """Calculate perimeter length of each shape."""
    measure(
        sdata=sdata,
        func=_perimeter,
        shape_key=shape_key,
        result_keys=result_key,
        num_workers=num_workers,
    )


def bounds(
    sdata: SpatialData,
    shape_key: str = "cell_boundaries",
    result_keys: List[str] = ["xmin", "ymin", "xmax", "ymax"],
    num_workers: int = 1,
) -> None:
    """Calculate bounding box coordinates of each shape."""
    measure(
        sdata=sdata,
        func=_bounds,
        shape_key=shape_key,
        result_keys=result_keys,
        num_workers=num_workers,
    )


def centroid(
    sdata: SpatialData,
    shape_key: str = "cell_boundaries",
    result_keys: List[str] = ["x", "y"],
    num_workers: int = 1,
) -> None:
    """Calculate centroid of each shape."""
    measure(
        sdata=sdata,
        func=_centroid,
        shape_key=shape_key,
        result_keys=result_keys,
        num_workers=num_workers,
    )
