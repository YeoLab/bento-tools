from functools import wraps
from typing import Callable, Union

import dask
import dask.bag as db
import dask.dataframe as dd
import geopandas as gpd
import pandas as pd
from anndata import AnnData
from shapely.geometry import Polygon, MultiPolygon
from spatialdata import SpatialData
from tqdm.dask import TqdmCallback

from bento._utils import get_points, get_feature_key


def point_feature(func: Callable) -> Callable:
    """Internal decorator for point feature functions that handles preprocessing."""

    @wraps(func)
    def wrapper(
        points: dd.DataFrame,
        shape: Union[Polygon, MultiPolygon],
        gene_key: str = None,
    ) -> dict:
        points = points.compute()  # Convert from dask to pandas
        if gene_key:
            result = points.groupby(gene_key, observed=True).apply(
                lambda x: func(x, shape)
            )
        else:
            result = func(points, shape)
        return result

    return wrapper


def process_point_calculation(
    sdata: SpatialData,
    calc_func: Callable,
    points_key: str = "transcripts",
    shape_key: str = "cell_boundaries",
    by_gene: bool = False,
    num_workers: int = 1,
) -> pd.DataFrame:
    """Internal function to process point calculations with parallel processing."""
    # Get points data
    points = get_points(sdata, points_key=points_key, astype="dask", sync=True)

    # Get shape data
    shape = sdata.shapes[shape_key].sort_index().geometry
    shape_names = shape.index.tolist()

    points_by_shape = points.groupby(shape_key)

    if by_gene:
        gene_key = get_feature_key(sdata)
    else:
        gene_key = None

    # Create processing bags
    args = [
        (points_by_shape.get_group(s), shape.iloc[i], gene_key)
        for i, s in enumerate(shape_names)
    ]
    bags = db.from_sequence(args).map(lambda x: calc_func(*x))

    # Compute results with progress bar
    with TqdmCallback(desc="Processing"), dask.config.set(num_workers=num_workers):
        result = bags.compute()

    if not by_gene:
        result = pd.Series(result, index=shape_names)
    else:
        result = pd.DataFrame(result, index=shape_names)

    return result


def catalog(key: str = None):
    """Decorator to register a new point feature calculation.

    This decorator handles all the boilerplate of:
    1. Converting points to pandas DataFrame
    2. Optional gene grouping
    3. Parallel processing across shapes
    4. Saving results to the SpatialData object

    Parameters
    ----------
    key : str, optional
        Default key to use when storing results. If None, uses the function name.

    Example
    -------
    @register_point_feature(key="my_feature")
    def my_feature(points, shape):
        '''Calculate some feature of points within a shape.'''
        # points is a pandas DataFrame with x,y columns
        # shape is a Polygon/MultiPolygon
        return some_calculation(points, shape)

    # Use like other features
    my_feature(sdata, by_gene=True)
    """

    def decorator(func: Callable):
        feature_key = key or func.__name__

        @wraps(func)
        def wrapper(
            sdata: SpatialData,
            points_key: str = "transcripts",
            shape_key: str = "cell_boundaries",
            by_gene: bool = False,
            key: str = None,
            num_workers: int = 1,
        ) -> None:
            # Use provided key or default
            result_key = key or feature_key

            # Add point_feature decorator
            decorated_func = point_feature(func)

            # Process calculation
            result = process_point_calculation(
                sdata=sdata,
                calc_func=decorated_func,
                points_key=points_key,
                shape_key=shape_key,
                by_gene=by_gene,
                num_workers=num_workers,
            )

            # Save results
            if not by_gene:
                sdata.shapes[shape_key][result_key] = result
            else:
                sdata.tables[result_key] = AnnData(
                    result,
                    obs=pd.DataFrame(index=result.index),
                    var=pd.DataFrame(index=result.columns),
                )

        return wrapper

    return decorator


# ============================ FEATURE FUNCTIONS ============================


def _distance_edge(points: pd.DataFrame, shape: Union[Polygon, MultiPolygon]) -> float:
    """Calculate mean distance from points to shape edge.

    Parameters
    ----------
    points : pd.DataFrame
        Points with x,y coordinates
    shape : Union[Polygon, MultiPolygon]
        Shape to calculate distance to

    Returns
    -------
    float
        Mean distance from points to shape edge
    """
    points = gpd.GeoSeries.from_xy(points["x"], points["y"])
    return points.distance(shape).mean()


@catalog(key="tx_offset")
def _offset(points: pd.DataFrame, shape: Union[Polygon, MultiPolygon]) -> float:
    """Calculate mean distance from points to shape centroid.

    Parameters
    ----------
    points : pd.DataFrame
        Points with x,y coordinates
    shape : Union[Polygon, MultiPolygon]
        Shape to calculate offset from

    Returns
    -------
    float
        Mean distance from points to shape centroid
    """
    points = gpd.GeoSeries.from_xy(points["x"], points["y"])
    centroid = shape.centroid
    return points.distance(centroid).mean()


@catalog(key="tx_density")
def _density(points: pd.DataFrame, shape: Union[Polygon, MultiPolygon]) -> float:
    """Calculate density of points within shape.

    Parameters
    ----------
    points : pd.DataFrame
        Points with x,y coordinates
    shape : Union[Polygon, MultiPolygon]
        Shape to calculate density within

    Returns
    -------
    float
        Number of points divided by shape area
    """
    return len(points) / shape.area


# ============================ PUBLIC API WRAPPERS ============================


@catalog(key="tx_distance_edge")
def distance_edge(
    sdata: SpatialData,
    points_key: str = "transcripts",
    shape_key: str = "cell_boundaries",
    by_gene: bool = False,
    key: str = None,
    num_workers: int = 1,
) -> None:
    """Calculate mean distance from points to shape edge.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object
    """
    _offset(
        sdata=sdata,
        points_key=points_key,
        shape_key=shape_key,
        by_gene=by_gene,
        key=key,
        num_workers=num_workers,
    )


@catalog(key="tx_density")
def density(
    sdata: SpatialData,
    points_key: str = "transcripts",
    shape_key: str = "cell_boundaries",
    by_gene: bool = False,
    key: str = None,
    num_workers: int = 1,
) -> None:
    """Calculate density of points within shape.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object
    """
    _density(
        sdata=sdata,
        points_key=points_key,
        shape_key=shape_key,
        by_gene=by_gene,
        key=key,
        num_workers=num_workers,
    )
