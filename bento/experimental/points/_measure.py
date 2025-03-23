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


def _enable_gene_groups(func: Callable) -> Callable:
    """Enable gene groups for point feature functions."""

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


def _apply_func(
    sdata: SpatialData,
    func: Callable,
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

    point_feature_func = _enable_gene_groups(func)

    # Create processing bags
    args = [
        (points_by_shape.get_group(s), shape.iloc[i], gene_key)
        for i, s in enumerate(shape_names)
    ]
    bags = db.from_sequence(args).map(lambda x: point_feature_func(*x))

    core_text = "cores" if num_workers > 1 else "core"
    # Compute results with progress bar
    with (
        TqdmCallback(desc=f"Using {num_workers} {core_text}"),
        dask.config.set(
            num_workers=num_workers,
            threads_per_worker=1,
        ),
    ):
        result = bags.compute()

    if not by_gene:
        result = pd.Series(result, index=shape_names)
    else:
        result = pd.DataFrame(result, index=shape_names)

    return result


def measure(
    sdata: SpatialData,
    func: Callable,
    by_gene: bool = False,
    points_key: str = "transcripts",
    shape_key: str = "cell_boundaries",
    result_key: str = None,
    num_workers: int = 1,
) -> pd.DataFrame:
    """Process points with parallel processing.

    This function applies a function to each shape in the SpatialData object.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object
    fn : Callable
        Function to apply to each shape
    points_key : str, optional
        Key to use for points
    shape_key : str, optional
        Key to use for shapes
    by_gene : bool, optional
        Whether to apply the function by gene
    result_key : str, optional
        Key to use for results
    num_workers : int, optional
        Number of workers to use for parallel processing

    Modifies
    -------
    sdata : SpatialData
        SpatialData object with results added to shapes or tables at:
        - `shapes[shape_key][result_key]` if `by_gene=False`
        - `tables[result_key]` if `by_gene=True`
    """

    # Process calculation
    result = _apply_func(
        sdata=sdata,
        func=func,
        points_key=points_key,
        shape_key=shape_key,
        by_gene=by_gene,
        num_workers=num_workers,
    )

    # Save results
    if not by_gene:
        sdata.shapes[shape_key][result_key] = result
        print(f"Saved to: sdata['{shape_key}']['{result_key}']")
    else:
        sdata.tables[result_key] = AnnData(
            result,
            obs=pd.DataFrame(index=result.index),
            var=pd.DataFrame(index=result.columns),
        )
        print(f"Saved to: sdata['{result_key}']")


# ============================ FEATURE FUNCTIONS ============================


def _edge_distance(points: pd.DataFrame, shape: Union[Polygon, MultiPolygon]) -> float:
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


def _centroid_distance(
    points: pd.DataFrame, shape: Union[Polygon, MultiPolygon]
) -> float:
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


def edge_distance(
    sdata: SpatialData,
    points_key: str = "transcripts",
    shape_key: str = "cell_boundaries",
    by_gene: bool = False,
    result_key: str = "tx_cell_edge_dist",
    num_workers: int = 1,
) -> None:
    """Calculate mean distance from points to shape edge.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object
    """

    measure(
        sdata=sdata,
        func=_edge_distance,
        points_key=points_key,
        shape_key=shape_key,
        by_gene=by_gene,
        result_key=result_key,
        num_workers=num_workers,
    )


def centroid_distance(
    sdata: SpatialData,
    points_key: str = "transcripts",
    shape_key: str = "cell_boundaries",
    by_gene: bool = False,
    result_key: str = "tx_cell_centroid_dist",
    num_workers: int = 1,
) -> None:
    """Calculate mean distance from points to shape centroid."""

    measure(
        sdata=sdata,
        func=_centroid_distance,
        points_key=points_key,
        shape_key=shape_key,
        by_gene=by_gene,
        result_key=result_key,
        num_workers=num_workers,
    )


def density(
    sdata: SpatialData,
    points_key: str = "transcripts",
    shape_key: str = "cell_boundaries",
    by_gene: bool = False,
    result_key: str = "tx_cell_density",
    num_workers: int = 1,
) -> None:
    """Calculate density of points within shape.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object
    """
    measure(
        sdata=sdata,
        func=_density,
        points_key=points_key,
        shape_key=shape_key,
        by_gene=by_gene,
        result_key=result_key,
        num_workers=num_workers,
    )
