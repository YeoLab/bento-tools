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
    func : Callable
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


def _span(shape: Union[Polygon, MultiPolygon]) -> float:
    """Calculate maximum diameter of shape."""
    if not shape:
        return np.nan

    shape_coo = np.array(shape.exterior.coords.xy).T
    return distance_matrix(shape_coo, shape_coo).max()


def _ripley(
    points: pd.DataFrame,
    shape: Union[Polygon, MultiPolygon],
) -> dict:
    """Calculate Ripley's L-function statistics for point patterns."""
    if not shape or len(points) < 2:
        return {
            "l_max": np.nan,
            "l_max_gradient": np.nan,
            "l_min_gradient": np.nan,
            "l_monotony": np.nan,
            "l_half_radius": np.nan,
        }

    # Get cell properties
    cell_span = _span(shape)
    cell_minx, cell_miny, cell_maxx, cell_maxy = shape.bounds
    cell_area = shape.area

    estimator = RipleysKEstimator(
        area=cell_area,
        x_min=cell_minx,
        y_min=cell_miny,
        x_max=cell_maxx,
        y_max=cell_maxy,
    )

    quarter_span = cell_span / 4
    radii = np.linspace(1, quarter_span * 2, num=int(quarter_span * 2))

    # Get points coordinates
    points_geo = np.array([points.geometry.x, points.geometry.y]).T

    # Compute ripley function stats
    stats = estimator.Hfunction(data=points_geo, radii=radii, mode="none")

    # Max value of the L-function
    l_max = max(stats)

    # Max and min value of the gradient of L
    ripley_smooth = pd.Series(stats).rolling(5).mean()
    ripley_smooth.dropna(inplace=True)

    # Can't take gradient of single number
    if len(ripley_smooth) < 2:
        ripley_smooth = np.array([0, 0])

    ripley_gradient = np.gradient(ripley_smooth)
    l_max_gradient = ripley_gradient.max()
    l_min_gradient = ripley_gradient.min()

    # Monotony of L-function in the interval
    l_monotony = spearmanr(radii, stats)[0]

    # L-function at L/4 where length of the cell L is max dist between 2 points on polygon defining cell border
    l_half_radius = estimator.Hfunction(
        data=points_geo, radii=[quarter_span], mode="none"
    )[0]

    return {
        "l_max": l_max,
        "l_max_gradient": l_max_gradient,
        "l_min_gradient": l_min_gradient,
        "l_monotony": l_monotony,
        "l_half_radius": l_half_radius,
    }


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


from scipy.stats import spearmanr
import numpy as np
from astropy.stats import RipleysKEstimator
from scipy.spatial import distance_matrix


def ripley(
    sdata: SpatialData,
    points_key: str = "transcripts",
    shape_key: str = "cell_boundaries",
    by_gene: bool = False,
    result_key: str = "tx_ripley_stats",
    num_workers: int = 1,
) -> None:
    """Calculate Ripley's L-function statistics for point patterns.

    The L-function is evaluated at r=[1,d], where d is half the cell's maximum diameter.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object
    points_key : str, optional
        Key for points in sdata.points, by default "transcripts"
    shape_key : str, optional
        Key for shapes in sdata.shapes, by default "cell_boundaries"
    by_gene : bool, optional
        Whether to calculate statistics per gene, by default False
    result_key : str, optional
        Key for results, by default "tx_ripley_stats"
    num_workers : int, optional
        Number of workers for parallel processing, by default 1

    Modifies
    -------
    sdata : SpatialData
        Adds the following metrics to shapes:
        - l_max: Maximum value of L-function
        - l_max_gradient: Maximum gradient of L-function
        - l_min_gradient: Minimum gradient of L-function
        - l_monotony: Spearman correlation between L-function and radius
        - l_half_radius: L-function value at quarter cell diameter
    """
    gene_key = "gene" if by_gene else None
    measure(
        sdata=sdata,
        func=_ripley,
        points_key=points_key,
        shape_key=shape_key,
        gene_key=gene_key,
        result_key=result_key,
        num_workers=num_workers,
    )
