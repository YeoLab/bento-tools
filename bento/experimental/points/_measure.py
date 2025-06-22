from functools import wraps
from typing import Callable, Union, List

import spatialdata as sd
import dask
import dask.bag as db
import dask.dataframe as dd
import emoji
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix
from shapely.geometry import MultiPolygon, Polygon
from spatialdata import SpatialData
from tqdm.auto import tqdm

from bento._utils import get_feature_key, get_points
from bento._logging import logger

from .. import shapes as shp

from ._distance import _distances
from ._polarity import _polarity
from ._density import _density
from ._moments import _hu_moments
from ._ripley import _ripley
from ._ripley import _morans_i


def _enable_gene_groups(func: Callable) -> Callable:
    """Enable gene groups for point feature functions."""

    @wraps(func)
    def wrapper(
        points: pd.DataFrame,
        gene_key: str = None,
        shape: Union[Polygon, MultiPolygon] = None,
        **kwargs,
    ) -> dict:
        if gene_key:
            result = points.groupby(gene_key, observed=True).apply(
                lambda x: func(x, shape=shape, **kwargs)
            )
        else:
            result = func(points, shape=shape, **kwargs)
        return result

    return wrapper


def measure(
    sdata: SpatialData,
    func: Callable,
    points_key: str,
    shape_key: str,
    feature_key: str,
    shape_measure_keys: List[str] = None,
    result_key: str = None,
    n_jobs: int = 1,
    recompute: bool = True,
    progress: bool = True,
    leave: bool = True,
) -> pd.DataFrame:
    """Process point calculations with parallel processing.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object
    func : Callable
        Function to apply to points in shapes
    points_key : str
        Key to use for points
    shape_key : str
        Key to use for shapes
    shape_measure_keys : List[str]
        Keys to query for shape measurements
    feature_key : str
        Key to use for features
    result_key : str, optional
        Key to use for results
    n_jobs : int, optional
        Number of jobs to use for parallel processing
    recompute : bool, optional
        Whether to recompute the result
    progress : bool, optional
        Whether to show a progress bar
    leave : bool, optional
        Whether to leave the progress bar open

    Modifies
    -------
    sdata : SpatialData object with results added to:
        - `tables[result_key]` <n_labels, n_channels> with each metric stored as a layer
    """
    if not recompute and result_key in sdata.tables:
        logger.info(
            f"Skipping, recompute is False. {result_key} exists in sdata.tables[{result_key}]"
        )
        return

    # Get points data
    points = get_points(sdata, points_key=points_key, astype="dask", sync=True)

    # Pre-compute groups for faster access
    points = points[
        ["x", "y", shape_key, feature_key]
    ].compute()  # Convert to pandas for faster group operations
    points_by_shape = points.groupby(shape_key)
    group_indices = points_by_shape.indices
    shape_names = list(group_indices.keys())

    # Get shape data
    if shape_measure_keys:
        shape_data = (
            sdata.shapes[shape_key]
            .loc[shape_names, ["geometry", *shape_measure_keys]]
            .values
        )
    else:
        shape_data = sdata.shapes[shape_key].loc[shape_names, ["geometry"]].values

    point_feature_func = _enable_gene_groups(func)

    # Process shapes in batches to reduce memory usage
    batch_size = min(len(shape_names), 5000)
    partition_size = min(100, batch_size // (3 * n_jobs))
    results = []

    core_text = "cores" if n_jobs > 1 else "core"
    logger.info(f"Computing with {n_jobs} {core_text} in batches of {batch_size}")

    # Process in batches and stream results
    dask.config.set(num_workers=n_jobs, threads_per_worker=1)
    valid_shape_names = []
    if progress:
        pbar = tqdm(
            total=len(shape_names),
            desc=emoji.emojize(":hourglass_not_done:"),
            leave=leave,
        )
    for i in range(0, len(shape_names), batch_size):
        batch_names = shape_names[i : i + batch_size]
        batch_points = [points.iloc[group_indices[s]] for s in batch_names]
        batch_shape_data = shape_data[i : i + batch_size]

        # Create processing bags for this batch only using direct indexing
        batch_args = []
        for i, s in enumerate(batch_names):
            try:
                # Get all required shape measurements
                shape_measurements = batch_shape_data[i]
                # First element is always the geometry
                shape = shape_measurements[0]
                # Remaining elements are the measurements
                measurements = shape_measurements[1:]
                batch_args.append((batch_points[i], feature_key, shape, *measurements))
                valid_shape_names.append(batch_names[i])
            except KeyError:
                # Shape not in shapes
                logger.debug(f"Shape {s} has no points")
                continue
        # Process this batch
        batch_bags = db.from_sequence(batch_args, partition_size=partition_size).map(
            lambda x: point_feature_func(
                x[0], x[1], shape=x[2], **dict(zip(shape_measure_keys, x[3:]))
            )
        )
        batch_results = batch_bags.compute()
        results.extend(batch_results)
        pbar.update(len(batch_names)) if progress else None

    pbar.close() if progress else None
    pbar.set_description(f"{emoji.emojize(':bento:')} Done.") if progress else None
    logger.debug("Compiling results")
    for i, r in enumerate(results):
        cell_result = pd.DataFrame.from_records(r.values)
        cell_result[feature_key] = r.index.tolist()
        cell_result[shape_key] = valid_shape_names[i]
        results[i] = cell_result
    results = pd.concat(results) if len(results) > 0 else results

    var_names = list(results[feature_key].unique())

    # Reshape results, one df per metric with labels as index and channels as columns
    result_layers = {}
    metrics = results.columns.drop([feature_key, shape_key])
    for metric in metrics:
        result_layers[metric] = results.pivot(
            index=shape_key, columns=feature_key, values=metric
        )

        # Add nan for invalid shapes
        result_layers[metric] = (
            result_layers[metric].reindex(shape_names).fillna(np.nan)
        )

    if result_key in sdata:
        sdata[result_key].layers = result_layers
    else:
        empty_x = csr_matrix(np.zeros_like(result_layers[metrics[0]]))
        # Set row and column names to match the dataframe
        table = AnnData(X=empty_x, layers=result_layers)
        table.obs_names = shape_names
        table.var_names = var_names
        table.obs["region"] = shape_key
        table.obs["instance"] = table.obs.index
        sdata[result_key] = sd.models.TableModel.parse(table)
        sdata.set_table_annotates_spatialelement(
            result_key, shape_key, region_key="region", instance_key="instance"
        )
    logger.info(f"Saved to: sdata['{result_key}']")


def distance(
    sdata: SpatialData,
    points_key: str = "transcripts",
    shape_key: str = "cell_boundaries",
    feature_key: str = "feature_name",
    result_key: str = "tx_distance",
    n_jobs: int = 1,
    recompute: bool = True,
    progress: bool = True,
    leave: bool = True,
) -> None:
    """Calculate distance stats from points to shape.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object
    """
    shp.radius(
        sdata,
        shape_key=shape_key,
        n_jobs=n_jobs,
        recompute=recompute,
        progress=False,
        leave=False,
    )

    measure(
        sdata=sdata,
        func=_distances,
        points_key=points_key,
        shape_key=shape_key,
        shape_measure_keys=["radius"],
        feature_key=feature_key,
        result_key=result_key,
        n_jobs=n_jobs,
        recompute=recompute,
        progress=progress,
        leave=leave,
    )


def polarity(
    sdata: SpatialData,
    points_key: str = "transcripts",
    shape_key: str = "cell_boundaries",
    feature_key: str = "feature_name",
    result_key: str = "tx_polarity",
    n_jobs: int = 1,
    recompute: bool = True,
    progress: bool = True,
    leave: bool = True,
) -> None:
    """Calculate polarity of points within shape.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object
    """
    shp.centroid(
        sdata,
        shape_key=shape_key,
        n_jobs=n_jobs,
        recompute=recompute,
        progress=False,
        leave=False,
    )
    shp.radius(
        sdata,
        shape_key=shape_key,
        n_jobs=n_jobs,
        recompute=recompute,
        progress=False,
        leave=False,
    )
    measure(
        sdata=sdata,
        func=_polarity,
        points_key=points_key,
        shape_key=shape_key,
        shape_measure_keys=["radius", "x", "y"],
        feature_key=feature_key,
        result_key=result_key,
        n_jobs=n_jobs,
        recompute=recompute,
        progress=progress,
        leave=leave,
    )


def density(
    sdata: SpatialData,
    points_key: str = "transcripts",
    shape_key: str = "cell_boundaries",
    feature_key: str = "feature_name",
    result_key: str = "tx_density",
    n_jobs: int = 1,
    recompute: bool = True,
    progress: bool = True,
    leave: bool = True,
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
        feature_key=feature_key,
        result_key=result_key,
        n_jobs=n_jobs,
        recompute=recompute,
        progress=progress,
        leave=leave,
    )


def moments(
    sdata: SpatialData,
    points_key: str = "transcripts",
    shape_key: str = "cell_boundaries",
    feature_key: str = "feature_name",
    result_key: str = "tx_moments",
    n_jobs: int = 1,
    recompute: bool = True,
    progress: bool = True,
    leave: bool = True,
) -> None:
    """Calculate relative hu moments of points to shape.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object
    """
    shp.centroid(
        sdata,
        shape_key=shape_key,
        n_jobs=n_jobs,
        recompute=recompute,
        progress=False,
        leave=False,
    )
    measure(
        sdata=sdata,
        func=_hu_moments,
        points_key=points_key,
        shape_key=shape_key,
        shape_measure_keys=["x", "y"],
        feature_key=feature_key,
        result_key=result_key,
        n_jobs=n_jobs,
        recompute=recompute,
        progress=progress,
        leave=leave,
    )


def morans_i(
    sdata: SpatialData,
    points_key: str = "transcripts",
    shape_key: str = "cell_boundaries",
    feature_key: str = "feature_name",
    result_key: str = "tx_morans_i",
    n_jobs: int = 1,
    recompute: bool = True,
    progress: bool = True,
    leave: bool = True,
) -> None:
    """Calculate Moran's I for points within shape.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object
    points_key : str, optional
        Key for points in sdata.points, by default "transcripts"
    shape_key : str, optional
        Key for shapes in sdata.shapes, by default "cell_boundaries"
    feature_key : str, optional
        Key for features in sdata.features, by default "feature_name"
    result_key : str, optional
        Key for results, by default "tx_morans_i"
    n_jobs : int, optional
        Number of workers for parallel processing, by default 1

    Modifies
    -------
    sdata : SpatialData
        Adds the following metrics to shapes:
        - morans_i: Moran's I statistic
        - morans_p: P-value for Moran's I statistic
        - morans_z: Z-score for Moran's I statistic
    """
    measure(
        sdata=sdata,
        func=_morans_i,
        points_key=points_key,
        shape_key=shape_key,
        feature_key=feature_key,
        result_key=result_key,
        n_jobs=n_jobs,
        recompute=recompute,
        progress=progress,
        leave=leave,
    )


def ripley(
    sdata: SpatialData,
    points_key: str = "transcripts",
    shape_key: str = "cell_boundaries",
    feature_key: str = "feature_name",
    result_key: str = "tx_ripley_stats",
    n_jobs: int = 1,
    recompute: bool = True,
    progress: bool = True,
    leave: bool = True,
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
    n_jobs : int, optional
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
    measure(
        sdata=sdata,
        func=_ripley,
        points_key=points_key,
        shape_key=shape_key,
        feature_key=feature_key,
        result_key=result_key,
        n_jobs=n_jobs,
        recompute=recompute,
        progress=progress,
        leave=leave,
    )
