from functools import wraps
from typing import Callable, Union, List, Dict, Tuple

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
            result = points.groupby(gene_key, observed=True).apply(lambda x: func(x, shape=shape, **kwargs))
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
        logger.info(f"Skipping, recompute is False. {result_key} exists in sdata.tables[{result_key}]")
        return

    # Get points data
    points = get_points(sdata, points_key=points_key, astype="dask", sync=True)

    # Pre-compute groups for faster access
    points = points[["x", "y", shape_key, feature_key]].compute()  # Convert to pandas for faster group operations
    points_by_shape = points.groupby(shape_key)
    group_indices = points_by_shape.indices
    shape_names = list(group_indices.keys())

    # Get shape data
    if shape_measure_keys:
        shape_data = sdata.shapes[shape_key].loc[shape_names, ["geometry", *shape_measure_keys]].values
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
                lambda x: point_feature_func(x[0], x[1], shape=x[2], **dict(zip(shape_measure_keys or [], x[3:])))
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
        result_layers[metric] = results.pivot(index=shape_key, columns=feature_key, values=metric)

        # Add nan for invalid shapes
        result_layers[metric] = result_layers[metric].reindex(shape_names).fillna(np.nan)

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
        sdata.set_table_annotates_spatialelement(result_key, shape_key, region_key="region", instance_key="instance")
    logger.info(f"Saved to: sdata['{result_key}']")


