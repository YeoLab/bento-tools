from typing import Callable, List

import dask
import dask.bag as db
import numpy as np
import pandas as pd
import spatialdata as sd
from anndata import AnnData
from scipy.sparse import csr_matrix
from spatialdata import SpatialData

from bento._logging import logger
from bento._utils import get_points


def _enable_gene_groups(func: Callable) -> Callable:
    def wrapper(points: pd.DataFrame, gene_key: str = None, shape=None, **kwargs) -> dict:
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
    if not recompute and result_key in sdata.tables:
        logger.info(f"Skipping, recompute is False. {result_key} exists in sdata.tables[{result_key}]")
        return

    points = get_points(sdata, points_key=points_key, astype="dask", sync=True)
    points = points[["x", "y", shape_key, feature_key]].compute()
    points_by_shape = points.groupby(shape_key)
    group_indices = points_by_shape.indices
    shape_names = list(group_indices.keys())

    if shape_measure_keys:
        shape_data = sdata.shapes[shape_key].loc[shape_names, ["geometry", *shape_measure_keys]].values
    else:
        shape_data = sdata.shapes[shape_key].loc[shape_names, ["geometry"]].values

    point_feature_func = _enable_gene_groups(func)

    batch_size = min(len(shape_names), 5000)
    partition_size = min(100, batch_size // (3 * n_jobs))
    results = []

    core_text = "cores" if n_jobs > 1 else "core"
    logger.info(f"Computing with {n_jobs} {core_text} in batches of {batch_size}")

    dask.config.set(num_workers=n_jobs, threads_per_worker=1)
    valid_shape_names = []

    pbar = None
    if progress:
        from tqdm.auto import tqdm as _tqdm

        pbar = _tqdm(total=len(shape_names), desc="Processing", leave=leave)

    for i in range(0, len(shape_names), batch_size):
        batch_names = shape_names[i : i + batch_size]
        batch_points = [points.iloc[group_indices[s]] for s in batch_names]
        batch_shape_data = shape_data[i : i + batch_size]

        batch_args = []
        for j, _ in enumerate(batch_names):
            shape_measurements = batch_shape_data[j]
            shape_geom = shape_measurements[0]
            measurements = shape_measurements[1:]
            batch_args.append((batch_points[j], feature_key, shape_geom, *measurements))
            valid_shape_names.append(batch_names[j])

        batch_bags = db.from_sequence(batch_args, partition_size=partition_size).map(
            lambda x: point_feature_func(x[0], x[1], shape=x[2], **dict(zip(shape_measure_keys or [], x[3:])))
        )
        batch_results = batch_bags.compute()
        results.extend(batch_results)
        if pbar:
            pbar.update(len(batch_names))

    if pbar:
        pbar.close()

    for i, r in enumerate(results):
        cell_result = pd.DataFrame.from_records(r.values)
        cell_result[feature_key] = r.index.tolist()
        cell_result[shape_key] = valid_shape_names[i]
        results[i] = cell_result
    results = pd.concat(results) if len(results) > 0 else results

    var_names = list(results[feature_key].unique()) if len(results) > 0 else []

    result_layers = {}
    if len(results) > 0:
        metrics = results.columns.drop([feature_key, shape_key])
        for metric in metrics:
            result_layers[metric] = results.pivot(index=shape_key, columns=feature_key, values=metric)
            result_layers[metric] = result_layers[metric].reindex(shape_names).fillna(np.nan)

    if result_key in sdata:
        sdata[result_key].layers = result_layers
    else:
        if len(result_layers) == 0:
            empty = np.zeros((len(shape_names), len(var_names)), dtype=float)
            empty_x = csr_matrix(empty)
            table = AnnData(X=empty_x, layers={})
        else:
            first = next(iter(result_layers.values()))
            empty_x = csr_matrix(np.zeros_like(first))
            table = AnnData(X=empty_x, layers=result_layers)
        table.obs_names = shape_names
        table.var_names = var_names
        table.obs["region"] = shape_key
        table.obs["instance"] = table.obs.index
        sdata[result_key] = sd.models.TableModel.parse(table)
        sdata.set_table_annotates_spatialelement(result_key, shape_key, region_key="region", instance_key="instance")
    logger.info(f"Saved to: sdata['{result_key}']")


