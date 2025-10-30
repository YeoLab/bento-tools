from typing import List, Dict

import numpy as np
import pandas as pd
import geopandas as gpd
from spatialdata import SpatialData

from bento._logging import logger
from bento._utils import get_points


def _compute_point_shape_distances(
    points_geo: gpd.GeoSeries,
    shapes_geo: gpd.GeoSeries,
    shape_names: List[str],
    progress: bool = True,
) -> np.ndarray:
    logger.info(f"Computing distances for {len(points_geo)} points")
    if len(points_geo) != len(shape_names):
        raise ValueError("Points and shape_names must have the same length")

    points_by_cell = points_geo.groupby(shape_names, sort=True, observed=True)
    shape_names_ordered = list(points_by_cell.groups.keys())
    grouped_shapes = shapes_geo.loc[shape_names_ordered]
    grouped_points = [points_by_cell.get_group(g) for g in shape_names_ordered]

    point_distances = np.empty(len(points_geo))
    point_indices = np.empty(len(points_geo))
    current_index = 0

    iterator = zip(grouped_shapes, grouped_points)
    if progress:
        from tqdm.auto import tqdm as _tqdm

        iterator = _tqdm(iterator, total=len(grouped_shapes), desc="Computing distances", mininterval=0.5)

    for shape, group_points in iterator:
        shape_boundary = shape.boundary
        distances = group_points.distance(shape_boundary).values
        point_distances[current_index : current_index + len(distances)] = distances
        point_indices[current_index : current_index + len(distances)] = group_points.index.values
        current_index += len(distances)

    all_distances = pd.Series(point_distances, index=point_indices).reindex_like(points_geo)
    return all_distances.values


def _compute_grouped_stats(
    distances: np.ndarray,
    feature_codes: np.ndarray,
    shape_codes: np.ndarray,
    n_features: int,
    n_shapes: int,
) -> Dict[str, np.ndarray]:
    if len(distances) != len(shape_codes) or len(distances) != len(feature_codes):
        raise ValueError("distances, shape_codes, and feature_codes must have the same length")
    if np.any(feature_codes < 0) or np.any(shape_codes < 0):
        raise ValueError("codes contain negative values")
    max_combined_index = n_features * n_shapes - 1
    dtype = np.int32 if max_combined_index < np.iinfo(np.int32).max else np.int64
    feature_codes = feature_codes.astype(dtype)
    shape_codes = shape_codes.astype(dtype)
    combined_indices = feature_codes * n_shapes + shape_codes
    sums = np.bincount(combined_indices, weights=distances, minlength=n_features * n_shapes)
    counts = np.bincount(combined_indices, minlength=n_features * n_shapes)
    sums = sums.reshape(n_features, n_shapes).T
    counts = counts.reshape(n_features, n_shapes).T
    means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    sum_squares = np.bincount(combined_indices, weights=distances**2, minlength=n_features * n_shapes)
    sum_squares = sum_squares.reshape(n_features, n_shapes).T
    variances = np.divide(sum_squares, counts, out=np.full_like(sum_squares, np.nan), where=counts > 0)
    variances -= means**2
    stds = np.sqrt(np.maximum(variances, 0))
    return {"mean": means, "std": stds}


def distance_stats(
    sdata: SpatialData,
    points_key: str = "transcripts",
    shape_key: str = "cell_boundaries",
    feature_key: str = "feature_name",
    table_key: str = "table",
    recompute: bool = True,
    progress: bool = True,
) -> None:
    if table_key not in sdata.tables:
        logger.error(f"Table '{table_key}' not found. Please run bt.io.prep() first to create the base table.")
        return

    existing_layers = ["dist_mean", "dist_std"]
    table = sdata.tables[table_key]
    if not recompute and all(layer in table.layers for layer in existing_layers):
        logger.info("Distance statistics layers already exist. Set recompute=True to recalculate.")
        return

    logger.info("Starting distance statistics computation...")

    points = get_points(sdata, points_key=points_key, astype="pandas")
    points_geo = gpd.GeoSeries(gpd.points_from_xy(points["x"], points["y"]))
    shapes_geo = sdata.shapes[shape_key].geometry

    feature_categories = points[feature_key].astype("category")
    feature_codes = feature_categories.cat.codes.values
    feature_names = feature_categories.cat.categories.tolist()

    shape_categories = points[shape_key].astype("category")
    shape_codes = shape_categories.cat.codes.values
    shape_names = shape_categories.values

    n_features = len(feature_names)
    n_shapes = len(shapes_geo)

    if progress:
        logger.info(f"Processing {n_features} features across {n_shapes} shapes")

    distances = _compute_point_shape_distances(points_geo, shapes_geo, shape_names, progress)
    stats_dict = _compute_grouped_stats(distances, feature_codes, shape_codes, n_features, n_shapes)

    for stat_name, stat_matrix in stats_dict.items():
        layer_name = "dist_mean" if stat_name == "mean" else ("dist_std" if stat_name == "std" else stat_name)
        result_df = pd.DataFrame(stat_matrix, index=table.obs_names, columns=feature_names)
        result_df = result_df.reindex(index=table.obs_names, columns=table.var_names, fill_value=0.0)
        table.layers[layer_name] = result_df.values

    logger.info(f"Distance statistics saved as layers: {list(stats_dict.keys())}")


