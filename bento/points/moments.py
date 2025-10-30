from typing import List

import numpy as np
import pandas as pd
import geopandas as gpd
from spatialdata import SpatialData

from bento._logging import logger
from bento._utils import get_points


def _compute_vectorized_hu_moments(group_points_x: np.ndarray, group_points_y: np.ndarray, centroid_x: float, centroid_y: float) -> list:
    centered_x = group_points_x - centroid_x
    centered_y = group_points_y - centroid_y
    m00 = len(group_points_x)
    if m00 < 2:
        return [np.nan] * 6
    m20 = np.sum(centered_x**2) / m00
    m02 = np.sum(centered_y**2) / m00
    m11 = np.sum(centered_x * centered_y) / m00
    m30 = np.sum(centered_x**3) / m00
    m21 = np.sum((centered_x**2) * centered_y) / m00
    m12 = np.sum(centered_x * (centered_y**2)) / m00
    m03 = np.sum(centered_y**3) / m00
    eta_20 = m20
    eta_02 = m02
    eta_11 = m11
    eta_30 = m30
    eta_21 = m21
    eta_12 = m12
    eta_03 = m03
    hu_1 = eta_20 + eta_02
    hu_2 = (eta_20 - eta_02) ** 2 + 4 * eta_11**2
    hu_3 = (eta_30 - 3 * eta_12) ** 2 + (3 * eta_21 - eta_03) ** 2
    hu_4 = (eta_30 + eta_12) ** 2 + (eta_21 + eta_03) ** 2
    hu_5 = (eta_30 - 3 * eta_12) * (eta_30 + eta_12) * ((eta_30 + eta_12) ** 2 - 3 * (eta_21 + eta_03) ** 2) + (
        3 * eta_21 - eta_03
    ) * (eta_21 + eta_03) * (3 * (eta_30 + eta_12) ** 2 - (eta_21 + eta_03) ** 2)
    hu_6 = (eta_20 - eta_02) * ((eta_30 + eta_12) ** 2 - (eta_21 + eta_03) ** 2) + 4 * eta_11 * (eta_30 + eta_12) * (
        eta_21 + eta_03
    )
    return [hu_1, hu_2, hu_3, hu_4, hu_5, hu_6]


def _compute_moments_distributions(points: pd.DataFrame, shape_centroids: gpd.GeoSeries, shape_key: str, feature_key: str) -> pd.DataFrame:
    grouped = points.groupby([shape_key, feature_key], observed=True)
    if len(grouped) == 0:
        logger.warning("No valid point groups found")
        return pd.DataFrame(columns=["hum_1", "hum_2", "hum_3", "hum_4", "hum_5", "hum_6", shape_key, feature_key])

    shape_centroids_x = shape_centroids.x
    shape_centroids_y = shape_centroids.y

    moments_results = []
    shape_names = []
    feature_names = []

    for (shape_name, feature_name), group in grouped:
        centroid_x = shape_centroids_x[shape_name]
        centroid_y = shape_centroids_y[shape_name]
        if pd.isna(centroid_x) or pd.isna(centroid_y):
            continue
        group_points_x = group["x"].values
        group_points_y = group["y"].values
        if len(group_points_x) < 2:
            continue
        hu_moments = _compute_vectorized_hu_moments(group_points_x, group_points_y, centroid_x, centroid_y)
        moments_results.append(hu_moments)
        shape_names.append(shape_name)
        feature_names.append(feature_name)

    if len(moments_results) == 0:
        logger.warning("No valid moments computed")
        return pd.DataFrame(columns=["hum_1", "hum_2", "hum_3", "hum_4", "hum_5", "hum_6", shape_key, feature_key])

    moments_matrix = np.array(moments_results)
    valid_mask = ~np.isnan(moments_matrix)
    log_moments_matrix = np.zeros_like(moments_matrix)
    abs_moments = np.abs(moments_matrix)
    nonzero_mask = valid_mask & (abs_moments > 1e-12)
    log_moments_matrix[nonzero_mask] = -(np.sign(moments_matrix[nonzero_mask]) * np.log(abs_moments[nonzero_mask]))
    log_moments_matrix[valid_mask & ~nonzero_mask] = 0.0
    log_moments_matrix[~valid_mask] = np.nan

    moments_df = pd.DataFrame(
        log_moments_matrix,
        columns=["hum_1", "hum_2", "hum_3", "hum_4", "hum_5", "hum_6"],
    )
    moments_df[shape_key] = shape_names
    moments_df[feature_key] = feature_names
    return moments_df


def moments(
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

    moment_names = ["hum_1", "hum_2", "hum_3", "hum_4", "hum_5", "hum_6"]
    table = sdata.tables[table_key]
    if not recompute and all(layer in table.layers for layer in moment_names):
        logger.info("Moments layers already exist. Set recompute=True to recalculate.")
        return

    logger.info("Starting moments computation...")

    points = get_points(sdata, points_key=points_key, astype="pandas")
    shapes = sdata.shapes[shape_key]
    shape_centroids = shapes.centroid

    moments_df = _compute_moments_distributions(points, shape_centroids, shape_key, feature_key)
    if len(moments_df) == 0:
        logger.warning("No valid moments computed. Check that shapes have valid centroids.")
        return

    for hum_name in moment_names:
        moment_df = (
            moments_df.loc[:, [hum_name, shape_key, feature_key]]
            .pivot(index=shape_key, columns=feature_key, values=hum_name)
            .reindex(index=table.obs_names, columns=table.var_names, fill_value=np.nan)
        )
        table.layers[hum_name] = moment_df

    logger.info(f"Moments saved to table.layers: {moment_names}")


