"""
Vectorized moments computation for point features.

This module contains functions for computing Hu moment invariants
using efficient vectorized operations.
"""

from typing import Tuple
import numpy as np
import pandas as pd
import geopandas as gpd

from bento._logging import logger


def compute_vectorized_hu_moments(
    group_points_x: np.ndarray, group_points_y: np.ndarray, centroid_x: float, centroid_y: float
) -> list:
    """
    Compute Hu moments for a group of points relative to a centroid.

    This is a vectorized version of the Hu moment computation that can be applied
    to multiple point groups simultaneously using numpy operations.

    Parameters
    ----------
    group_points_x : np.ndarray
        X coordinates of points in the group
    group_points_y : np.ndarray
        Y coordinates of points in the group
    centroid_x : float
        X coordinate of the shape centroid
    centroid_y : float
        Y coordinate of the shape centroid

    Returns
    -------
    list
        List of 6 Hu moment invariants
    """
    # Center points relative to shape centroid
    centered_x = group_points_x - centroid_x
    centered_y = group_points_y - centroid_y

    # Compute central moments
    m00 = len(group_points_x)  # zeroth moment (count)
    if m00 < 2:  # Need at least 2 points for meaningful moments
        return [np.nan] * 6

    # Raw central moments
    m20 = np.sum(centered_x**2) / m00
    m02 = np.sum(centered_y**2) / m00
    m11 = np.sum(centered_x * centered_y) / m00
    m30 = np.sum(centered_x**3) / m00
    m21 = np.sum((centered_x**2) * centered_y) / m00
    m12 = np.sum(centered_x * (centered_y**2)) / m00
    m03 = np.sum(centered_y**3) / m00

    # Normalized central moments (eta)
    # For translation and scale invariance
    eta_20 = m20
    eta_02 = m02
    eta_11 = m11
    eta_30 = m30
    eta_21 = m21
    eta_12 = m12
    eta_03 = m03

    # Compute Hu moment invariants
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

    # Return raw moments without log scaling
    return [hu_1, hu_2, hu_3, hu_4, hu_5, hu_6]


def compute_moments_distributions(
    points: pd.DataFrame,
    shape_centroids: gpd.GeoSeries,
    shape_key: str,
    feature_key: str,
) -> pd.DataFrame:
    """
    Compute Hu moments for each feature-shape combination using vectorized operations.

    Parameters:
        points: Points dataframe with coordinates, shape and feature columns
        shapes: Shapes dataframe with centroids
        feature_codes: Array of feature codes for points
        shape_key: Key for shapes

    Returns:
        moments_df : pd.DataFrame
            Shape (n_valid_groups, 6) for the 6 Hu moments
    """

    # Group by (shape_idx, feature_code) - we need individual points for moment computation
    logger.debug("Grouping points for moments computation...")
    grouped = points.groupby([shape_key, feature_key], observed=True)

    if len(grouped) == 0:
        logger.warning("No valid point groups found")
        return np.array([]).reshape(0, 6), np.array([]), np.array([])

    logger.debug(f"Found {len(grouped)} shape-feature combinations with points")

    # Get shape centroids for reference
    shape_centroids_x = shape_centroids.x
    shape_centroids_y = shape_centroids.y

    # Process each group and compute Hu moments
    moments_results = []
    shape_names = []
    feature_names = []

    for (shape_name, feature_name), group in grouped:
        # Get shape centroid
        centroid_x = shape_centroids_x[shape_name]
        centroid_y = shape_centroids_y[shape_name]

        # Skip if centroid is invalid
        if pd.isna(centroid_x) or pd.isna(centroid_y):
            continue

        # Get points for this group
        group_points_x = group["x"].values
        group_points_y = group["y"].values

        if len(group_points_x) < 2:  # Need at least 2 points for meaningful moments
            continue

        # Compute Hu moments for this group
        hu_moments = compute_vectorized_hu_moments(group_points_x, group_points_y, centroid_x, centroid_y)

        # Store results
        moments_results.append(hu_moments)  # (6,)
        shape_names.append(shape_name)
        feature_names.append(feature_name)

    if len(moments_results) == 0:
        logger.warning("No valid moments computed")
        return np.array([]).reshape(0, 6), np.array([]), np.array([])

    moments_matrix = np.array(moments_results)  # (n_valid_groups, 6)

    # Apply log scaling for numerical stability (vectorized)
    # Only process non-NaN values
    valid_mask = ~np.isnan(moments_matrix)
    log_moments_matrix = np.zeros_like(moments_matrix)

    # Apply log scaling where moments are valid and non-zero
    abs_moments = np.abs(moments_matrix)
    nonzero_mask = valid_mask & (abs_moments > 1e-12)

    log_moments_matrix[nonzero_mask] = -(np.sign(moments_matrix[nonzero_mask]) * np.log(abs_moments[nonzero_mask]))
    log_moments_matrix[valid_mask & ~nonzero_mask] = 0.0
    log_moments_matrix[~valid_mask] = np.nan

    logger.debug(f"Computed Hu moments for {len(moments_results)} valid groups")
    logger.debug(f"Moments matrix shape: {log_moments_matrix.shape}")

    moments_df = pd.DataFrame(
        log_moments_matrix,
        columns=["hum_1", "hum_2", "hum_3", "hum_4", "hum_5", "hum_6"],
    )
    moments_df[shape_key] = shape_names
    moments_df[feature_key] = feature_names

    return moments_df
