from typing import List, Dict
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm.auto import tqdm

from bento._logging import logger


def compute_point_shape_distances(
    points_geo: gpd.GeoSeries,
    shapes_geo: gpd.GeoSeries,
    shape_names: List[str],
    progress: bool = True,
) -> np.ndarray:
    """
    Compute distances from points to their assigned cell boundaries.

    This function uses existing point-cell assignments (no spatial search needed)
    and adds a distance column to the original points DataFrame.

    Note: Always uses single-threaded execution for optimal performance since
    the core distance computation is already vectorized and multiprocessing
    overhead outweighs benefits.

    Parameters:
        points_geo: GeoSeries
            transcript coordinates
        shapes_geo: GeoDataFrame
            shape geometries indexed by shape IDs
        shape_names: List[str]
            List of shape names
        progress: bool
            Whether to show progress bar
    Returns:
        distances: np.ndarray
            Distances from points to their assigned cell boundaries

    Examples:
        >>> # Add distances to existing points
        >>> distances = compute_point_cell_distances(points_df, shapes_gdf)

        >>> # n_workers parameter is ignored but kept for compatibility
        >>> distances = compute_point_cell_distances(
        ...     points_df, shapes_gdf, n_workers=8
        ... )
    """
    logger.info(f"Computing distances for {len(points_geo)} points")

    # Validate inputs
    if len(points_geo) != len(shape_names):
        raise ValueError("Points and shape_names must have the same length")

    # Group points by their cell assignments
    logger.debug("Grouping points by cell assignments...")
    points_by_cell = points_geo.groupby(shape_names, sort=True, observed=True)
    shape_names_ordered = list(points_by_cell.groups.keys())
    grouped_shapes = shapes_geo.loc[shape_names_ordered]
    grouped_points = [points_by_cell.get_group(g) for g in shape_names_ordered]

    # Pre-allocate arrays to store distances and indices
    point_distances = np.empty(len(points_geo))
    point_indices = np.empty(len(points_geo))
    current_index = 0

    iterator = zip(grouped_shapes, grouped_points)
    if progress:
        iterator = tqdm(iterator, total=len(grouped_shapes), desc="Computing distances", mininterval=0.5)

    for shape, group_points in iterator:
        # Distance to polygon boundary, not to the polygon itself
        shape_boundary = shape.boundary
        distances = group_points.distance(shape_boundary).values

        point_distances[current_index : current_index + len(distances)] = distances
        point_indices[current_index : current_index + len(distances)] = group_points.index.values
        current_index += len(distances)

    # Reindex distances to match original point order
    all_distances = pd.Series(point_distances, index=point_indices).reindex_like(points_geo)

    # Log statistics
    valid_distances = ~np.isnan(all_distances)
    n_valid = np.sum(valid_distances)
    if n_valid > 0:
        mean_dist = np.nanmean(all_distances)
        logger.info(
            f"Computed distances for {n_valid}/{len(points_geo)} points (mean: {mean_dist:.2f}, std: {np.nanstd(all_distances):.2f})"
        )
    else:
        logger.warning("No valid distances computed")

    return all_distances.values


def compute_grouped_stats(
    distances: np.ndarray,
    feature_codes: np.ndarray,
    shape_codes: np.ndarray,
    n_features: int,
    n_shapes: int,
) -> Dict[str, np.ndarray]:
    """
    Compute mean and std distances for each feature-shape combination using bincount.

    Parameters:
    -----------
    distances : np.ndarray
        Array of distances from points to shape boundaries
    feature_names : List[str]
        List of feature names
    feature_codes : np.ndarray
        Array of feature codes for each distance measurement
    shape_names : List[str]
        List of shape names
    shape_codes : np.ndarray
        Array of shape codes for each distance measurement
    n_features : int
        Total number of features
    n_shapes : int
        Total number of shapes

    Returns:
    --------
    Dictionary with 'mean' and 'std' arrays of shape (n_shapes, n_features)
    """
    # Validate inputs
    if len(distances) != len(shape_codes) or len(distances) != len(feature_codes):
        raise ValueError("distances, shape_codes, and feature_codes must have the same length")

    # Check for negative indices
    if np.any(feature_codes < 0):
        raise ValueError(f"feature_codes contains negative values: {feature_codes[feature_codes < 0]}")
    if np.any(shape_codes < 0):
        raise ValueError(f"shape_codes contains negative values: {shape_codes[shape_codes < 0]}")

    # Convert to larger integer types to prevent overflow
    # The maximum combined index will be (n_features-1) * n_shapes + (n_shapes-1)
    # which equals n_features * n_shapes - 1
    max_combined_index = n_features * n_shapes - 1

    # Choose appropriate dtype based on the maximum value we'll need
    if max_combined_index < np.iinfo(np.int32).max:
        dtype = np.int32
    else:
        dtype = np.int64

    # Convert arrays to prevent overflow
    feature_codes = feature_codes.astype(dtype)
    shape_codes = shape_codes.astype(dtype)

    # Create combined indices for feature-shape pairs
    combined_indices = feature_codes * n_shapes + shape_codes

    # Final check for negative combined indices (should not happen now)
    if np.any(combined_indices < 0):
        raise ValueError(
            f"combined_indices contains negative values: min={combined_indices.min()}, max={combined_indices.max()}"
        )

    # Use bincount to compute sums and counts
    sums = np.bincount(combined_indices, weights=distances, minlength=n_features * n_shapes)
    counts = np.bincount(combined_indices, minlength=n_features * n_shapes)

    # Reshape to (n_features, n_shapes) and transpose to (n_shapes, n_features)
    sums = sums.reshape(n_features, n_shapes).T
    counts = counts.reshape(n_features, n_shapes).T

    # Compute means
    means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)

    # Compute standard deviations
    # For std, we need sum of squares
    sum_squares = np.bincount(combined_indices, weights=distances**2, minlength=n_features * n_shapes)
    sum_squares = sum_squares.reshape(n_features, n_shapes).T

    # std = sqrt((sum_squares/counts) - (sums/counts)^2)
    variances = np.divide(sum_squares, counts, out=np.full_like(sum_squares, np.nan), where=counts > 0)
    variances -= means**2
    stds = np.sqrt(np.maximum(variances, 0))  # Ensure non-negative

    return {"mean": means, "std": stds}
