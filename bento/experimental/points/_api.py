"""
Public API for vectorized point feature computation.

This module provides the main user-facing functions for computing
point features using efficient vectorized operations.
"""

from typing import Union, List, Dict, Tuple

import numpy as np
import pandas as pd
from spatialdata import SpatialData
from anndata import AnnData
from scipy.sparse import csr_matrix
import spatialdata as sd
import geopandas as gpd

from bento._logging import logger
from .. import shapes as shp
from ._measure import measure
from ._ripley import _ripley, _morans_i
from bento._utils import get_points

# Import vectorized functions from new modules
from ._utils import validate_and_setup, store_results
from ._distance import compute_point_shape_distances, compute_grouped_stats
from ._polarity import compute_polarity
from ._moments import compute_moments_distributions


def distance_stats(
    sdata: SpatialData,
    points_key: str = "transcripts",
    shape_key: str = "cell_boundaries",
    feature_key: str = "feature_name",
    table_key: str = "table",
    recompute: bool = True,
    progress: bool = True,
) -> None:
    """
    Calculate distance statistics (mean and std) from points to shape boundaries.

    This function efficiently computes distance-based features using vectorized
    operations and saves results as layers in the specified table.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object
    points_key : str
        Key for points in sdata.points
    shape_key : str
        Key for shapes in sdata.shapes
    feature_key : str
        Key for features (e.g., genes) in points
    table_key : str
        Key for table in sdata.tables where results will be saved as layers
    recompute : bool
        Whether to recompute if result already exists
    progress : bool
        Whether to show progress bar
    """
    # Check if table exists
    if table_key not in sdata.tables:
        logger.error(f"Table '{table_key}' not found. Please run bt.io.prep() first to create the base table.")
        return

    # Check if layers already exist and recompute is False
    existing_layers = ["dist_mean", "dist_std"]
    table = sdata.tables[table_key]
    if not recompute and all(layer in table.layers for layer in existing_layers):
        logger.info("Distance statistics layers already exist. Set recompute=True to recalculate.")
        return

    logger.info("Starting distance statistics computation...")

    # Step 2: Load and preprocess data
    points = get_points(sdata, points_key=points_key, astype="pandas")
    points_geo = gpd.GeoSeries(gpd.points_from_xy(points["x"], points["y"]))
    shapes = sdata.shapes[shape_key]
    shapes_geo = shapes.geometry

    # Convert feature names to codes for efficient bincount
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

    # Step 3: Compute distance distributions
    distances = compute_point_shape_distances(
        points_geo=points_geo,
        shapes_geo=shapes_geo,
        shape_names=shape_names,
        progress=progress,
    )

    # Step 4: Compute statistics using vectorized operations
    stats_dict = compute_grouped_stats(distances, feature_codes, shape_codes, n_features, n_shapes)

    # Step 5: Format results and save as layers
    for stat_name, stat_matrix in stats_dict.items():
        if stat_name == "mean":
            layer_name = "dist_mean"
        elif stat_name == "std":
            layer_name = "dist_std"
        else:
            layer_name = stat_name

        # Convert to DataFrame with proper indexing
        result_df = pd.DataFrame(stat_matrix, index=table.obs_names, columns=feature_names)

        # Reindex to match table structure
        result_df = result_df.reindex(index=table.obs_names, columns=table.var_names, fill_value=0.0)

        # Save as layer
        table.layers[layer_name] = result_df.values

    logger.info(f"Distance statistics saved as layers: {list(stats_dict.keys())}")


def polarity(
    sdata: SpatialData,
    points_key: str = "transcripts",
    shape_key: str = "cell_boundaries",
    feature_key: str = "feature_name",
    table_key: str = "table",
    recompute: bool = True,
    progress: bool = True,
) -> None:
    """
    Calculate polarity of points within shapes using vectorized computation.

    Polarity measures the displacement of the center of mass from the shape centroid,
    normalized by the shape radius. Results are saved as a layer in the specified table.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object
    points_key : str
        Key for points in sdata.points
    shape_key : str
        Key for shapes in sdata.shapes
    feature_key : str
        Key for features (e.g., genes) in points
    table_key : str
        Key for table in sdata.tables where results will be saved as layers
    recompute : bool
        Whether to recompute if result already exists
    progress : bool
        Whether to show progress bar
    """
    # Check if table exists
    if table_key not in sdata.tables:
        logger.error(f"Table '{table_key}' not found. Please run bt.io.prep() first to create the base table.")
        return

    # Check if layer already exists and recompute is False
    table = sdata.tables[table_key]
    if not recompute and "polarity" in table.layers:
        logger.info("Polarity layer already exists. Set recompute=True to recalculate.")
        return

    logger.info("Starting polarity computation...")

    # Load and preprocess data
    points = get_points(sdata, points_key=points_key, astype="geopandas")
    shapes = sdata.shapes[shape_key]

    # Check and compute radius if needed
    if "radius" not in shapes.columns or recompute:
        logger.info("Computing shape radii...")
        shp.radius(
            sdata,
            shape_key=shape_key,
            recompute=recompute,
            progress=progress,
            leave=False,
        )

    # Extract shape properties for polarity computation
    shape_radii = shapes["radius"]

    # Compute polarity distributions
    polarity_values = compute_polarity(
        points=points, shapes=shapes, shape_radii=shape_radii, shape_key=shape_key, feature_key=feature_key
    )

    # Create result matrix
    feature_names = points[feature_key].unique().tolist()
    polarity_df = polarity_values.pivot(index=shape_key, columns=feature_key, values="polarity")

    # Reindex to match table structure
    polarity_df = polarity_df.reindex(index=table.obs_names, columns=table.var_names, fill_value=0.0)

    # Save as layer
    table.layers["polarity"] = polarity_df.values

    logger.info("Polarity saved as layer: polarity")


def moments(
    sdata: SpatialData,
    points_key: str = "transcripts",
    shape_key: str = "cell_boundaries",
    feature_key: str = "feature_name",
    table_key: str = "table",
    recompute: bool = True,
    progress: bool = True,
) -> None:
    """
    Calculate Hu moments of points within shapes using vectorized computation.

    Hu moments are translation, scale, and rotation invariant shape descriptors
    that characterize the spatial distribution of points within each shape.
    Results are saved as layers in the specified table.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object
    points_key : str
        Key for points in sdata.points
    shape_key : str
        Key for shapes in sdata.shapes
    feature_key : str
        Key for features (e.g., genes) in points
    table_key : str
        Key for table in sdata.tables where results will be saved as layers
    recompute : bool
        Whether to recompute if result already exists
    progress : bool
        Whether to show progress bar
    """
    # Check if table exists
    if table_key not in sdata.tables:
        logger.error(f"Table '{table_key}' not found. Please run bt.io.prep() first to create the base table.")
        return

    # Check if layers already exist and recompute is False
    moment_names = ["hum_1", "hum_2", "hum_3", "hum_4", "hum_5", "hum_6"]
    table = sdata.tables[table_key]
    if not recompute and all(layer in table.layers for layer in moment_names):
        logger.info("Moments layers already exist. Set recompute=True to recalculate.")
        return

    logger.info("Starting moments computation...")

    # Get points as DataFrame
    points = get_points(sdata, points_key=points_key, astype="pandas")
    # Get shapes as DataFrame
    shapes = sdata.shapes[shape_key]
    shape_centroids = shapes.centroid

    # Step 4: Compute moments distributions
    moments_df = compute_moments_distributions(points, shape_centroids, shape_key, feature_key)

    if len(moments_df) == 0:
        logger.warning("No valid moments computed. Check that shapes have valid centroids.")
        return

    logger.debug(f"Computed moments for {len(moments_df)} shape-feature combinations")

    # Step 5: Create and save layers for each Hu moment
    for hum_name in moment_names:
        # Convert to DataFrame and reindex to match table structure
        moment_df = (
            moments_df.loc[:, [hum_name, shape_key, feature_key]]
            .pivot(index=shape_key, columns=feature_key, values=hum_name)
            .reindex(index=table.obs_names, columns=table.var_names, fill_value=np.nan)
        )

        # Save as layer
        table.layers[hum_name] = moment_df

    logger.info(f"Moments saved to table.layers: {moment_names}")


def density(
    sdata: SpatialData,
    points_key: str = "transcripts",
    feature_key: str = "feature_name",
    shape_key: str = "cell_boundaries",
    table_key: str = "table",
    recompute: bool = True,
    progress: bool = True,
    leave: bool = True,
) -> None:
    """
    Calculate point density (points per area) within shapes using existing count table.

    This function efficiently computes density by dividing pre-computed point counts
    by shape areas. Results are saved as a layer in the specified table.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object
    points_key : str
        Not used in this function but kept for compatibility
    shape_key : str
        Key for shapes in sdata.shapes
    table_key : str
        Key for table in sdata.tables where results will be saved as layers
    recompute : bool
        Whether to recompute if result already exists
    progress : bool
        Whether to show progress bar
    leave : bool
        Whether to leave progress bar (ignored in new implementation)
    """
    # Check if table exists
    if table_key not in sdata.tables:
        logger.error(f"Table '{table_key}' not found. Please run bt.io.prep() first to create the base table.")
        return

    # Check if layer already exists and recompute is False
    table = sdata.tables[table_key]
    if not recompute and "density" in table.layers:
        logger.info("Density layer already exists. Set recompute=True to recalculate.")
        return

    logger.info("Starting density computation...")

    # Step 3: Get count matrix from existing table
    count_matrix = table.X.toarray() if hasattr(table.X, "toarray") else table.X

    # Verify the table matches our shape_key
    if table.obs["region"].iloc[0] != shape_key:
        logger.error(f"Table region '{table.obs['region'].iloc[0]}' doesn't match shape_key '{shape_key}'")
        return

    shapes = sdata.shapes[shape_key]
    n_shapes, n_features = count_matrix.shape

    if progress:
        logger.info(f"Computing density for {n_features} features across {n_shapes} shapes")

    # Step 4: Ensure cell areas are computed
    area_col = f"{shape_key}_area"
    if area_col not in shapes.columns or recompute:
        shp.area(
            sdata=sdata,
            shape_key=shape_key,
            result_key=area_col,
            recompute=recompute,
            progress=progress,
            leave=leave,
        )
        logger.info(f"Computed areas for {shapes.shape[0]} shapes")

    # Step 5: Get areas aligned with table
    areas = shapes.loc[table.obs_names, area_col].values.reshape(-1, 1)

    # Check for invalid areas
    invalid_areas = (areas <= 0) | np.isnan(areas)
    if np.any(invalid_areas):
        logger.warning(f"Found {np.sum(invalid_areas)} shapes with invalid areas (<=0 or NaN)")

    # Step 6: Compute density = counts / area
    # Broadcast division: (n_shapes, n_features) / (n_shapes, 1) = (n_shapes, n_features)
    density_matrix = np.divide(
        count_matrix, areas, out=np.full_like(count_matrix, np.nan, dtype=float), where=~invalid_areas
    )

    # Step 7: Save as layer
    table.layers["density"] = density_matrix

    logger.info(f"Density saved as layer. Non-zero densities: {np.sum(density_matrix > 0)}")


def morans_i(
    sdata: SpatialData,
    points_key: str = "transcripts",
    shape_key: str = "cell_boundaries",
    feature_key: str = "feature_name",
    table_key: str = "table",
    n_jobs: int = 1,
    recompute: bool = True,
    progress: bool = True,
    leave: bool = True,
) -> None:
    """
    Calculate Moran's I spatial autocorrelation of points within shapes.

    Results are saved as a layer in the specified table.
    """
    # Check if table exists
    if table_key not in sdata.tables:
        logger.error(f"Table '{table_key}' not found. Please run bt.io.prep() first to create the base table.")
        return

    # Check if layer already exists and recompute is False
    table = sdata.tables[table_key]
    if not recompute and "morans_i" in table.layers:
        logger.info("Moran's I layer already exists. Set recompute=True to recalculate.")
        return

    # Use existing measure function but with modified result handling
    # We'll need to temporarily redirect output to capture results
    temp_table_key = f"temp_{table_key}_morans"

    measure(
        sdata=sdata,
        func=_morans_i,
        points_key=points_key,
        shape_key=shape_key,
        feature_key=feature_key,
        result_key=temp_table_key,
        n_jobs=n_jobs,
        recompute=recompute,
        progress=progress,
        leave=leave,
    )

    # Move result from temporary table to layer in main table
    if temp_table_key in sdata.tables:
        temp_table = sdata.tables[temp_table_key]
        # Align with main table structure
        result_matrix = temp_table.X.toarray() if hasattr(temp_table.X, "toarray") else temp_table.X

        # Create aligned matrix
        aligned_matrix = np.full((len(table.obs_names), len(table.var_names)), np.nan)

        # Map indices
        obs_mapping = {name: i for i, name in enumerate(table.obs_names)}
        var_mapping = {name: i for i, name in enumerate(table.var_names)}

        for i, obs_name in enumerate(temp_table.obs_names):
            if obs_name in obs_mapping:
                for j, var_name in enumerate(temp_table.var_names):
                    if var_name in var_mapping:
                        aligned_matrix[obs_mapping[obs_name], var_mapping[var_name]] = result_matrix[i, j]

        # Save as layer
        table.layers["morans_i"] = aligned_matrix

        # Clean up temporary table
        del sdata.tables[temp_table_key]

        logger.info("Moran's I saved as layer: morans_i")


def ripley(
    sdata: SpatialData,
    points_key: str = "transcripts",
    shape_key: str = "cell_boundaries",
    feature_key: str = "feature_name",
    table_key: str = "table",
    n_jobs: int = 1,
    recompute: bool = True,
    progress: bool = True,
    leave: bool = True,
) -> None:
    """
    Calculate Ripley's K and L statistics for points within shapes.

    Results are saved as layers in the specified table.
    """
    # Check if table exists
    if table_key not in sdata.tables:
        logger.error(f"Table '{table_key}' not found. Please run bt.io.prep() first to create the base table.")
        return

    # Check if layers already exist and recompute is False
    ripley_layers = ["ripley_k", "ripley_l"]  # Assuming these are the layer names from _ripley function
    table = sdata.tables[table_key]
    if not recompute and any(layer in table.layers for layer in ripley_layers):
        logger.info("Ripley layers already exist. Set recompute=True to recalculate.")
        return

    # Use existing measure function but with modified result handling
    # We'll need to temporarily redirect output to capture results
    temp_table_key = f"temp_{table_key}_ripley"

    measure(
        sdata=sdata,
        func=_ripley,
        points_key=points_key,
        shape_key=shape_key,
        feature_key=feature_key,
        result_key=temp_table_key,
        n_jobs=n_jobs,
        recompute=recompute,
        progress=progress,
        leave=leave,
    )

    # Move results from temporary table to layers in main table
    if temp_table_key in sdata.tables:
        temp_table = sdata.tables[temp_table_key]

        # Handle multiple layers if they exist in the temporary table
        for layer_name, layer_data in temp_table.layers.items():
            # Align with main table structure
            if hasattr(layer_data, "toarray"):
                result_matrix = layer_data.toarray()
            else:
                result_matrix = layer_data

            # Create aligned matrix
            aligned_matrix = np.full((len(table.obs_names), len(table.var_names)), np.nan)

            # Map indices
            obs_mapping = {name: i for i, name in enumerate(table.obs_names)}
            var_mapping = {name: i for i, name in enumerate(table.var_names)}

            for i, obs_name in enumerate(temp_table.obs_names):
                if obs_name in obs_mapping:
                    for j, var_name in enumerate(temp_table.var_names):
                        if var_name in var_mapping:
                            aligned_matrix[obs_mapping[obs_name], var_mapping[var_name]] = result_matrix[i, j]

            # Save as layer with ripley prefix if not already prefixed
            final_layer_name = layer_name if layer_name.startswith("ripley_") else f"ripley_{layer_name}"
            table.layers[final_layer_name] = aligned_matrix

        # If no layers in temp table, use the main data
        if len(temp_table.layers) == 0:
            result_matrix = temp_table.X.toarray() if hasattr(temp_table.X, "toarray") else temp_table.X

            # Create aligned matrix
            aligned_matrix = np.full((len(table.obs_names), len(table.var_names)), np.nan)

            # Map indices
            obs_mapping = {name: i for i, name in enumerate(table.obs_names)}
            var_mapping = {name: i for i, name in enumerate(table.var_names)}

            for i, obs_name in enumerate(temp_table.obs_names):
                if obs_name in obs_mapping:
                    for j, var_name in enumerate(temp_table.var_names):
                        if var_name in var_mapping:
                            aligned_matrix[obs_mapping[obs_name], var_mapping[var_name]] = result_matrix[i, j]

            # Save as layer
            table.layers["ripley"] = aligned_matrix

        # Clean up temporary table
        del sdata.tables[temp_table_key]

        logger.info(f"Ripley statistics saved as layers in table: {table_key}")
