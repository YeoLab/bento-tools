"""
Shared utility functions for vectorized point feature computation.

This module contains common functions used across multiple point features
including validation, data loading, and result storage.
"""

from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import spatialdata as sd
from spatialdata import SpatialData
from anndata import AnnData
from scipy.sparse import csr_matrix
import geopandas as gpd

from bento._logging import logger
from bento._utils import get_points


def validate_and_setup(
    sdata: SpatialData, table_key: str, recompute: bool, computation_name: str = "computation"
) -> bool:
    """
    Validate inputs and check if computation should proceed.

    Parameters:
        sdata: SpatialData object
        table_key: Key for storing results
        recompute: Whether to recompute if result already exists
        computation_name: Name of computation for logging

    Returns:
        bool: True if computation should proceed, False if skipping
    """
    if not recompute and table_key in sdata.tables:
        logger.info(f"Skipping, recompute is False. {table_key} exists in sdata.tables")
        return False

    logger.info(f"Starting {computation_name}...")
    return True


def encode_point_shape_data(
    sdata: SpatialData, points_key: str, shape_key: str, feature_key: str, progress: bool = True
) -> Tuple[gpd.GeoSeries, gpd.GeoSeries, List[str], np.ndarray, List[str], np.ndarray, int, int]:
    """
    Load and preprocess points and shapes data for feature computation.

    Parameters:
        sdata: SpatialData object
        points_key: Key for points in sdata.points
        shape_key: Key for shapes in sdata.shapes
        feature_key: Key for features in points
        progress: Whether to show progress information

    Returns:
        points : GeoSeries
            Point geometries
        shapes : GeoSeries
            Shape geometries
        feature_names : List[str]
            Feature names
        feature_codes : np.ndarray
            Feature codes
        shape_names : List[str]
            Shape names
        shape_codes : np.ndarray
            Point group assignments
        n_features : int
            Number of features
        n_shapes : int
            Number of shapes
    """
    # Get points and shapes
    points = get_points(sdata, points_key=points_key, astype="pandas")
    shapes_geo = sdata.shapes[shape_key].geometry

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

    points_geo = gpd.GeoSeries(gpd.points_from_xy(points["x"], points["y"]))

    return points_geo, shapes_geo, feature_names, feature_codes, shape_names, shape_codes, n_features, n_shapes


def store_results(
    sdata: SpatialData,
    result_layers: Dict[str, pd.DataFrame],
    table_key: str = "table",
    region_key: str = None,
) -> None:
    """
    Store computed results in SpatialData object.

    Parameters:
        sdata: SpatialData object to store results in
        result_layers: Dictionary of computed results as DataFrames. Index and columns must match sdata[table_key].obs_names and sdata[table_key].var_names respectively.
        table_key: Key for storing results; default is "table", the counts table created by bt.io.prep()
        region_key: Key for shapes in sdata
    """
    if region_key is None:
        raise ValueError("region_key is required")

    # Create AnnData table
    table = sdata[table_key]
    for layer_name, layer_data in result_layers.items():
        table.layers[layer_name] = layer_data

    table.obs["region"] = region_key
    table.obs["instance"] = table.obs.index

    # Store in sdata
    sdata[table_key] = sd.models.TableModel.parse(table)
    sdata.set_table_annotates_spatialelement(table_key, region_key, region_key="region", instance_key="instance")

    logger.info(f"Saved results to: sdata['{table_key}']")
