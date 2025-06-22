from functools import wraps
from typing import Callable, Union, List, Dict, Any, Optional
import warnings

import polars as pl
import numpy as np
import numba
from numba import njit, prange, float64, int64, boolean
from shapely.geometry import MultiPolygon, Polygon
from spatialdata import SpatialData
from tqdm.auto import tqdm
import emoji
import pandas as pd

from bento._utils import get_points
from bento._logging import logger

# Numba configuration for better performance
numba.set_num_threads(4)  # Adjust based on system


@njit(parallel=True)
def _fast_groupby_indices(group_col: np.ndarray, unique_groups: np.ndarray) -> Dict[int, np.ndarray]:
    """Fast groupby indices computation using Numba."""
    indices = {}
    for i in prange(len(unique_groups)):
        group_val = unique_groups[i]
        mask = group_col == group_val
        indices[i] = np.where(mask)[0]
    return indices


@njit(parallel=True)
def _batch_process_shapes(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    group_ids: np.ndarray,
    feature_ids: np.ndarray,
    shape_measurements: np.ndarray,
    batch_size: int,
    n_shapes: int,
) -> tuple:
    """Process shapes in batches with Numba for memory efficiency."""

    # Pre-allocate results arrays
    max_results = batch_size * 10  # Estimate max results per batch
    result_shape_ids = np.full(max_results, -1, dtype=int64)
    result_feature_ids = np.full(max_results, -1, dtype=int64)
    result_values = np.full(max_results, np.nan, dtype=float64)
    result_count = 0

    for i in prange(0, n_shapes, batch_size):
        end_idx = min(i + batch_size, n_shapes)

        for shape_idx in range(i, end_idx):
            # Get points for this shape
            shape_mask = group_ids == shape_idx
            if not np.any(shape_mask):
                continue

            shape_x = x_coords[shape_mask]
            shape_y = y_coords[shape_mask]
            shape_features = feature_ids[shape_mask]

            # Process each feature for this shape
            unique_features = np.unique(shape_features)
            for feature_id in unique_features:
                feature_mask = shape_features == feature_id
                feature_x = shape_x[feature_mask]
                feature_y = shape_y[feature_mask]

                # Calculate metrics (placeholder - will be replaced with actual functions)
                if len(feature_x) > 0:
                    result_shape_ids[result_count] = shape_idx
                    result_feature_ids[result_count] = feature_id
                    result_values[result_count] = np.mean(feature_x)  # Placeholder metric
                    result_count += 1

    return result_shape_ids[:result_count], result_feature_ids[:result_count], result_values[:result_count]


class PolarsMeasureEngine:
    """Optimized measurement engine using Polars and Numba."""

    def __init__(self, n_jobs: int = 1, batch_size: int = 1000):
        self.n_jobs = n_jobs
        self.batch_size = batch_size

    def _convert_to_polars(self, sdata: SpatialData, points_key: str, shape_key: str, feature_key: str) -> pl.DataFrame:
        """Convert SpatialData points to Polars DataFrame efficiently."""

        # Get points as Polars DataFrame directly if possible
        try:
            points = get_points(sdata, points_key=points_key, astype="polars")
        except:
            # Fallback to pandas then convert
            points_pd = get_points(sdata, points_key=points_key, astype="pandas")
            points = pl.from_pandas(points_pd)

        # Select only needed columns
        points = points.select(["x", "y", shape_key, feature_key])

        # Convert shape_key to categorical for memory efficiency
        points = points.with_columns([pl.col(shape_key).cast(pl.Categorical), pl.col(feature_key).cast(pl.Categorical)])

        return points

    def _prepare_shape_data(
        self, sdata: SpatialData, shape_key: str, shape_names: List[str], shape_measure_keys: List[str] = None
    ) -> Dict:
        """Prepare shape data for efficient access."""

        shape_data = {}

        if shape_measure_keys:
            shape_df = sdata.shapes[shape_key].loc[shape_names, ["geometry", *shape_measure_keys]]
        else:
            shape_df = sdata.shapes[shape_key].loc[shape_names, ["geometry"]]

        # Convert to more efficient format
        for i, shape_name in enumerate(shape_names):
            shape_data[shape_name] = {
                "geometry": shape_df.iloc[i]["geometry"],
                "measurements": shape_df.iloc[i][1:].to_dict() if shape_measure_keys else {},
            }

        return shape_data

    def _process_batch_polars(
        self,
        batch_points: pl.DataFrame,
        shape_data: Dict,
        func: Callable,
        shape_key: str,
        feature_key: str,
        shape_measure_keys: List[str] = None,
    ) -> pl.DataFrame:
        """Process a batch of points using Polars expressions."""

        # Use Polars groupby operations for better performance
        results = []

        # Group by shape first (matching original implementation)
        grouped_by_shape = batch_points.group_by(shape_key, maintain_order=False)

        # Apply function to each shape group
        for shape_name, shape_points in grouped_by_shape:
            # Extract shape name from tuple (Polars groupby returns tuple)
            if isinstance(shape_name, tuple):
                shape_name = shape_name[0]

            if shape_name not in shape_data:
                continue

            shape_info = shape_data[shape_name]
            shape = shape_info["geometry"]
            measurements = shape_info["measurements"]

            # Convert shape points to pandas for compatibility with existing functions
            shape_points_pandas = shape_points.to_pandas()

            # Apply the wrapped measurement function (which handles feature grouping internally)
            try:
                # Pass feature_key as gene_key parameter to the wrapped function
                result = func(shape_points_pandas, gene_key=feature_key, shape=shape, **measurements)

                # Handle the result format from _enable_gene_groups wrapper
                if isinstance(result, pd.Series):
                    # Result is a Series with feature names as index and metric values
                    for feature_name, feature_result in result.items():
                        if isinstance(feature_result, dict):
                            for metric_name, metric_value in feature_result.items():
                                results.append(
                                    {
                                        "shape": shape_name,
                                        "feature": feature_name,
                                        "metric": metric_name,
                                        "value": metric_value,
                                    }
                                )
                        else:
                            # Single metric value
                            results.append(
                                {
                                    "shape": shape_name,
                                    "feature": feature_name,
                                    "metric": "value",
                                    "value": feature_result,
                                }
                            )
                elif isinstance(result, dict):
                    # Direct dict result (no feature grouping)
                    for metric_name, metric_value in result.items():
                        results.append(
                            {"shape": shape_name, "feature": "all", "metric": metric_name, "value": metric_value}
                        )
            except Exception as e:
                logger.debug(f"Error processing shape {shape_name}: {e}")
                continue

        return pl.DataFrame(results) if results else pl.DataFrame()

    def _process_batch_polars_ordered(
        self,
        batch_points: pl.DataFrame,
        shape_data: Dict,
        func: Callable,
        shape_key: str,
        feature_key: str,
        shape_names: List[str],
        shape_measure_keys: List[str] = None,
    ) -> pl.DataFrame:
        """Process a batch of points using Polars expressions in the specified shape order."""

        # Use Polars groupby operations for better performance
        results = []

        # Process shapes in the specified order
        for shape_name in shape_names:
            if shape_name not in shape_data:
                continue

            # Filter points for this specific shape
            shape_points = batch_points.filter(pl.col(shape_key) == shape_name)

            if len(shape_points) == 0:
                continue

            shape_info = shape_data[shape_name]
            shape = shape_info["geometry"]
            measurements = shape_info["measurements"]

            # Convert shape points to pandas for compatibility with existing functions
            shape_points_pandas = shape_points.to_pandas()

            # Apply the wrapped measurement function (which handles feature grouping internally)
            try:
                # Pass feature_key as gene_key parameter to the wrapped function
                result = func(shape_points_pandas, gene_key=feature_key, shape=shape, **measurements)

                # Handle the result format from _enable_gene_groups wrapper
                if isinstance(result, pd.Series):
                    # Result is a Series with feature names as index and metric values
                    for feature_name, feature_result in result.items():
                        if isinstance(feature_result, dict):
                            for metric_name, metric_value in feature_result.items():
                                results.append(
                                    {
                                        "shape": shape_name,
                                        "feature": feature_name,
                                        "metric": metric_name,
                                        "value": metric_value,
                                    }
                                )
                        else:
                            # Single metric value
                            results.append(
                                {
                                    "shape": shape_name,
                                    "feature": feature_name,
                                    "metric": "value",
                                    "value": feature_result,
                                }
                            )
                elif isinstance(result, dict):
                    # Direct dict result (no feature grouping)
                    for metric_name, metric_value in result.items():
                        results.append(
                            {"shape": shape_name, "feature": "all", "metric": metric_name, "value": metric_value}
                        )
            except Exception as e:
                logger.debug(f"Error processing shape {shape_name}: {e}")
                continue

        return pl.DataFrame(results) if results else pl.DataFrame()

    def measure(
        self,
        sdata: SpatialData,
        func: Callable,
        points_key: str,
        shape_key: str,
        feature_key: str,
        shape_measure_keys: List[str] = None,
        result_key: str = None,
        recompute: bool = True,
        progress: bool = True,
    ) -> None:
        """Optimized measure function using Polars and Numba."""

        if not recompute and result_key in sdata.tables:
            logger.info(f"Skipping, recompute is False. {result_key} exists in sdata.tables")
            return

        # Apply the _enable_gene_groups wrapper to the function (matching original)
        from ._measure import _enable_gene_groups

        point_feature_func = _enable_gene_groups(func)

        # Convert to Polars DataFrame
        logger.info("Converting data to Polars format...")
        points = self._convert_to_polars(sdata, points_key, shape_key, feature_key)

        # Get unique shapes and features
        shape_names = points[shape_key].unique().to_list()
        feature_names = points[feature_key].unique().to_list()

        logger.info(f"Processing {len(shape_names)} shapes with {len(feature_names)} features")

        # Prepare shape data
        shape_data = self._prepare_shape_data(sdata, shape_key, shape_names, shape_measure_keys)

        # Process in batches
        all_results = []
        total_batches = (len(shape_names) + self.batch_size - 1) // self.batch_size

        if progress:
            pbar = tqdm(total=total_batches, desc=emoji.emojize(":hourglass_not_done:"))

        for i in range(0, len(shape_names), self.batch_size):
            batch_shape_names = shape_names[i : i + self.batch_size]

            # Filter points for this batch of shapes
            batch_points = points.filter(pl.col(shape_key).is_in(batch_shape_names))

            # Process batch in the correct order
            batch_results = self._process_batch_polars_ordered(
                batch_points,
                shape_data,
                point_feature_func,
                shape_key,
                feature_key,
                batch_shape_names,
                shape_measure_keys,
            )

            if not batch_results.is_empty():
                all_results.append(batch_results)

            if progress:
                pbar.update(1)

        if progress:
            pbar.close()
            pbar.set_description(f"{emoji.emojize(':bento:')} Done.")

        # Combine results
        if all_results:
            combined_results = pl.concat(all_results)
            self._save_results(sdata, combined_results, shape_names, feature_names, result_key)
        else:
            logger.warning("No results generated")

    def _save_results(
        self,
        sdata: SpatialData,
        results: pl.DataFrame,
        shape_names: List[str],
        feature_names: List[str],
        result_key: str,
    ):
        """Save results to SpatialData object efficiently."""

        # Pivot results to wide format
        result_layers = {}

        for metric in results["metric"].unique():
            metric_results = results.filter(pl.col("metric") == metric)

            # Pivot to wide format
            pivot_df = metric_results.pivot(
                values="value", index="shape", columns="feature", aggregate_function="first"
            )

            # Reindex to include all shapes and features
            pivot_df = pivot_df.with_columns([pl.col("shape").cast(pl.Categorical)])

            # Convert to pandas for compatibility with AnnData
            result_layers[metric] = pivot_df.to_pandas().set_index("shape")

        # Create or update AnnData object
        if result_key in sdata:
            sdata[result_key].layers = result_layers
        else:
            # Create new AnnData object
            from anndata import AnnData
            from scipy.sparse import csr_matrix
            import spatialdata as sd

            # Create empty matrix
            empty_x = csr_matrix((len(shape_names), len(feature_names)))

            table = AnnData(X=empty_x, layers=result_layers)
            table.obs_names = shape_names
            table.var_names = feature_names
            table.obs["region"] = "cell_boundaries"
            table.obs["instance"] = table.obs.index

            sdata[result_key] = sd.models.TableModel.parse(table)
            sdata.set_table_annotates_spatialelement(
                result_key, "cell_boundaries", region_key="region", instance_key="instance"
            )

        logger.info(f"Saved results to: sdata['{result_key}']")


# Optimized wrapper functions
def measure_optimized(
    sdata: SpatialData,
    func: Callable,
    points_key: str = "transcripts",
    shape_key: str = "cell_boundaries",
    feature_key: str = "feature_name",
    shape_measure_keys: List[str] = None,
    result_key: str = None,
    n_jobs: int = 1,
    batch_size: int = 1000,
    recompute: bool = True,
    progress: bool = True,
) -> None:
    """Optimized version of measure function using Polars + Numba."""

    engine = PolarsMeasureEngine(n_jobs=n_jobs, batch_size=batch_size)
    engine.measure(
        sdata=sdata,
        func=func,
        points_key=points_key,
        shape_key=shape_key,
        feature_key=feature_key,
        shape_measure_keys=shape_measure_keys,
        result_key=result_key,
        recompute=recompute,
        progress=progress,
    )


# Optimized versions of existing functions
def distance_optimized(
    sdata: SpatialData,
    points_key: str = "transcripts",
    shape_key: str = "cell_boundaries",
    feature_key: str = "feature_name",
    result_key: str = "tx_distance_optimized",
    n_jobs: int = 1,
    batch_size: int = 1000,
    recompute: bool = True,
    progress: bool = True,
) -> None:
    """Optimized distance calculation using Polars + Numba."""

    from .. import shapes as shp
    from ._distance import _distances

    # Pre-compute shape measurements
    shp.radius(
        sdata,
        shape_key=shape_key,
        n_jobs=n_jobs,
        recompute=recompute,
        progress=False,
        leave=False,
    )

    measure_optimized(
        sdata=sdata,
        func=_distances,
        points_key=points_key,
        shape_key=shape_key,
        shape_measure_keys=["radius"],
        feature_key=feature_key,
        result_key=result_key,
        n_jobs=n_jobs,
        batch_size=batch_size,
        recompute=recompute,
        progress=progress,
    )


def polarity_optimized(
    sdata: SpatialData,
    points_key: str = "transcripts",
    shape_key: str = "cell_boundaries",
    feature_key: str = "feature_name",
    result_key: str = "tx_polarity_optimized",
    n_jobs: int = 1,
    batch_size: int = 1000,
    recompute: bool = True,
    progress: bool = True,
) -> None:
    """Optimized polarity calculation using Polars + Numba."""

    from .. import shapes as shp
    from ._polarity import _polarity

    # Pre-compute shape measurements
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

    measure_optimized(
        sdata=sdata,
        func=_polarity,
        points_key=points_key,
        shape_key=shape_key,
        shape_measure_keys=["radius", "x", "y"],
        feature_key=feature_key,
        result_key=result_key,
        n_jobs=n_jobs,
        batch_size=batch_size,
        recompute=recompute,
        progress=progress,
    )


def density_optimized(
    sdata: SpatialData,
    points_key: str = "transcripts",
    shape_key: str = "cell_boundaries",
    feature_key: str = "feature_name",
    result_key: str = "tx_density_optimized",
    n_jobs: int = 1,
    batch_size: int = 1000,
    recompute: bool = True,
    progress: bool = True,
) -> None:
    """Optimized density calculation using Polars + Numba."""

    from ._density import _density

    measure_optimized(
        sdata=sdata,
        func=_density,
        points_key=points_key,
        shape_key=shape_key,
        feature_key=feature_key,
        result_key=result_key,
        n_jobs=n_jobs,
        batch_size=batch_size,
        recompute=recompute,
        progress=progress,
    )


def moments_optimized(
    sdata: SpatialData,
    points_key: str = "transcripts",
    shape_key: str = "cell_boundaries",
    feature_key: str = "feature_name",
    result_key: str = "tx_moments_optimized",
    n_jobs: int = 1,
    batch_size: int = 1000,
    recompute: bool = True,
    progress: bool = True,
) -> None:
    """Optimized moments calculation using Polars + Numba."""

    from .. import shapes as shp
    from ._moments import _hu_moments

    # Pre-compute shape measurements
    shp.centroid(
        sdata,
        shape_key=shape_key,
        n_jobs=n_jobs,
        recompute=recompute,
        progress=False,
        leave=False,
    )

    measure_optimized(
        sdata=sdata,
        func=_hu_moments,
        points_key=points_key,
        shape_key=shape_key,
        shape_measure_keys=["x", "y"],
        feature_key=feature_key,
        result_key=result_key,
        n_jobs=n_jobs,
        batch_size=batch_size,
        recompute=recompute,
        progress=progress,
    )
