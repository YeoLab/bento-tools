"""
Tests for optimized measure functions using Polars + Numba.
"""

import pytest
import numpy as np
import pandas as pd
import polars as pl
import spatialdata as sd
import geopandas as gpd
import dask.dataframe as dd
import logging
from spatialdata.models import PointsModel, ShapesModel
from shapely.geometry import Polygon

import bento as bt
import bento.experimental.points._measure as original_measure
import bento.experimental.points._measure_optimized as optimized_measure

# Set up logger for this module
logger = logging.getLogger(__name__)


class TestMeasureOptimized:
    """Test suite for optimized measure functions."""

    def _create_synthetic_dataset(self, n_shapes, n_genes, points_per_gene):
        """Create a synthetic dataset with the specified parameters."""
        np.random.seed(42)

        # Create square shapes first, then generate points within them
        shapes_data = {}
        points_data = {"x": [], "y": [], "cell_boundaries": [], "feature_name": []}

        # Calculate grid layout for shapes to avoid overlap
        grid_size = int(np.ceil(np.sqrt(n_shapes)))
        cell_size = 100.0 / grid_size  # 100x100 area

        for i in range(n_shapes):
            # Calculate grid position
            row = i // grid_size
            col = i % grid_size

            # Create a square shape with some padding
            center_x = col * cell_size + cell_size / 2
            center_y = row * cell_size + cell_size / 2
            size = cell_size * 0.8  # 80% of cell size to avoid overlap

            # Create square polygon
            half_size = size / 2
            coords = [
                (center_x - half_size, center_y - half_size),
                (center_x + half_size, center_y - half_size),
                (center_x + half_size, center_y + half_size),
                (center_x - half_size, center_y + half_size),
                (center_x - half_size, center_y - half_size),  # Close the polygon
            ]

            shapes_data[f"cell_{i}"] = Polygon(coords)

            # Generate points for each gene within this shape
            for gene_idx in range(n_genes):
                # Generate points within the square bounds
                x_coords = np.random.uniform(center_x - half_size, center_x + half_size, points_per_gene)
                y_coords = np.random.uniform(center_y - half_size, center_y + half_size, points_per_gene)

                points_data["x"].extend(x_coords)
                points_data["y"].extend(y_coords)
                points_data["cell_boundaries"].extend([f"cell_{i}"] * points_per_gene)
                points_data["feature_name"].extend([f"gene_{gene_idx}"] * points_per_gene)

        # Create SpatialData object
        points_df = pd.DataFrame(points_data)

        # Create GeoDataFrame for shapes
        shapes_gdf = gpd.GeoDataFrame({"geometry": list(shapes_data.values())}, index=list(shapes_data.keys()))

        # Convert points to DaskDataFrame
        points_ddf = dd.from_pandas(points_df, npartitions=1)
        points = PointsModel.parse(points_ddf)
        shapes = ShapesModel.parse(shapes_gdf)

        # Create SpatialData
        data = sd.SpatialData(points={"transcripts": points}, shapes={"cell_boundaries": shapes})
        data = bt.io.prep(
            data,
            points_key="transcripts",
            feature_key="feature_name",
            instance_key="cell_boundaries",
            shape_keys=["cell_boundaries"],
        )
        return data

    # Dataset parameterization
    @pytest.fixture
    def test_data(self, request):
        """Parameterized fixture for test datasets."""
        dataset_size = request.param
        if dataset_size == "small":
            return self._create_synthetic_dataset(n_shapes=100, n_genes=10, points_per_gene=10)
        elif dataset_size == "medium":
            return self._create_synthetic_dataset(n_shapes=1000, n_genes=10, points_per_gene=10)
        elif dataset_size == "large":
            return self._create_synthetic_dataset(n_shapes=10000, n_genes=10, points_per_gene=10)
        else:
            raise ValueError(f"Unknown dataset size: {dataset_size}")

    @pytest.mark.parametrize("test_data", ["small"], indirect=True)
    def test_convert_to_polars(self, test_data):
        """Test conversion of SpatialData to Polars DataFrame."""
        engine = optimized_measure.PolarsMeasureEngine()

        points = engine._convert_to_polars(
            test_data, points_key="transcripts", shape_key="cell_boundaries", feature_key="feature_name"
        )

        assert isinstance(points, pl.DataFrame)
        assert "x" in points.columns
        assert "y" in points.columns
        assert "cell_boundaries" in points.columns
        assert "feature_name" in points.columns

        # Check that categorical columns are properly set
        assert points["cell_boundaries"].dtype == pl.Categorical
        assert points["feature_name"].dtype == pl.Categorical

    @pytest.mark.parametrize("test_data", ["small"], indirect=True)
    def test_prepare_shape_data(self, test_data):
        """Test shape data preparation."""
        engine = optimized_measure.PolarsMeasureEngine()

        # Get some shape names
        points = engine._convert_to_polars(
            test_data, points_key="transcripts", shape_key="cell_boundaries", feature_key="feature_name"
        )
        shape_names = points["cell_boundaries"].unique().to_list()[:10]  # First 10 shapes

        shape_data = engine._prepare_shape_data(test_data, shape_key="cell_boundaries", shape_names=shape_names)

        assert isinstance(shape_data, dict)
        assert len(shape_data) == len(shape_names)

        # Check structure of shape data
        for shape_name, shape_info in shape_data.items():
            assert "geometry" in shape_info
            assert "measurements" in shape_info
            assert isinstance(shape_info["measurements"], dict)

    # Parameterized tests for different measure functions
    @pytest.mark.parametrize("test_data", ["small"], indirect=True)
    def test_density_optimized(self, test_data):
        """Test optimized density calculation."""
        from bento.experimental.points._density import _density

        # Test original function
        original_measure.measure(
            test_data,
            func=_density,
            points_key="transcripts",
            shape_key="cell_boundaries",
            feature_key="feature_name",
            result_key="test_density_original",
            shape_measure_keys=[],
            recompute=True,
            progress=False,
        )

        # Test optimized function
        optimized_measure.density_optimized(
            test_data,
            points_key="transcripts",
            shape_key="cell_boundaries",
            feature_key="feature_name",
            result_key="test_density_optimized",
            recompute=True,
            progress=False,
        )

        # Compare results
        original_results = test_data["test_density_original"].to_df("density")
        optimized_results = test_data["test_density_optimized"].to_df("density")

        # Results should be similar (allowing for small numerical differences)
        np.testing.assert_allclose(original_results.values, optimized_results.values, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("test_data", ["small"], indirect=True)
    def test_distance_optimized(self, test_data):
        """Test optimized distance calculation."""
        # Test original function
        original_measure.distance(
            test_data,
            points_key="transcripts",
            shape_key="cell_boundaries",
            feature_key="feature_name",
            result_key="test_distance_original",
            recompute=True,
            progress=False,
        )

        # Test optimized function
        optimized_measure.distance_optimized(
            test_data,
            points_key="transcripts",
            shape_key="cell_boundaries",
            feature_key="feature_name",
            result_key="test_distance_optimized",
            recompute=True,
            progress=False,
        )

        # Compare results
        original_results = test_data["test_distance_original"].to_df("dist_mean")
        optimized_results = test_data["test_distance_optimized"].to_df("dist_mean")

        # Results should be similar (allowing for small numerical differences)
        np.testing.assert_allclose(original_results.values, optimized_results.values, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("test_data", ["small"], indirect=True)
    def test_polarity_optimized(self, test_data):
        """Test optimized polarity calculation."""
        # Test original function
        original_measure.polarity(
            test_data,
            points_key="transcripts",
            shape_key="cell_boundaries",
            feature_key="feature_name",
            result_key="test_polarity_original",
            recompute=True,
            progress=False,
        )

        # Test optimized function
        optimized_measure.polarity_optimized(
            test_data,
            points_key="transcripts",
            shape_key="cell_boundaries",
            feature_key="feature_name",
            result_key="test_polarity_optimized",
            recompute=True,
            progress=False,
        )

        # Compare results
        original_results = test_data["test_polarity_original"].to_df("polarity")
        optimized_results = test_data["test_polarity_optimized"].to_df("polarity")

        # Results should be similar (allowing for small numerical differences)
        np.testing.assert_allclose(original_results.values, optimized_results.values, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("test_data", ["small"], indirect=True)
    def test_moments_optimized(self, test_data):
        """Test optimized moments calculation."""
        # Test original function
        original_measure.moments(
            test_data,
            points_key="transcripts",
            shape_key="cell_boundaries",
            feature_key="feature_name",
            result_key="test_moments_original",
            recompute=True,
            progress=False,
        )

        # Test optimized function
        optimized_measure.moments_optimized(
            test_data,
            points_key="transcripts",
            shape_key="cell_boundaries",
            feature_key="feature_name",
            result_key="test_moments_optimized",
            recompute=True,
            progress=False,
        )

        # Compare results for each moment
        for moment in ["hum_1", "hum_2", "hum_3", "hum_4", "hum_5", "hum_6"]:
            original_results = test_data["test_moments_original"].to_df(moment)
            optimized_results = test_data["test_moments_optimized"].to_df(moment)

            # Results should be similar (allowing for small numerical differences)
            np.testing.assert_allclose(original_results.values, optimized_results.values, rtol=1e-10, atol=1e-10)

    # Benchmark tests using pytest-benchmark with parameterization
    @pytest.mark.benchmark(group="density-benchmarks")
    @pytest.mark.parametrize("test_data", ["small"], indirect=True)
    def test_benchmark_density(self, benchmark, test_data):
        """Benchmark density calculation on small dataset."""
        benchmark.extra_info["dataset"] = "small"
        benchmark.extra_info["shapes"] = 100
        benchmark.extra_info["points"] = 10000

        def run_density():
            optimized_measure.density_optimized(
                test_data,
                points_key="transcripts",
                shape_key="cell_boundaries",
                feature_key="feature_name",
                result_key="benchmark_density_small",
                recompute=True,
                progress=False,
            )

        benchmark(run_density)

    @pytest.mark.benchmark(group="distance-benchmarks")
    @pytest.mark.parametrize("test_data", ["small", "medium", "large"], indirect=True)
    def test_benchmark_distance(self, benchmark, test_data):
        """Benchmark distance calculation on all dataset sizes."""
        n_shapes = len(test_data.shapes["cell_boundaries"])
        n_points = len(test_data.points["transcripts"])

        # Determine dataset size for metadata
        if n_shapes == 100:
            dataset_size = "small"
        elif n_shapes == 1000:
            dataset_size = "medium"
        elif n_shapes == 10000:
            dataset_size = "large"
        else:
            dataset_size = "unknown"

        benchmark.extra_info["dataset"] = dataset_size
        benchmark.extra_info["shapes"] = n_shapes
        benchmark.extra_info["points"] = n_points

        def run_distance():
            optimized_measure.distance_optimized(
                test_data,
                points_key="transcripts",
                shape_key="cell_boundaries",
                feature_key="feature_name",
                result_key=f"benchmark_distance_{dataset_size}",
                recompute=True,
                progress=False,
            )

        benchmark(run_distance)

    @pytest.mark.benchmark(group="distance-benchmarks")
    @pytest.mark.parametrize("test_data", ["small", "medium", "large"], indirect=True)
    def test_benchmark_distance_original(self, benchmark, test_data):
        """Benchmark original distance calculation on all dataset sizes."""
        n_shapes = len(test_data.shapes["cell_boundaries"])
        n_points = len(test_data.points["transcripts"])

        # Determine dataset size for metadata
        if n_shapes == 100:
            dataset_size = "small"
        elif n_shapes == 1000:
            dataset_size = "medium"
        elif n_shapes == 10000:
            dataset_size = "large"
        else:
            dataset_size = "unknown"

        benchmark.extra_info["dataset"] = dataset_size
        benchmark.extra_info["shapes"] = n_shapes
        benchmark.extra_info["points"] = n_points
        benchmark.extra_info["implementation"] = "original"

        def run_distance_original():
            original_measure.distance(
                test_data,
                points_key="transcripts",
                shape_key="cell_boundaries",
                feature_key="feature_name",
                result_key=f"benchmark_distance_original_{dataset_size}",
                recompute=True,
                progress=False,
            )

        benchmark(run_distance_original)

    @pytest.mark.benchmark(group="polarity-benchmarks")
    @pytest.mark.parametrize("test_data", ["small"], indirect=True)
    def test_benchmark_polarity(self, benchmark, test_data):
        """Benchmark polarity calculation on small dataset."""
        benchmark.extra_info["dataset"] = "small"
        benchmark.extra_info["shapes"] = 100
        benchmark.extra_info["points"] = 10000

        def run_polarity():
            optimized_measure.polarity_optimized(
                test_data,
                points_key="transcripts",
                shape_key="cell_boundaries",
                feature_key="feature_name",
                result_key="benchmark_polarity_small",
                recompute=True,
                progress=False,
            )

        benchmark(run_polarity)

    @pytest.mark.benchmark(group="moments-benchmarks")
    @pytest.mark.parametrize("test_data", ["small"], indirect=True)
    def test_benchmark_moments(self, benchmark, test_data):
        """Benchmark moments calculation on small dataset."""
        benchmark.extra_info["dataset"] = "small"
        benchmark.extra_info["shapes"] = 100
        benchmark.extra_info["points"] = 10000

        def run_moments():
            optimized_measure.moments_optimized(
                test_data,
                points_key="transcripts",
                shape_key="cell_boundaries",
                feature_key="feature_name",
                result_key="benchmark_moments_small",
                recompute=True,
                progress=False,
            )

        benchmark(run_moments)

    # Batch size impact tests
    @pytest.mark.benchmark(group="batch-size-impact")
    @pytest.mark.parametrize("batch_size", [100, 500, 1000, 2000])
    @pytest.mark.parametrize("test_data", ["medium"], indirect=True)
    def test_benchmark_batch_size_impact(self, benchmark, test_data, batch_size):
        """Benchmark different batch sizes on medium dataset."""
        benchmark.extra_info["batch_size"] = batch_size
        benchmark.extra_info["dataset"] = "medium"

        def run_density():
            optimized_measure.density_optimized(
                test_data,
                points_key="transcripts",
                shape_key="cell_boundaries",
                feature_key="feature_name",
                result_key=f"benchmark_batch_{batch_size}",
                batch_size=batch_size,
                recompute=True,
                progress=False,
            )

        benchmark(run_density)

    # Parallelization impact tests
    @pytest.mark.benchmark(group="parallelization-impact")
    @pytest.mark.parametrize("n_jobs", [1, 2, 4])
    @pytest.mark.parametrize("test_data", ["medium"], indirect=True)
    def test_benchmark_parallelization_impact(self, benchmark, test_data, n_jobs):
        """Benchmark different parallelization levels on medium dataset."""
        benchmark.extra_info["n_jobs"] = n_jobs
        benchmark.extra_info["dataset"] = "medium"

        def run_density():
            optimized_measure.density_optimized(
                test_data,
                points_key="transcripts",
                shape_key="cell_boundaries",
                feature_key="feature_name",
                result_key=f"benchmark_njobs_{n_jobs}",
                n_jobs=n_jobs,
                batch_size=100,
                recompute=True,
                progress=False,
            )

        benchmark(run_density)


    @pytest.mark.parametrize("test_data", ["small"], indirect=True)
    def test_error_handling(self, test_data):
        """Test error handling in optimized functions."""

        # Test with invalid shape key
        with pytest.raises(Exception):
            optimized_measure.density_optimized(
                test_data,
                points_key="transcripts",
                shape_key="invalid_shape",
                feature_key="feature_name",
                result_key="test_error",
                recompute=True,
                progress=False,
            )

        # Test with invalid points key
        with pytest.raises(Exception):
            optimized_measure.density_optimized(
                test_data,
                points_key="invalid_points",
                shape_key="cell_boundaries",
                feature_key="feature_name",
                result_key="test_error",
                recompute=True,
                progress=False,
            )

    def test_empty_data_handling(self):
        """Test handling of empty data scenarios."""

        # Create empty points data
        empty_points = pd.DataFrame(columns=["x", "y", "cell_boundaries", "feature_name"])
        empty_shapes = pd.DataFrame(columns=["geometry"])

        empty_data = sd.SpatialData(points={"transcripts": empty_points}, shapes={"cell_boundaries": empty_shapes})

        # Should not raise an error
        optimized_measure.density_optimized(
            empty_data,
            points_key="transcripts",
            shape_key="cell_boundaries",
            feature_key="feature_name",
            result_key="test_empty",
            recompute=True,
            progress=False,
        )

        # Check that result table exists but is empty
        assert "test_empty" in empty_data.tables
        assert empty_data["test_empty"].to_df().empty
