"""
Tests for point measurement features using efficient vectorized computation.
"""

import pytest
import numpy as np
import pandas as pd
import logging
from tests.conftest import (
    N_SHAPES_SMALL,
    N_SHAPES_MEDIUM,
    N_GENES,
    N_POINTS_PER_GENE,
    _create_unified_synthetic_dataset,
)

from bento.points import distance_stats, polarity, moments, density, morans_i, ripley

# Set up logger for this module
logger = logging.getLogger(__name__)

N_CELLS_SMALL: int = N_SHAPES_SMALL
N_CELLS_MEDIUM: int = N_SHAPES_MEDIUM
N_CELLS_LARGE: int = 10000
N_CELLS_XLARGE: int = 100000


# Dataset parameterization
@pytest.fixture
def test_data(request):
    """Parameterized fixture for test datasets."""
    dataset_size = request.param
    if dataset_size == "small":
        return _create_unified_synthetic_dataset(
            n_shapes=N_CELLS_SMALL, n_genes=N_GENES, points_per_gene=N_POINTS_PER_GENE, include_images=False
        )
    elif dataset_size == "medium":
        return _create_unified_synthetic_dataset(
            n_shapes=N_CELLS_MEDIUM, n_genes=N_GENES, points_per_gene=N_POINTS_PER_GENE, include_images=False
        )
    elif dataset_size == "large":
        return _create_unified_synthetic_dataset(
            n_shapes=N_CELLS_LARGE, n_genes=N_GENES, points_per_gene=N_POINTS_PER_GENE, include_images=False
        )
    elif dataset_size == "xlarge":
        return _create_unified_synthetic_dataset(
            n_shapes=N_CELLS_XLARGE, n_genes=N_GENES, points_per_gene=N_POINTS_PER_GENE, include_images=False
        )
    else:
        raise ValueError(f"Unknown dataset size: {dataset_size}")


class TestMeasures:
    """Test suite for point measurement features."""

    # Basic functionality tests
    @pytest.mark.parametrize("test_data", ["small"], indirect=True)
    def test_distance_stats(self, test_data):
        """Test distance statistics calculation."""
        distance_stats(
            test_data,
            points_key="transcripts",
            shape_key="cell_boundaries",
            feature_key="feature_name",
            recompute=True,
            progress=False,
        )

        # Check that results exist and have expected structure
        assert "table" in test_data.tables
        result_table = test_data.tables["table"]

        # Check that we have both mean and std layers
        assert "dist_mean" in result_table.layers
        assert "dist_std" in result_table.layers

        # Check dimensions
        logger.info(f"result_table.to_df('dist_mean').shape: {result_table.to_df('dist_mean').shape}")

        # Data integrity checks for distance layers
        dist_mean_df = result_table.to_df("dist_mean")
        dist_std_df = result_table.to_df("dist_std")

        # Check that we have some positive distances and variation
        logger.info(
            f"dist_mean_df values: min={dist_mean_df.values.min():.2f}, max={dist_mean_df.values.max():.2f}, mean={dist_mean_df.values.mean():.2f}"
        )

        # Values should be non-negative (distances are always >= 0)
        assert (dist_mean_df.values >= 0).all() or np.isnan(dist_mean_df.values).all()

        # Should have some variation in values (not all identical)
        non_nan_values = dist_mean_df.values[~np.isnan(dist_mean_df.values)]
        if len(non_nan_values) > 1:
            assert non_nan_values.std() >= 0

        # Standard deviations should be non-negative
        assert (dist_std_df.values >= 0).all() or np.isnan(dist_std_df.values).all()

        # Check that both layers have same index and columns
        pd.testing.assert_index_equal(dist_mean_df.index, dist_std_df.index)
        pd.testing.assert_index_equal(dist_mean_df.columns, dist_std_df.columns)

        assert dist_mean_df.values.std() > 0  # Should have some variation

    @pytest.mark.parametrize("test_data", ["small"], indirect=True)
    def test_polarity(self, test_data):
        """Test polarity calculation."""
        polarity(
            test_data,
            points_key="transcripts",
            shape_key="cell_boundaries",
            feature_key="feature_name",
            recompute=True,
            progress=False,
        )

        # Check that results exist and have expected structure
        assert "table" in test_data.tables
        result_table = test_data.tables["table"]

        # Check that we have polarity layer
        assert "polarity" in result_table.layers

        # Check dimensions
        assert result_table.to_df("polarity").shape == (N_CELLS_SMALL, N_GENES)  # 1000 shapes, 100 genes

        # Data integrity checks for polarity layer
        polarity_df = result_table.to_df("polarity")

        # Polarity values should be finite (not inf or extremely large)
        finite_values = polarity_df.values[~np.isnan(polarity_df.values)]
        if len(finite_values) > 0:
            assert np.all(np.isfinite(finite_values))
            # Polarity is typically normalized, so values should be reasonable
            assert np.all(finite_values >= 0)  # Polarity is typically non-negative

    @pytest.mark.parametrize("test_data", ["small"], indirect=True)
    def test_moments(self, test_data):
        """Test moments calculation."""
        moments(
            test_data,
            points_key="transcripts",
            shape_key="cell_boundaries",
            feature_key="feature_name",
            recompute=True,
            progress=False,
        )

        # Check that results exist and have expected structure
        assert "table" in test_data.tables
        result_table = test_data.tables["table"]

        # Check that we have all 6 Hu moment layers
        moment_names = ["hum_1", "hum_2", "hum_3", "hum_4", "hum_5", "hum_6"]
        for moment in moment_names:
            assert moment in result_table.layers
            assert result_table.to_df(moment).shape == (N_CELLS_SMALL, N_GENES)

        # Data integrity checks for moments layers
        for moment_name in moment_names:
            moment_df = result_table.to_df(moment_name)

            # Hu moments should be finite values
            finite_values = moment_df.values[~np.isnan(moment_df.values)]
            if len(finite_values) > 0:
                assert np.all(np.isfinite(finite_values))
                # Hu moments can be positive or negative, but should be reasonable
                assert not np.any(np.abs(finite_values) > 1e6)  # No extremely large values

    @pytest.mark.parametrize("test_data", ["small"], indirect=True)
    def test_density(self, test_data):
        """Test density calculation."""
        density(
            test_data,
            shape_key="cell_boundaries",
            recompute=True,
            progress=False,
        )

        # Check that results exist and have expected structure
        assert "table" in test_data.tables
        result_table = test_data.tables["table"]
        assert "density" in result_table.layers
        assert result_table.to_df("density").shape == (N_CELLS_SMALL, N_GENES)

        # Data integrity checks for density layer
        density_df = result_table.to_df("density")

        # Density values should be non-negative (points per area >= 0)
        finite_values = density_df.values[~np.isnan(density_df.values)]
        if len(finite_values) > 0:
            assert np.all(finite_values >= 0)
            assert np.all(np.isfinite(finite_values))
            # Should have some non-zero densities
            assert np.any(finite_values > 0)

    @pytest.mark.parametrize("test_data", ["small"], indirect=True)
    def test_morans_i(self, test_data):
        """Test Moran's I spatial autocorrelation calculation."""
        morans_i(
            test_data,
            points_key="transcripts",
            shape_key="cell_boundaries",
            feature_key="feature_name",
            recompute=True,
            progress=False,
            n_jobs=1,
        )

        # Check that results exist and have expected structure
        assert "table" in test_data.tables
        result_table = test_data.tables["table"]

        # Check that we have morans_i layer
        assert "morans_i" in result_table.layers

        # Check dimensions
        n_shapes = len(test_data.shapes["cell_boundaries"])
        assert result_table.to_df("morans_i").shape == (n_shapes, N_GENES)

        # Data integrity checks for morans_i layer
        morans_df = result_table.to_df("morans_i")

        # Moran's I values should be finite
        finite_values = morans_df.values[~np.isnan(morans_df.values)]
        if len(finite_values) > 0:
            assert np.all(np.isfinite(finite_values))
            # Moran's I ranges typically from -1 to 1, but can be outside this range
            # Values should be reasonable (not extremely large)
            assert not np.any(np.abs(finite_values) > 10)  # No extremely large values

    @pytest.mark.parametrize("test_data", ["small"], indirect=True)
    def test_ripley(self, test_data):
        """Test Ripley statistics calculation."""
        ripley(
            test_data,
            points_key="transcripts",
            shape_key="cell_boundaries",
            feature_key="feature_name",
            recompute=True,
            progress=False,
            n_jobs=1,
        )

        # Check that results exist and have expected structure
        assert "table" in test_data.tables
        result_table = test_data.tables["table"]

        # Ripley should have some statistical measures
        # Check that at least one ripley layer exists
        ripley_layers = [k for k in result_table.layers.keys() if k.startswith("ripley")]
        assert len(ripley_layers) > 0, "Expected at least one ripley layer"

        # Check dimensions match
        n_shapes = len(test_data.shapes["cell_boundaries"])
        for layer_name in ripley_layers:
            assert result_table.to_df(layer_name).shape == (n_shapes, N_GENES)

        # Data integrity checks for Ripley layers
        for layer_name in ripley_layers:
            ripley_df = result_table.to_df(layer_name)

            # Ripley statistics should be finite
            finite_values = ripley_df.values[~np.isnan(ripley_df.values)]
            if len(finite_values) > 0:
                assert np.all(np.isfinite(finite_values))
                # Ripley statistics should be non-negative for most measures
                # (l_max, l_max_gradient, l_min_gradient, l_half_radius)
                # Some like l_monotony can be negative (correlation coefficient)
                assert not np.any(np.abs(finite_values) > 1e6)  # No extremely large values


@pytest.mark.benchmark
class TestBenchmarks:
    @pytest.mark.parametrize("test_data", ["small"], indirect=True)
    @pytest.mark.parametrize("measure", [distance_stats, polarity, moments, density])
    def test_benchmark_compare_measures(self, benchmark, test_data, measure):
        """Benchmark comparison of distance, polarity, and moments."""
        benchmark.extra_info["measure"] = measure.__name__

        def run_measure():
            measure(
                test_data,
                points_key="transcripts",
                shape_key="cell_boundaries",
                feature_key="feature_name",
                recompute=True,
                progress=False,
            )

        benchmark(run_measure)

    # Benchmark tests for performance comparison
    @pytest.mark.benchmark(group="distance-benchmarks")
    @pytest.mark.parametrize("test_data", ["small", "medium"], indirect=True)
    def test_benchmark_distance_stats(self, benchmark, test_data):
        """Benchmark distance statistics calculation."""
        n_shapes = len(test_data.shapes["cell_boundaries"])
        n_points = len(test_data.points["transcripts"])

        # Determine dataset size for metadata
        if n_shapes == N_CELLS_SMALL:
            dataset_size = "small"
        elif n_shapes == N_CELLS_MEDIUM:
            dataset_size = "medium"
        elif n_shapes == N_CELLS_LARGE:
            dataset_size = "large"
        elif n_shapes == N_CELLS_XLARGE:
            dataset_size = "xlarge"
        else:
            dataset_size = "unknown"

        benchmark.extra_info["dataset"] = dataset_size
        benchmark.extra_info["shapes"] = n_shapes
        benchmark.extra_info["points"] = n_points

        def run_distance_stats():
            distance_stats(
                test_data,
                points_key="transcripts",
                shape_key="cell_boundaries",
                feature_key="feature_name",
                recompute=True,
                progress=False,
            )

        benchmark(run_distance_stats)

    @pytest.mark.benchmark(group="polarity-benchmarks")
    @pytest.mark.parametrize("test_data", ["small", "medium"], indirect=True)
    def test_benchmark_polarity(self, benchmark, test_data):
        """Benchmark polarity calculation."""
        n_shapes = len(test_data.shapes["cell_boundaries"])
        n_points = len(test_data.points["transcripts"])

        # Determine dataset size for metadata
        if n_shapes == N_CELLS_SMALL:
            dataset_size = "small"
        elif n_shapes == N_CELLS_MEDIUM:
            dataset_size = "medium"
        elif n_shapes == N_CELLS_LARGE:
            dataset_size = "large"
        elif n_shapes == N_CELLS_XLARGE:
            dataset_size = "xlarge"
        else:
            dataset_size = "unknown"

        benchmark.extra_info["dataset"] = dataset_size
        benchmark.extra_info["shapes"] = n_shapes
        benchmark.extra_info["points"] = n_points

        def run_polarity():
            polarity(
                test_data,
                points_key="transcripts",
                shape_key="cell_boundaries",
                feature_key="feature_name",
                recompute=True,
                progress=False,
            )

        benchmark(run_polarity)

    @pytest.mark.benchmark(group="moments-benchmarks")
    @pytest.mark.parametrize("test_data", ["small", "medium"], indirect=True)
    def test_benchmark_moments(self, benchmark, test_data):
        """Benchmark moments calculation."""
        n_shapes = len(test_data.shapes["cell_boundaries"])
        n_points = len(test_data.points["transcripts"])

        # Determine dataset size for metadata
        if n_shapes == N_CELLS_SMALL:
            dataset_size = "small"
        elif n_shapes == N_CELLS_MEDIUM:
            dataset_size = "medium"
        elif n_shapes == N_CELLS_LARGE:
            dataset_size = "large"
        elif n_shapes == N_CELLS_XLARGE:
            dataset_size = "xlarge"
        else:
            dataset_size = "unknown"

        benchmark.extra_info["dataset"] = dataset_size
        benchmark.extra_info["shapes"] = n_shapes
        benchmark.extra_info["points"] = n_points

        def run_moments():
            moments(
                test_data,
                points_key="transcripts",
                shape_key="cell_boundaries",
                feature_key="feature_name",
                recompute=True,
                progress=False,
            )

        benchmark(run_moments)

    @pytest.mark.benchmark(group="ripley-benchmarks")
    @pytest.mark.parametrize("test_data", ["small", "medium"], indirect=True)
    def test_benchmark_ripley(self, benchmark, test_data):
        """Benchmark Ripley statistics calculation."""
        n_shapes = len(test_data.shapes["cell_boundaries"])
        n_points = len(test_data.points["transcripts"])

        # Determine dataset size for metadata
        if n_shapes == N_CELLS_SMALL:
            dataset_size = "small"
        elif n_shapes == N_CELLS_MEDIUM:
            dataset_size = "medium"
        elif n_shapes == N_CELLS_LARGE:
            dataset_size = "large"
        elif n_shapes == N_CELLS_XLARGE:
            dataset_size = "xlarge"
        else:
            dataset_size = "unknown"

        benchmark.extra_info["dataset"] = dataset_size
        benchmark.extra_info["shapes"] = n_shapes
        benchmark.extra_info["points"] = n_points

        def run_ripley():
            ripley(
                test_data,
                points_key="transcripts",
                shape_key="cell_boundaries",
                feature_key="feature_name",
                recompute=True,
                progress=False,
            )

        benchmark(run_ripley)


class TestErrors:
    """Error handling tests."""

    # Error handling tests
    @pytest.mark.parametrize("test_data", ["small"], indirect=True)
    def test_error_handling(self, test_data):
        """Test error handling in measurement functions."""

        # Test with invalid shape key
        with pytest.raises(Exception):
            distance_stats(
                test_data,
                points_key="transcripts",
                shape_key="invalid_shape",
                feature_key="feature_name",
                recompute=True,
                progress=False,
            )

        # Test with invalid points key
        with pytest.raises(Exception):
            distance_stats(
                test_data,
                points_key="invalid_points",
                shape_key="cell_boundaries",
                feature_key="feature_name",
                recompute=True,
                progress=False,
            )

    # Additional test for skip behavior when recompute=False
    @pytest.mark.parametrize("test_data", ["small"], indirect=True)
    def test_skip_recompute(self, test_data):
        """Test that functions skip computation when recompute=False and results exist."""

        # First computation
        distance_stats(
            test_data,
            points_key="transcripts",
            shape_key="cell_boundaries",
            feature_key="feature_name",
            recompute=True,
            progress=False,
        )

        # Store original result
        original_result = test_data.tables["table"].to_df("dist_mean").copy()

        # Second computation with recompute=False should skip
        distance_stats(
            test_data,
            points_key="transcripts",
            shape_key="cell_boundaries",
            feature_key="feature_name",
            recompute=False,  # Should skip
            progress=False,
        )

        # Result should be unchanged
        new_result = test_data.tables["table"].to_df("dist_mean")
        pd.testing.assert_frame_equal(original_result, new_result)

    # Test for multiple features in same table
    @pytest.mark.parametrize("test_data", ["small"], indirect=True)
    def test_multiple_features_same_table(self, test_data):
        """Test that multiple features can be stored as layers in the same table."""

        # Compute distance stats
        distance_stats(
            test_data,
            points_key="transcripts",
            shape_key="cell_boundaries",
            feature_key="feature_name",
            recompute=True,
            progress=False,
        )

        # Compute polarity
        polarity(
            test_data,
            points_key="transcripts",
            shape_key="cell_boundaries",
            feature_key="feature_name",
            recompute=True,
            progress=False,
        )

        # Compute density
        density(
            test_data,
            shape_key="cell_boundaries",
            recompute=True,
            progress=False,
        )

        # Check that all features are stored in the same table
        assert "table" in test_data.tables
        result_table = test_data.tables["table"]

        # Check that we have all expected layers
        expected_layers = ["dist_mean", "dist_std", "polarity", "density"]
        for layer_name in expected_layers:
            assert layer_name in result_table.layers

