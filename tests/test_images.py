"""
Tests for image measurement features.
"""

import pytest
import numpy as np

import bento as bt
from bento.images import total_intensity, mean_intensity, regionprops


class TestImageFeatures:
    """Test suite for image measurement features."""

    def test_total_intensity(self, synthetic_data):
        """Test total intensity calculation."""
        total_intensity(
            synthetic_data,
            image_key="test_image",
            label_key="test_labels",
            num_workers=1,
        )

        # Check that table was created with correct naming
        table_key = "test_labels.test_image.total"
        assert table_key in synthetic_data.tables

        # Check table structure
        table = synthetic_data.tables[table_key]
        assert len(table.obs) > 0  # Should have observations (labels)
        assert len(table.var) > 0  # Should have variables (channels)

        # Check that total_intensity layer exists
        assert "total_intensity" in table.layers

        # Check data integrity
        intensity_df = table.to_df("total_intensity")
        assert intensity_df.shape[0] > 0  # Should have rows (labels)
        assert intensity_df.shape[1] > 0  # Should have columns (channels)

        # Total intensity values should be non-negative
        finite_values = intensity_df.values[~np.isnan(intensity_df.values)]
        if len(finite_values) > 0:
            assert np.all(finite_values >= 0)
            assert np.all(np.isfinite(finite_values))
            # Should have some non-zero intensities
            assert np.any(finite_values > 0)

    def test_mean_intensity(self, synthetic_data):
        """Test mean intensity calculation."""
        mean_intensity(
            synthetic_data,
            image_key="test_image",
            label_key="test_labels",
            num_workers=1,
        )

        # Check that table was created with correct naming
        table_key = "test_labels.test_image.mean"
        assert table_key in synthetic_data.tables

        # Check table structure
        table = synthetic_data.tables[table_key]
        assert len(table.obs) > 0  # Should have observations (labels)
        assert len(table.var) > 0  # Should have variables (channels)

        # Check that mean_intensity layer exists
        assert "mean_intensity" in table.layers

        # Check data integrity
        intensity_df = table.to_df("mean_intensity")
        assert intensity_df.shape[0] > 0  # Should have rows (labels)
        assert intensity_df.shape[1] > 0  # Should have columns (channels)

        # Mean intensity values should be non-negative and reasonable
        finite_values = intensity_df.values[~np.isnan(intensity_df.values)]
        if len(finite_values) > 0:
            assert np.all(finite_values >= 0)
            assert np.all(np.isfinite(finite_values))
            # Mean intensity should be within valid range (0-255 for our test data)
            assert np.all(finite_values <= 255)

    def test_regionprops(self, synthetic_data):
        """Test region properties calculation."""
        regionprops(
            synthetic_data,
            image_key="test_image",
            label_key="test_labels",
            num_workers=1,
        )

        # Check that table was created with correct naming
        table_key = "test_labels.test_image.rprops"
        assert table_key in synthetic_data.tables

        # Check table structure
        table = synthetic_data.tables[table_key]
        assert len(table.obs) > 0  # Should have observations (labels)
        assert len(table.var) > 0  # Should have variables (channels)

        # Check that Hu moment layers exist (regionprops computes moments_weighted_hu)
        # These are typically named like "moments_weighted_hu-1", "moments_weighted_hu-2", etc.
        hu_layers = [k for k in table.layers.keys() if "hu" in k.lower() or "moment" in k.lower()]
        assert len(hu_layers) > 0, "Expected at least one Hu moment layer"

        # Check data integrity for first Hu moment layer
        first_layer = hu_layers[0]
        props_df = table.to_df(first_layer)
        assert props_df.shape[0] > 0  # Should have rows (labels)
        assert props_df.shape[1] > 0  # Should have columns (channels)

        # Hu moments should be finite values
        finite_values = props_df.values[~np.isnan(props_df.values)]
        if len(finite_values) > 0:
            assert np.all(np.isfinite(finite_values))
            # Hu moments should be reasonable (not extremely large)
            assert not np.any(np.abs(finite_values) > 1e6)


class TestImageFeatureIntegration:
    """Integration tests for image features."""

    def test_all_features_same_data(self, synthetic_data):
        """Test that all features can be computed on the same image dataset."""
        # Clear any existing tables first
        tables_to_remove = [k for k in list(synthetic_data.tables.keys()) if k.startswith("test_labels.test_image")]
        for k in tables_to_remove:
            del synthetic_data.tables[k]
        
        # Compute all features
        total_intensity(synthetic_data, image_key="test_image", label_key="test_labels", num_workers=1)
        mean_intensity(synthetic_data, image_key="test_image", label_key="test_labels", num_workers=1)
        regionprops(synthetic_data, image_key="test_image", label_key="test_labels", num_workers=1)

        # Check that all tables were created
        expected_tables = [
            "test_labels.test_image.total",
            "test_labels.test_image.mean",
            "test_labels.test_image.rprops",
        ]
        for table_key in expected_tables:
            assert table_key in synthetic_data.tables

    def test_specific_channels(self, synthetic_data):
        """Test that features work with specific channel selection."""
        # Clear any existing tables first (fixture is session-scoped)
        tables_to_remove = [k for k in list(synthetic_data.tables.keys()) if k.startswith("test_labels.test_image")]
        for k in tables_to_remove:
            del synthetic_data.tables[k]
        
        # Get actual channel names from the image
        channel_names = list(synthetic_data.images["test_image"].coords["c"].values)
        assert len(channel_names) >= 2, "Need at least 2 channels for this test"
        
        # Test with single channel
        total_intensity(
            synthetic_data,
            image_key="test_image",
            label_key="test_labels",
            img_channels=channel_names[0],
            num_workers=1,
        )

        table_key = "test_labels.test_image.total"
        assert table_key in synthetic_data.tables
        table = synthetic_data.tables[table_key]
        # Should have only one channel
        assert len(table.var) == 1

        # Delete the table to test with multiple channels
        del synthetic_data.tables[table_key]
        
        # Test with multiple channels
        total_intensity(
            synthetic_data,
            image_key="test_image",
            label_key="test_labels",
            img_channels=channel_names[:2],
            num_workers=1,
        )

        # Should now have 2 channels
        table = synthetic_data.tables[table_key]
        assert len(table.var) == 2


class TestImageFeatureErrors:
    """Error handling tests for image features."""

    def test_invalid_image_key(self, synthetic_data):
        """Test error handling with invalid image key."""
        with pytest.raises(Exception):
            total_intensity(
                synthetic_data,
                image_key="invalid_image",
                label_key="test_labels",
                num_workers=1,
            )

    def test_invalid_label_key(self, synthetic_data):
        """Test error handling with invalid label key."""
        with pytest.raises(Exception):
            total_intensity(
                synthetic_data,
                image_key="test_image",
                label_key="invalid_labels",
                num_workers=1,
            )

    def test_invalid_channels(self, synthetic_data):
        """Test error handling with invalid channel names."""
        with pytest.raises(Exception):
            total_intensity(
                synthetic_data,
                image_key="test_image",
                label_key="test_labels",
                img_channels="invalid_channel",
                num_workers=1,
            )

