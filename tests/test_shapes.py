"""
Tests for shape measurement features.
"""

import pytest
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

import bento as bt
from bento.shapes import (
    area,
    aspect_ratio,
    bounds,
    centroid,
    opening,
    perimeter,
    radius,
    second_moment,
    span,
)


class TestShapeFeatures:
    """Test suite for shape measurement features."""

    def test_area(self, small_data):
        """Test area calculation."""
        area(
            small_data,
            shape_key="cell_boundaries",
            recompute=True,
            progress=False,
        )

        # Check that area column exists
        assert "area" in small_data.shapes["cell_boundaries"].columns

        # Check data integrity
        areas = small_data.shapes["cell_boundaries"]["area"]
        assert len(areas) > 0

        # Areas should be non-negative
        finite_areas = areas[~np.isnan(areas)]
        if len(finite_areas) > 0:
            assert np.all(finite_areas >= 0)
            assert np.all(np.isfinite(finite_areas))

        # Verify areas match shapely.area
        for idx, shape in small_data.shapes["cell_boundaries"].iterrows():
            expected_area = shape.geometry.area
            computed_area = shape["area"]
            if not np.isnan(expected_area):
                assert abs(expected_area - computed_area) < 1e-6

    def test_aspect_ratio(self, small_data):
        """Test aspect ratio calculation."""
        aspect_ratio(
            small_data,
            shape_key="cell_boundaries",
            recompute=True,
            progress=False,
        )

        # Check that aspect_ratio column exists
        assert "aspect_ratio" in small_data.shapes["cell_boundaries"].columns

        # Check data integrity
        aspect_ratios = small_data.shapes["cell_boundaries"]["aspect_ratio"]
        assert len(aspect_ratios) > 0

        # Aspect ratio should be >= 1 (length/width)
        finite_ratios = aspect_ratios[~np.isnan(aspect_ratios)]
        if len(finite_ratios) > 0:
            assert np.all(finite_ratios >= 1.0)
            assert np.all(np.isfinite(finite_ratios))

    def test_bounds(self, small_data):
        """Test bounds calculation."""
        bounds(
            small_data,
            shape_key="cell_boundaries",
            recompute=True,
            progress=False,
        )

        # Check that all 4 columns exist
        expected_cols = ["xmin", "ymin", "xmax", "ymax"]
        for col in expected_cols:
            assert col in small_data.shapes["cell_boundaries"].columns

        # Check data integrity
        shapes_df = small_data.shapes["cell_boundaries"]
        assert len(shapes_df) > 0

        # Verify bounds ordering: xmin < xmax, ymin < ymax
        finite_mask = ~(
            np.isnan(shapes_df["xmin"])
            | np.isnan(shapes_df["xmax"])
            | np.isnan(shapes_df["ymin"])
            | np.isnan(shapes_df["ymax"])
        )
        if np.any(finite_mask):
            assert np.all(shapes_df.loc[finite_mask, "xmin"] <= shapes_df.loc[finite_mask, "xmax"])
            assert np.all(shapes_df.loc[finite_mask, "ymin"] <= shapes_df.loc[finite_mask, "ymax"])

        # Verify bounds match shapely.bounds
        for idx, shape in shapes_df.iterrows():
            expected_bounds = shape.geometry.bounds
            if not any(np.isnan(expected_bounds)):
                assert abs(expected_bounds[0] - shape["xmin"]) < 1e-6
                assert abs(expected_bounds[1] - shape["ymin"]) < 1e-6
                assert abs(expected_bounds[2] - shape["xmax"]) < 1e-6
                assert abs(expected_bounds[3] - shape["ymax"]) < 1e-6

    def test_centroid(self, small_data):
        """Test centroid calculation."""
        centroid(
            small_data,
            shape_key="cell_boundaries",
            recompute=True,
            progress=False,
        )

        # Check that x and y columns exist
        assert "x" in small_data.shapes["cell_boundaries"].columns
        assert "y" in small_data.shapes["cell_boundaries"].columns

        # Check data integrity
        shapes_df = small_data.shapes["cell_boundaries"]
        assert len(shapes_df) > 0

        # Verify centroids are within bounds
        if "xmin" in shapes_df.columns and "xmax" in shapes_df.columns:
            finite_mask = ~(
                np.isnan(shapes_df["x"])
                | np.isnan(shapes_df["y"])
                | np.isnan(shapes_df["xmin"])
                | np.isnan(shapes_df["xmax"])
            )
            if np.any(finite_mask):
                assert np.all(shapes_df.loc[finite_mask, "x"] >= shapes_df.loc[finite_mask, "xmin"])
                assert np.all(shapes_df.loc[finite_mask, "x"] <= shapes_df.loc[finite_mask, "xmax"])

    def test_opening(self, small_data):
        """Test morphological opening calculation."""
        opening(
            small_data,
            shape_key="cell_boundaries",
            proportion=0.1,
            recompute=True,
            progress=False,
        )

        # Check that opened_shape column exists
        assert "opened_shape" in small_data.shapes["cell_boundaries"].columns

        # Check data integrity
        opened_shapes = small_data.shapes["cell_boundaries"]["opened_shape"]
        assert len(opened_shapes) > 0

        # Verify geometry types are preserved
        for idx, shape in small_data.shapes["cell_boundaries"].iterrows():
            opened = shape["opened_shape"]
            if opened is not None:
                assert isinstance(opened, Polygon) or hasattr(opened, "geom_type")

    def test_perimeter(self, small_data):
        """Test perimeter calculation."""
        perimeter(
            small_data,
            shape_key="cell_boundaries",
            recompute=True,
            progress=False,
        )

        # Check that perimeter column exists
        assert "perimeter" in small_data.shapes["cell_boundaries"].columns

        # Check data integrity
        perimeters = small_data.shapes["cell_boundaries"]["perimeter"]
        assert len(perimeters) > 0

        # Perimeters should be non-negative
        finite_perimeters = perimeters[~np.isnan(perimeters)]
        if len(finite_perimeters) > 0:
            assert np.all(finite_perimeters >= 0)
            assert np.all(np.isfinite(finite_perimeters))

        # Verify perimeters match shapely.length
        for idx, shape in small_data.shapes["cell_boundaries"].iterrows():
            expected_perimeter = shape.geometry.length
            computed_perimeter = shape["perimeter"]
            if not np.isnan(expected_perimeter):
                assert abs(expected_perimeter - computed_perimeter) < 1e-6

    def test_radius(self, small_data):
        """Test radius calculation."""
        radius(
            small_data,
            shape_key="cell_boundaries",
            recompute=True,
            progress=False,
        )

        # Check that radius column exists
        assert "radius" in small_data.shapes["cell_boundaries"].columns

        # Check data integrity
        radii = small_data.shapes["cell_boundaries"]["radius"]
        assert len(radii) > 0

        # Radii should be non-negative
        finite_radii = radii[~np.isnan(radii)]
        if len(finite_radii) > 0:
            assert np.all(finite_radii >= 0)
            assert np.all(np.isfinite(finite_radii))

    def test_second_moment(self, small_data):
        """Test second moment calculation."""
        second_moment(
            small_data,
            shape_key="cell_boundaries",
            recompute=True,
            progress=False,
        )

        # Check that second_moment column exists
        assert "second_moment" in small_data.shapes["cell_boundaries"].columns

        # Check data integrity
        moments = small_data.shapes["cell_boundaries"]["second_moment"]
        assert len(moments) > 0

        # Second moments should be non-negative
        finite_moments = moments[~np.isnan(moments)]
        if len(finite_moments) > 0:
            assert np.all(finite_moments >= 0)
            assert np.all(np.isfinite(finite_moments))

    def test_span(self, small_data):
        """Test span calculation."""
        span(
            small_data,
            shape_key="cell_boundaries",
            recompute=True,
            progress=False,
        )

        # Check that span column exists
        assert "span" in small_data.shapes["cell_boundaries"].columns

        # Check data integrity
        spans = small_data.shapes["cell_boundaries"]["span"]
        assert len(spans) > 0

        # Spans should be non-negative
        finite_spans = spans[~np.isnan(spans)]
        if len(finite_spans) > 0:
            assert np.all(finite_spans >= 0)
            assert np.all(np.isfinite(finite_spans))


class TestShapeFeatureIntegration:
    """Integration tests for shape features."""

    def test_all_features_same_shape(self, small_data):
        """Test that all features can be computed on the same shape dataset."""
        # Compute all features
        area(small_data, shape_key="cell_boundaries", recompute=True, progress=False)
        aspect_ratio(small_data, shape_key="cell_boundaries", recompute=True, progress=False)
        bounds(small_data, shape_key="cell_boundaries", recompute=True, progress=False)
        centroid(small_data, shape_key="cell_boundaries", recompute=True, progress=False)
        opening(small_data, shape_key="cell_boundaries", recompute=True, progress=False)
        perimeter(small_data, shape_key="cell_boundaries", recompute=True, progress=False)
        radius(small_data, shape_key="cell_boundaries", recompute=True, progress=False)
        second_moment(small_data, shape_key="cell_boundaries", recompute=True, progress=False)
        span(small_data, shape_key="cell_boundaries", recompute=True, progress=False)

        # Check that all columns exist
        expected_cols = [
            "area",
            "aspect_ratio",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "x",
            "y",
            "opened_shape",
            "perimeter",
            "radius",
            "second_moment",
            "span",
        ]
        for col in expected_cols:
            assert col in small_data.shapes["cell_boundaries"].columns

    def test_recompute_parameter(self, small_data):
        """Test that recompute=False skips computation when results exist."""
        # First computation
        area(small_data, shape_key="cell_boundaries", recompute=True, progress=False)
        original_area = small_data.shapes["cell_boundaries"]["area"].copy()

        # Second computation with recompute=False should skip
        area(small_data, shape_key="cell_boundaries", recompute=False, progress=False)
        new_area = small_data.shapes["cell_boundaries"]["area"]

        # Results should be unchanged
        pd.testing.assert_series_equal(original_area, new_area)


class TestShapeFeatureErrors:
    """Error handling tests for shape features."""

    def test_invalid_shape_key(self, small_data):
        """Test error handling with invalid shape key."""
        with pytest.raises(Exception):
            area(small_data, shape_key="invalid_shape_key", recompute=True, progress=False)

    def test_multiple_shape_keys(self, small_data):
        """Test that features work with multiple shape keys."""
        # Test with nucleus_boundaries if available
        if "nucleus_boundaries" in small_data.shapes:
            area(small_data, shape_key="nucleus_boundaries", recompute=True, progress=False)
            assert "area" in small_data.shapes["nucleus_boundaries"].columns

