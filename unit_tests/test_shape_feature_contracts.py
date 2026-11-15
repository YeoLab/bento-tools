"""Shape feature values and cache behavior on synthetic rectangles."""

import geopandas as gpd
import numpy as np
import pytest

from bento import tl


@pytest.mark.parametrize(
    "name,values",
    [
        ("area", [12.0, 4.0]),
        ("aspect_ratio", [4 / 3, 1.0]),
        ("perimeter", [14.0, 8.0]),
    ],
)
def test_shape_values_and_recompute(rectangles, name, values):
    function = getattr(tl, name)
    assert function(rectangles, "cells") is None
    key = f"cells_{name}"
    np.testing.assert_allclose(rectangles.shapes["cells"][key], values)
    rectangles.shapes["cells"][key] = -1.0
    function(rectangles, "cells")
    np.testing.assert_array_equal(rectangles.shapes["cells"][key], [-1, -1])
    function(rectangles, "cells", recompute=True)
    np.testing.assert_allclose(rectangles.shapes["cells"][key], values)


def test_bounds_are_aligned_with_shape_indices(rectangles):
    tl.bounds(rectangles, "cells")
    columns = [f"cells_{name}" for name in ["minx", "miny", "maxx", "maxy"]]
    np.testing.assert_allclose(
        rectangles.shapes["cells"][columns], [[0, 0, 4, 3], [10, 10, 12, 12]]
    )
    rectangles.shapes["cells"][columns] = -1
    tl.bounds(rectangles, "cells")
    np.testing.assert_array_equal(
        rectangles.shapes["cells"][columns], np.full((2, 4), -1)
    )
    tl.bounds(rectangles, "cells", recompute=True)
    np.testing.assert_allclose(
        rectangles.shapes["cells"][columns], [[0, 0, 4, 3], [10, 10, 12, 12]]
    )


def test_density_counts_molecules_per_area(rectangles):
    tl.density(rectangles, "cells")
    np.testing.assert_allclose(
        rectangles.shapes["cells"]["cells_density"], [2 / 12, 1 / 4]
    )


def test_radius_of_square_uses_distance_from_center_to_boundary(rectangles):
    tl.radius(rectangles, "cells")
    value = rectangles.shapes["cells"].loc["square", "cells_radius"]
    assert 1.0 <= value <= np.sqrt(2)
    rectangles.shapes["cells"]["cells_radius"] = -1
    tl.radius(rectangles, "cells")
    assert rectangles.shapes["cells"].loc["square", "cells_radius"] == -1
    tl.radius(rectangles, "cells", recompute=True)
    assert rectangles.shapes["cells"].loc["square", "cells_radius"] == pytest.approx(
        value
    )


def test_zero_opening_preserves_geometries(rectangles):
    tl.opening(rectangles, "cells", proportion=0)
    result = rectangles.shapes["cells"]["cells_open_0_shape"]
    assert gpd.GeoSeries(result).geom_equals(rectangles.shapes["cells"].geometry).all()


def test_raster_grid_stays_inside_shapes_and_preserves_metadata(rectangles):
    original_attrs = rectangles.points["transcripts"].attrs.copy()
    tl.raster(rectangles, "cells", step=1)
    raster = rectangles.points["cells_raster"].compute()
    assert not raster.empty
    for name, shape in rectangles.shapes["cells"].geometry.items():
        group = raster[raster["cells"] == name]
        assert not group.empty
        assert shape.covers(gpd.points_from_xy(group.x, group.y)).all()
        np.testing.assert_array_equal(
            group[["x", "y"]].to_numpy(),
            rectangles.shapes["cells"].loc[name, "cells_raster"],
        )
    assert rectangles.points["transcripts"].attrs == original_attrs
    assert (
        "feature_key"
        not in rectangles.points["cells_raster"].attrs["spatialdata_attrs"]
    )
    assert (
        rectangles.points["cells_raster"].attrs["spatialdata_attrs"]["instance_key"]
        == "cells"
    )


def test_second_moment_matches_squared_distances_from_centroid(rectangles):
    tl.second_moment(rectangles, "cells")
    result = rectangles.shapes["cells"]["cells_moment"]
    assert (result > 0).all()
    square = rectangles.shapes["cells"].loc["square", "cells_raster"]
    expected = np.square(square - [11, 11]).sum(axis=1).mean()
    assert result.loc["square"] == pytest.approx(expected)


def test_span_of_three_by_four_rectangle(rectangles):
    tl.span(rectangles, "cells")
    assert rectangles.shapes["cells"].loc["wide", "cells_span"] == pytest.approx(5.0)
    rectangles.shapes["cells"]["cells_span"] = -1
    tl.span(rectangles, "cells")
    assert rectangles.shapes["cells"].loc["wide", "cells_span"] == -1
    tl.span(rectangles, "cells", recompute=True)
    assert rectangles.shapes["cells"].loc["wide", "cells_span"] == pytest.approx(5.0)


@pytest.mark.xfail(
    strict=True, reason="https://github.com/YeoLab/bento-tools/issues/196"
)
def test_span_preserves_noninteger_diameter(rectangles):
    tl.span(rectangles, "cells")
    assert rectangles.shapes["cells"].loc["square", "cells_span"] == pytest.approx(
        np.sqrt(8)
    )
