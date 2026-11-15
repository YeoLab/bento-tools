"""Numerical contracts for point features on hand-computable geometries."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from bento import tl


@pytest.fixture
def points():
    """Two points inside a square and two outside, with known distances."""
    xs, ys = [0.0, 1.0, 3.0, 4.0], [0.0] * 4
    return gpd.GeoDataFrame(
        {
            "x": xs,
            "y": ys,
            "nucleus": [box(-2, -2, 2, 2)] * 4,
            "nucleus_index": ["n", "n", "", ""],
            "cell_radius": [10.0] * 4,
            "cell_raster": [np.array([[-2.0, 0.0], [2.0, 0.0]])] * 4,
        },
        geometry=gpd.points_from_xy(xs, ys),
    )


@pytest.mark.parametrize(
    "feature,expected",
    [
        (
            tl.ShapeDistance,
            {"nucleus_inner_distance": 1.5, "nucleus_outer_distance": 1.5},
        ),
        (tl.ShapeOffset, {"nucleus_inner_offset": 0.5, "nucleus_outer_offset": 3.5}),
        (
            tl.ShapeProximity,
            {"nucleus_inner_proximity": 0.85, "nucleus_outer_proximity": 0.85},
        ),
        (
            tl.ShapeAsymmetry,
            {"nucleus_inner_asymmetry": 0.05, "nucleus_outer_asymmetry": 0.35},
        ),
        (tl.ShapeEnrichment, {"nucleus_enrichment": 0.5}),
        (tl.ShapeDispersion, {"nucleus_dispersion": 6.5}),
        (tl.ShapeDispersionNorm, {"nucleus_dispersion_norm": 1.625}),
        (tl.PointDispersion, {"point_dispersion": 2.5}),
        (tl.PointDispersionNorm, {"point_dispersion_norm": 0.3125}),
    ],
)
def test_exact_numerical_features(points, feature, expected):
    assert feature("cell", "nucleus").extract(points) == pytest.approx(expected)


@pytest.mark.parametrize(
    "feature",
    [
        tl.ShapeDistance,
        tl.ShapeOffset,
        tl.ShapeProximity,
        tl.ShapeAsymmetry,
        tl.ShapeDispersion,
        tl.ShapeDispersionNorm,
    ],
)
@pytest.mark.parametrize("missing", [None, np.nan])
def test_missing_shape_produces_nan_features(points, feature, missing):
    points["nucleus"] = [missing] * len(points)
    values = feature("cell", "nucleus").extract(points)
    assert values
    assert all(np.isnan(v) for v in values.values())


@pytest.mark.parametrize("inside", [True, False])
@pytest.mark.parametrize("feature", [tl.ShapeDistance, tl.ShapeOffset])
def test_absent_inner_or_outer_points_have_nan_distance(points, feature, inside):
    points["nucleus_index"] = "n" if inside else ""
    result = feature("cell", "nucleus").extract(points)
    absent = "outer" if inside else "inner"
    present = "inner" if inside else "outer"
    assert np.isnan(
        result[
            f"nucleus_{absent}_{'distance' if feature is tl.ShapeDistance else 'offset'}"
        ]
    )
    assert np.isfinite(
        result[
            f"nucleus_{present}_{'distance' if feature is tl.ShapeDistance else 'offset'}"
        ]
    )


@pytest.mark.parametrize("inside,expected", [(True, 1.0), (False, 0.0)])
def test_enrichment_boundary_cases(points, inside, expected):
    points["nucleus_index"] = "n" if inside else ""
    assert tl.ShapeEnrichment("cell", "nucleus").extract(points) == {
        "nucleus_enrichment": expected
    }


def test_instance_shape_includes_every_point(points):
    points["cell"] = box(-5, -5, 5, 5)
    assert tl.ShapeEnrichment("cell", "cell").extract(points) == {
        "cell_enrichment": 1.0
    }
    result = tl.ShapeDistance("cell", "cell").extract(points)
    assert result["cell_inner_distance"] == pytest.approx(3.0)
    assert np.isnan(result["cell_outer_distance"])


def test_normalized_point_dispersion_without_raster(points):
    points["cell_raster"] = [np.empty((0, 2))] * len(points)
    result = tl.PointDispersionNorm("cell", "nucleus").extract(points)
    assert np.isnan(result["point_dispersion_norm"])


def test_feature_requirements_do_not_leak_between_instances():
    proximity = tl.ShapeProximity("cell", "nucleus")
    dispersion = tl.PointDispersion("other")
    assert proximity.cell_features == {"radius"}
    assert proximity.attributes == {"nucleus", "cell_radius"}
    assert dispersion.cell_features == set()
    assert dispersion.attributes == set()


@pytest.mark.parametrize("span", [4.0, 28.0])
def test_ripley_statistics_match_no_pairs_at_quarter_span(span):
    xs, ys = [5.0, 5.0, 15.0, 15.0], [5.0, 15.0, 5.0, 15.0]
    frame = gpd.GeoDataFrame(
        {
            "x": xs,
            "y": ys,
            "cell_span": [span] * 4,
            "cell_minx": [0.0] * 4,
            "cell_miny": [0.0] * 4,
            "cell_maxx": [20.0] * 4,
            "cell_maxy": [20.0] * 4,
            "cell_area": [400.0] * 4,
        },
        geometry=gpd.points_from_xy(xs, ys),
    )
    result = tl.RipleyStats("cell").extract(frame)
    # All pairs are at least 10 units apart, so K(r)=0 and H(r)=-r for r<10.
    assert result["l_half_radius"] == pytest.approx(-span / 4)
    assert set(result) == {
        "l_max",
        "l_max_gradient",
        "l_min_gradient",
        "l_monotony",
        "l_half_radius",
    }
    assert np.isfinite(list(result.values())).all()
    if span == 4:
        assert result["l_max"] == pytest.approx(-1)
        assert result["l_max_gradient"] == result["l_min_gradient"] == 0
