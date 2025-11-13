import bento as bt
import geopandas as gpd


def test_overlay_intersection(synthetic_data):
    s1 = "nucleus_boundaries"
    s2 = "cell_boundaries"
    name = "overlay_result"

    # Perform overlay operation using GeoDataFrame.overlay()
    shape1 = synthetic_data.shapes[s1]
    shape2 = synthetic_data.shapes[s2]
    expected_result = shape1.overlay(shape2, how="intersection", make_valid=True)

    # Perform overlay operation using bento.geo.overlay()
    bt.geo.overlay(synthetic_data, s1, s2, name, how="intersection")

    assert name in synthetic_data.shapes
    assert isinstance(synthetic_data.shapes[name], gpd.GeoDataFrame)
    assert (
        synthetic_data[name]
        .geom_equals_exact(expected_result, tolerance=1, align=False)
        .all()
    )
