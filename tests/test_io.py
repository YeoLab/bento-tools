from tests import conftest

import bento as bt

from bento._constants import SHAPE_ID_KEY


def test_points_indexing(small_data):
    # Check points indexing
    assert "cell_boundaries" in small_data.points["transcripts"].columns
    assert "nucleus_boundaries" in small_data.points["transcripts"].columns


def test_shapes_indexing(small_data):
    # Check shapes are added to sdata
    cell_key = bt.ut.get_cell_key(small_data)
    assert bt._constants.SHAPE_GRAPH_KEY in small_data.tables.keys()
    assert (
        SHAPE_ID_KEY in small_data.shapes[cell_key].columns
        and SHAPE_ID_KEY in small_data.shapes["nucleus_boundaries"].columns
    )
    assert cell_key in small_data.shapes["nucleus_boundaries"].columns
    assert "nucleus_boundaries" in small_data.shapes[cell_key].columns

    # check shape indexing is accurate in both directions
    assert (
        small_data[cell_key].set_index(SHAPE_ID_KEY)["nucleus_boundaries"].to_dict()
        == conftest.CELL_TO_NUCLEUS_MAP
    )
    assert (
        small_data["nucleus_boundaries"].set_index(SHAPE_ID_KEY)[cell_key].to_dict()
        == conftest.NUCLEUS_TO_CELL_MAP
    )


def test_points_attrs(small_data):
    # Check points attrs
    assert "transform" in small_data.points["transcripts"].attrs.keys()
    assert (
        small_data.points["transcripts"].attrs["spatialdata_attrs"]["feature_key"]
        == "feature_name"
    )


def test_shapes_attrs(small_data):
    # Check shapes attrs
    assert "transform" in small_data.shapes["cell_boundaries"].attrs.keys()
    assert "transform" in small_data.shapes["nucleus_boundaries"].attrs.keys()
