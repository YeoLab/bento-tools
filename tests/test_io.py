
def test_points_indexing(synthetic_data):
    # Check points indexing
    assert "cell_boundaries" in synthetic_data.points["transcripts"].columns
    assert "nucleus_boundaries" in synthetic_data.points["transcripts"].columns


def test_shapes_indexing(synthetic_data):
    # Check shapes are added to sdata
    assert "cell_boundaries" in synthetic_data.shapes["cell_boundaries"].columns
    assert "cell_boundaries" in synthetic_data.shapes["nucleus_boundaries"].columns
    assert "nucleus_boundaries" in synthetic_data.shapes["cell_boundaries"].columns

    # check shape indexing exists in both directions
    assert "nucleus_boundaries" in synthetic_data["cell_boundaries"].columns
    assert "cell_boundaries" in synthetic_data["nucleus_boundaries"].columns


def test_points_attrs(synthetic_data):
    # Check points attrs
    assert "transform" in synthetic_data.points["transcripts"].attrs.keys()
    assert (
        synthetic_data.points["transcripts"].attrs["spatialdata_attrs"]["feature_key"]
        == "feature_name"
    )
    assert (
        synthetic_data.points["transcripts"].attrs["spatialdata_attrs"]["instance_key"]
        == "cell_boundaries"
    )


def test_shapes_attrs(synthetic_data):
    # Check shapes attrs
    assert "transform" in synthetic_data.shapes["cell_boundaries"].attrs.keys()
    assert "transform" in synthetic_data.shapes["nucleus_boundaries"].attrs.keys()


def test_index_dtypes(synthetic_data):
    # Check index dtypes
    assert synthetic_data.shapes["cell_boundaries"].index.dtype == "object"
    assert synthetic_data.shapes["nucleus_boundaries"].index.dtype == "object"
    assert synthetic_data.points["transcripts"].index.dtype == "int64"
