import bento as bt
import pandas as pd
import geopandas as gpd
import dask
dask.config.set({'dataframe.query-planning': False})
import dask.dataframe as dd


def test_get_points(synthetic_data):
    pd_sync = bt.ut.get_points(
        sdata=synthetic_data, points_key="transcripts", astype="pandas", sync=True
    )

    assert type(pd_sync) == pd.DataFrame

    gdf_sync = bt.ut.get_points(
        sdata=synthetic_data, points_key="transcripts", astype="geopandas", sync=True
    )

    assert type(gdf_sync) == gpd.GeoDataFrame

    dd_sync = bt.ut.get_points(
        sdata=synthetic_data, points_key="transcripts", astype="dask", sync=True
    )

    assert type(dd_sync) == dd.core.DataFrame


def test_get_shape(synthetic_data):
    sync = bt.ut.get_shape(sdata=synthetic_data, shape_key="nucleus_boundaries", sync=True)

    assert type(sync) == gpd.GeoSeries


def test_set_points_metadata(synthetic_data):
    list_metadata = [0] * len(synthetic_data.points["transcripts"])
    series_metadata = pd.Series(list_metadata)
    dataframe_metadata = pd.DataFrame(
        {"0": list_metadata, "1": list_metadata, "2": list_metadata}
    )
    column_names = [
        "list_metadata",
        "series_metadata",
        "dataframe_metadata0",
        "dataframe_metadata1",
        "dataframe_metadata2",
    ]

    bt.ut.set_points_metadata(
        sdata=synthetic_data,
        points_key="transcripts",
        metadata=list_metadata,
        columns=column_names[0],
    )
    bt.ut.set_points_metadata(
        sdata=synthetic_data,
        points_key="transcripts",
        metadata=series_metadata,
        columns=column_names[1],
    )
    bt.ut.set_points_metadata(
        sdata=synthetic_data,
        points_key="transcripts",
        metadata=dataframe_metadata,
        columns=[column_names[2], column_names[3], column_names[4]],
    )
    assert all(column in synthetic_data.points["transcripts"] for column in column_names)


def test_set_shape_metadata(synthetic_data):
    list_metadata = [0] * len(synthetic_data.shapes["cell_boundaries"])
    series_metadata = pd.Series(list_metadata)
    dataframe_metadata = pd.DataFrame(
        {"0": list_metadata, "1": list_metadata, "2": list_metadata}
    )
    column_names = [
        "list_metadata",
        "series_metadata",
        "dataframe_metadata0",
        "dataframe_metadata1",
        "dataframe_metadata2",
    ]

    bt.ut.set_shape_metadata(
        sdata=synthetic_data,
        shape_key="cell_boundaries",
        metadata=list_metadata,
        column_names=column_names[0],
    )
    bt.ut.set_shape_metadata(
        sdata=synthetic_data,
        shape_key="cell_boundaries",
        metadata=series_metadata,
        column_names=column_names[1],
    )
    bt.ut.set_shape_metadata(
        sdata=synthetic_data,
        shape_key="cell_boundaries",
        metadata=dataframe_metadata,
        column_names=[column_names[2], column_names[3], column_names[4]],
    )
    assert set(column_names).issubset(synthetic_data.shapes["cell_boundaries"])


def test_get_points_metadata(synthetic_data):
    list_metadata = [0] * len(synthetic_data.points["transcripts"])
    series_metadata = pd.Series(list_metadata)
    column_names = ["list_metadata", "series_metadata"]

    bt.ut.set_points_metadata(
        sdata=synthetic_data,
        points_key="transcripts",
        metadata=list_metadata,
        columns=column_names[0],
    )
    bt.ut.set_points_metadata(
        sdata=synthetic_data,
        points_key="transcripts",
        metadata=series_metadata,
        columns=column_names[1],
    )

    pd_metadata_single = bt.ut.get_points_metadata(
        sdata=synthetic_data,
        points_key="transcripts",
        metadata_keys=column_names[0],
        astype="pandas",
    )
    dd_metadata_single = bt.ut.get_points_metadata(
        sdata=synthetic_data,
        points_key="transcripts",
        metadata_keys=column_names[0],
        astype="dask",
    )
    pd_metadata = bt.ut.get_points_metadata(
        sdata=synthetic_data,
        points_key="transcripts",
        metadata_keys=[column_names[0], column_names[1]],
        astype="pandas",
    )
    dd_metadata = bt.ut.get_points_metadata(
        sdata=synthetic_data,
        points_key="transcripts",
        metadata_keys=[column_names[0], column_names[1]],
        astype="dask",
    )

    assert type(pd_metadata_single) == pd.DataFrame
    assert column_names[0] in pd_metadata_single

    assert type(dd_metadata_single) == dd.core.DataFrame
    assert column_names[0] in dd_metadata_single

    assert type(pd_metadata) == pd.DataFrame
    assert type(dd_metadata) == dd.core.DataFrame
    assert "list_metadata" in pd_metadata_single
    assert "list_metadata" in dd_metadata_single
    assert all(column in pd_metadata for column in column_names)
    assert all(column in dd_metadata for column in column_names)


def test_get_shape_metadata(synthetic_data):
    list_metadata = [0] * len(synthetic_data.shapes["cell_boundaries"])
    series_metadata = pd.Series(list_metadata)
    column_names = ["list_metadata", "series_metadata"]

    bt.ut.set_shape_metadata(
        sdata=synthetic_data,
        shape_key="cell_boundaries",
        metadata=list_metadata,
        column_names=column_names[0],
    )
    bt.ut.set_shape_metadata(
        sdata=synthetic_data,
        shape_key="cell_boundaries",
        metadata=series_metadata,
        column_names=column_names[1],
    )
    metadata_single = bt.ut.get_shape_metadata(
        sdata=synthetic_data, shape_key="cell_boundaries", metadata_keys=column_names[0]
    )
    metadata = bt.ut.get_shape_metadata(
        sdata=synthetic_data,
        shape_key="cell_boundaries",
        metadata_keys=[column_names[0], column_names[1]],
    )

    assert type(metadata_single) == pd.DataFrame
    assert column_names[0] in metadata_single

    assert type(metadata) == pd.DataFrame
    assert all(column in metadata for column in column_names)
