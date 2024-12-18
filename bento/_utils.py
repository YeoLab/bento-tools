# Geometric operations for SpatialData ShapeElements wrapping GeoPandas GeoDataFrames.
from typing import List, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import dask

dask.config.set({"dataframe.query-planning": False})
import dask.dataframe as dd
from spatialdata import SpatialData
from spatialdata.models import PointsModel, ShapesModel, TableModel


def filter_by_gene(
    sdata: SpatialData,
    min_count: int = 10,
    points_key: str = "transcripts",
    feature_key: str = "feature_name",
) -> SpatialData:
    """Filter out genes with low expression.

    Parameters
    ----------
    sdata : SpatialData
        Input SpatialData object
    min_count : int, default 10
        Minimum number of molecules required per gene
    points_key : str, default "transcripts"
        Key for points in sdata.points
    feature_key : str, default "feature_name"
        Column name containing gene identifiers

    Returns
    -------
    SpatialData
        Updated object with filtered:
        - points[points_key]: Only points from expressed genes
        - tables["table"]: Only expressed genes
    """
    gene_filter = (sdata.tables["table"].X >= min_count).sum(axis=0) > 0
    filtered_table = sdata.tables["table"][:, gene_filter]

    filtered_genes = list(
        sdata.tables["table"].var_names.difference(filtered_table.var_names)
    )
    points = get_points(sdata, points_key=points_key, astype="pandas", sync=False)
    points = points[~points[feature_key].isin(filtered_genes)]
    points[feature_key] = points[feature_key].cat.remove_unused_categories()

    transform = sdata[points_key].attrs
    points = PointsModel.parse(
        dd.from_pandas(points, npartitions=1), coordinates={"x": "x", "y": "y"}
    )
    points.attrs = transform
    sdata.points[points_key] = points

    try:
        del sdata.tables["table"]
    except KeyError:
        pass
    sdata.tables["table"] = TableModel.parse(filtered_table)

    return sdata


def get_points(
    sdata: SpatialData,
    points_key: str = "transcripts",
    astype: str = "pandas",
    sync: bool = True,
) -> Union[pd.DataFrame, dd.DataFrame, gpd.GeoDataFrame]:
    """Get points data synchronized with cell boundaries.

    Parameters
    ----------
    sdata : SpatialData
        Input SpatialData object
    points_key : str, default "transcripts"
        Key for points in sdata.points
    astype : str, default "pandas"
        Return type: 'pandas', 'dask', or 'geopandas'
    sync : bool, default True
        Whether to sync points with instance_key shapes

    Returns
    -------
    Union[pd.DataFrame, dd.DataFrame, gpd.GeoDataFrame]
        Points data in requested format

    Raises
    ------
    ValueError
        If points_key not found or invalid astype
    """
    if points_key not in sdata.points.keys():
        raise ValueError(f"Points key {points_key} not found in sdata.points")

    if astype not in ["pandas", "dask", "geopandas"]:
        raise ValueError(
            f"astype must be one of ['dask', 'pandas', 'geopandas'], not {astype}"
        )

    # Sync points to instance_key
    if sync:
        _sync_points(sdata, points_key)

    points = sdata.points[points_key]

    if astype == "pandas":
        return points.compute()
    elif astype == "dask":
        return points
    elif astype == "geopandas":
        points = points.compute()
        return gpd.GeoDataFrame(
            points, geometry=gpd.points_from_xy(points.x, points.y), copy=True
        )


def get_shape(
    sdata: SpatialData, 
    shape_key: str, 
    sync: bool = True
) -> gpd.GeoSeries:
    """Get shape geometries synchronized with cell boundaries.

    Parameters
    ----------
    sdata : SpatialData
        Input SpatialData object
    shape_key : str
        Key for shapes in sdata.shapes
    sync : bool, default True
        Whether to sync shapes with instance_key shapes

    Returns
    -------
    gpd.GeoSeries
        Shape geometries

    Raises
    ------
    ValueError
        If shape_key not found in sdata.shapes
    """
    instance_key = sdata.tables["table"].uns["spatialdata_attrs"]["instance_key"]

    # Make sure shape exists in sdata.shapes
    if shape_key not in sdata.shapes.keys():
        raise ValueError(f"Shape {shape_key} not found in sdata.shapes")

    if sync and shape_key != instance_key:
        _sync_shapes(sdata, shape_key, instance_key)
        shape_index = sdata.shapes[shape_key][instance_key]
        valid_shapes = shape_index != ""
        return sdata.shapes[shape_key][valid_shapes].geometry

    return sdata.shapes[shape_key].geometry


def get_points_metadata(
    sdata: SpatialData,
    metadata_keys: Union[List[str], str],
    points_key: str,
    astype: str = "pandas",
) -> Union[pd.DataFrame, dd.DataFrame]:
    """Get metadata columns from points data.

    Parameters
    ----------
    sdata : SpatialData
        Input SpatialData object
    metadata_keys : str or list of str
        Column name(s) to retrieve
    points_key : str
        Key for points in sdata.points
    astype : str, default "pandas"
        Return type: 'pandas' or 'dask'

    Returns
    -------
    Union[pd.DataFrame, dd.DataFrame]
        Requested metadata columns

    Raises
    ------
    ValueError
        If points_key or metadata_keys not found
    """
    if points_key not in sdata.points.keys():
        raise ValueError(f"Points key {points_key} not found in sdata.points")
    if astype not in ["pandas", "dask"]:
        raise ValueError(f"astype must be one of ['dask', 'pandas'], not {astype}")
    if isinstance(metadata_keys, str):
        metadata_keys = [metadata_keys]
    for key in metadata_keys:
        if key not in sdata.points[points_key].columns:
            raise ValueError(
                f"Metadata key {key} not found in sdata.points[{points_key}]"
            )

    metadata = sdata.points[points_key][metadata_keys]

    if astype == "pandas":
        return metadata.compute()
    elif astype == "dask":
        return metadata


def get_shape_metadata(
    sdata: SpatialData,
    metadata_keys: Union[List[str], str],
    shape_key: str,
) -> pd.DataFrame:
    """Get metadata columns from shapes data.

    Parameters
    ----------
    sdata : SpatialData
        Input SpatialData object
    metadata_keys : str or list of str
        Column name(s) to retrieve
    shape_key : str
        Key for shapes in sdata.shapes

    Returns
    -------
    pd.DataFrame
        Requested metadata columns

    Raises
    ------
    ValueError
        If shape_key or metadata_keys not found
    """
    if shape_key not in sdata.shapes.keys():
        raise ValueError(f"Shape key {shape_key} not found in sdata.shapes")
    if isinstance(metadata_keys, str):
        metadata_keys = [metadata_keys]
    for key in metadata_keys:
        if key not in sdata.shapes[shape_key].columns:
            raise ValueError(
                f"Metadata key {key} not found in sdata.shapes[{shape_key}]"
            )

    return sdata.shapes[shape_key][metadata_keys]


def set_points_metadata(
    sdata: SpatialData,
    points_key: str,
    metadata: Union[List, pd.Series, pd.DataFrame, np.ndarray],
    columns: Union[List[str], str],
) -> None:
    """Add metadata columns to points data.

    Parameters
    ----------
    sdata : SpatialData
        Input SpatialData object
    points_key : str
        Key for points in sdata.points
    metadata : array-like
        Data to add as new columns
    columns : str or list of str
        Names for new columns

    Raises
    ------
    ValueError
        If points_key not found
    """
    if points_key not in sdata.points.keys():
        raise ValueError(f"{points_key} not found in sdata.points")

    columns = [columns] if isinstance(columns, str) else columns

    # metadata = pd.DataFrame(np.array(metadata), columns=columns)
    metadata = np.array(metadata)

    transform = sdata.points[points_key].attrs
    points = sdata.points[points_key].compute()
    points.loc[:, columns] = metadata
    points = PointsModel.parse(
        dd.from_pandas(points, npartitions=1), coordinates={"x": "x", "y": "y"}
    )
    points.attrs = transform
    sdata.points[points_key] = points

    # sdata.points[points_key] = sdata.points[points_key].reset_index(drop=True)
    # for name, series in metadata.items():
    #     series = series.fillna("") if series.dtype == object else series
    #     series = dd.from_pandas(
    #         series, npartitions=sdata.points[points_key].npartitions
    #     ).reset_index(drop=True)
    #     sdata.points[points_key] = sdata.points[points_key].assign(**{name: series})


def set_shape_metadata(
    sdata: SpatialData,
    shape_key: str,
    metadata: Union[List, pd.Series, pd.DataFrame, np.ndarray],
    column_names: Union[List[str], str] = None,
) -> None:
    """Add metadata columns to shapes data.

    Parameters
    ----------
    sdata : SpatialData
        Input SpatialData object
    shape_key : str
        Key for shapes in sdata.shapes
    metadata : array-like
        Data to add as new columns
    column_names : str or list of str, optional
        Names for new columns. If None, use metadata column names

    Raises
    ------
    ValueError
        If shape_key not found
    """
    if shape_key not in sdata.shapes.keys():
        raise ValueError(f"Shape {shape_key} not found in sdata.shapes")

    shape_index = sdata.shapes[shape_key].index

    if isinstance(metadata, list):
        metadata = pd.Series(metadata, index=shape_index)

    if isinstance(metadata, pd.Series) or isinstance(metadata, np.ndarray):
        metadata = pd.DataFrame(metadata)

    if column_names is not None:
        metadata.columns = (
            [column_names] if isinstance(column_names, str) else column_names
        )

    # Fill missing values in string columns with empty string
    str_columns = metadata.select_dtypes(include="object", exclude="number").columns
    metadata[str_columns] = metadata[str_columns].fillna("")

    # Fill missing values in categorical columns with empty string
    cat_columns = metadata.select_dtypes(include="category").columns
    for col in cat_columns:
        if "" not in metadata[col].cat.categories:
            metadata[col] = metadata[col].cat.add_categories([""]).fillna("")

    sdata.shapes[shape_key] = sdata.shapes[shape_key].assign(
        **metadata.reindex(shape_index).to_dict()
    )
    # sdata.shapes[shape_key].loc[:, metadata.columns] = metadata.reindex(shape_index)


def _sync_points(sdata: SpatialData, points_key: str) -> None:
    """Synchronize points with cell boundaries.

    Updates sdata.points[points_key] to only include points within cells.

    Parameters
    ----------
    sdata : SpatialData
        Input SpatialData object
    points_key : str
        Key for points in sdata.points

    """
    points = sdata.points[points_key].compute()
    instance_key = get_instance_key(sdata)

    # Only keep points within instance_key shape
    cells = set(sdata.shapes[instance_key].index)
    transform = sdata.points[points_key].attrs
    points_valid = points[
        points[instance_key].isin(cells)
    ]  # TODO why doesnt this grab the right cells
    # Set points back to SpatialData object
    points_valid = PointsModel.parse(
        dd.from_pandas(points_valid, npartitions=1),
        coordinates={"x": "x", "y": "y"},
    )
    points_valid.attrs = transform
    sdata.points[points_key] = points_valid


def _sync_shapes(sdata: SpatialData, shape_key: str, instance_key: str) -> None:
    """Synchronize shapes with cell boundaries.

    Updates sdata.shapes[shape_key] to only include shapes within cells.

    Parameters
    ----------
    sdata : SpatialData
        Input SpatialData object
    shape_key : str
        Key for shapes to sync
    instance_key : str
        Key for cell boundaries

    """
    shapes = sdata.shapes[shape_key]
    instance_shapes = sdata.shapes[instance_key]
    if shape_key == instance_key:
        return

    # Only keep shapes within instance_key shape
    cells = set(instance_shapes.index)
    shapes = shapes[shapes[instance_key].isin(cells)]

    # Set shapes back to SpatialData object
    transform = sdata.shapes[shape_key].attrs
    shapes_valid = ShapesModel.parse(shapes)
    shapes_valid.attrs = transform
    sdata.shapes[shape_key] = shapes_valid


def get_instance_key(sdata: SpatialData) -> str:
    """Get key for cell boundaries.

    Parameters
    ----------
    sdata : SpatialData
        Input SpatialData object

    Returns
    -------
    str
        Key for cell boundaries in sdata.shapes

    Raises
    ------
    KeyError
        If instance key attribute not found
    """
    try:
        return sdata.points["transcripts"].attrs["spatialdata_attrs"]["instance_key"]
    except KeyError:
        raise KeyError(
            "Instance key attribute not found in spatialdata object. Run bento.io.prep() to setup SpatialData object for bento-tools."
        )


def get_feature_key(sdata: SpatialData) -> str:
    """Get key for gene identifiers.

    Parameters
    ----------
    sdata : SpatialData
        Input SpatialData object

    Returns
    -------
    str
        Column name containing gene identifiers

    Raises
    ------
    KeyError
        If feature key attribute not found
    """
    try:
        return sdata.points["transcripts"].attrs["spatialdata_attrs"]["feature_key"]
    except KeyError:
        raise KeyError("Feature key attribute not found in spatialdata object.")
