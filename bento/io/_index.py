from typing import List, Union

import pandas as pd
import geopandas as gpd
from spatialdata._core.spatialdata import SpatialData
from spatialdata.models import ShapesModel
from .._utils import (
    get_points,
    set_points_metadata,
    set_shape_metadata,
)
from .._constants import SHAPE_ID_KEY


def _sjoin_points(
    sdata: SpatialData,
    points_key: str,
    shape_keys: List[str],
):
    """Index points to shapes and add as columns to `sdata.points[points_key]`. Only supports 2D points for now.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object
    points_key : str
        Key for points DataFrame in `sdata.points`

    Returns
    -------
    SpatialData
        Updated SpatialData object with `sdata.points[points_key]` containing new columns for each shape index
    """
    # Grab all shape geometries
    query_shapes = {}
    for shape in shape_keys:
        query_shapes[shape] = gpd.GeoDataFrame(geometry=sdata.shapes[shape].geometry)

    # Grab points as GeoDataFrame
    points = get_points(sdata, points_key, astype="geopandas", sync=False)
    points.index.name = "pt_index"

    # Index points to shapes
    indexed_points = {}
    for shape_key, shape in query_shapes.items():
        shape = query_shapes[shape_key]
        shape.index.name = None  # Forces sjoin to name index "index_right"
        shape.index = shape.index.astype(str)

        indexed_points[shape_key] = (
            points.sjoin(shape, how="left", predicate="intersects")
            .reset_index()
            .drop_duplicates(subset="pt_index")["index_right"]
            .fillna("")
            .values.flatten()
        )

    index_points = pd.DataFrame(indexed_points)
    set_points_metadata(
        sdata, points_key, index_points, columns=list(indexed_points.keys())
    )

    return sdata


def _sjoin_shapes(
    sdata: SpatialData,
    parent_key: str,
    child_keys: List[str],
    instance_map_type: Union[str, dict],
):
    """Adds polygon indexes to sdata.shapes[parent_key][child_key] for point feature analysis.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object
    parent_key : str
        Key for the shape that will be used as the parent for all indexing.
    child_keys : List[str]
        Names of the shapes to add.
    instance_map_type : str
        Type of instance mapping to use. "1tomany" or "manyto1".

    Returns
    -------
    SpatialData
        Updated SpatialData object with `sdata.shapes[parent_key]` containing new columns for each shape index
    """

    # Cast to list if not already
    if isinstance(child_keys, str):
        child_keys = [child_keys]

    # Check if shapes are already indexed to parent shape
    existing_child_keys = set(child_keys) & set(sdata.shapes[parent_key].columns)
    if len(existing_child_keys) == len(child_keys):
        print("SpatialData shapes already mapped. Skipped mapping.")
        return sdata
    if len(existing_child_keys) > 0:
        print(f"{parent_key}: {existing_child_keys} maps already exist. Skipping.")

    # Get child keys that are not already indexed
    child_keys = (
        set(child_keys) - set(sdata.shapes[parent_key].columns) - set(parent_key)
    )

    parent_shape = gpd.GeoDataFrame(
        sdata.shapes[parent_key][[SHAPE_ID_KEY]],
        geometry=sdata.shapes[parent_key].geometry,
    ).reset_index(drop=True)

    # sjoin shapes to parent shape
    for child_key in child_keys:
        child_shape = sdata.shapes[child_key].copy()
        child_attrs = child_shape.attrs
        # Hack for polygons that are 99% contained in parent shape or have shared boundaries
        child_shape = gpd.GeoDataFrame(
            child_shape[[SHAPE_ID_KEY]], geometry=child_shape.buffer(-10e-6)
        ).reset_index(drop=True)

        # For 1tomany, create multipolygons for groups of child shapes such that each parent shape
        # gets a single child shape at most
        if instance_map_type == "1tomany":
            child_shape = (
                child_shape.sjoin(
                    parent_shape,
                    how="left",
                    predicate="covered_by",
                )
                .dissolve(
                    by=f"{SHAPE_ID_KEY}_right",
                    observed=True,
                    dropna=False,
                    aggfunc="first",
                )
                .reset_index(drop=True)[["geometry", f"{SHAPE_ID_KEY}_left"]]
                .rename(columns={f"{SHAPE_ID_KEY}_left": SHAPE_ID_KEY})
            )

        # Map child shape index to parent shape
        parent2child = (
            parent_shape.sjoin(child_shape, how="left", predicate="covers")
            .drop_duplicates(subset=f"{SHAPE_ID_KEY}_left", keep="first")
            .drop_duplicates(subset=f"{SHAPE_ID_KEY}_right", keep="first")
            .set_index(f"{SHAPE_ID_KEY}_left")[f"{SHAPE_ID_KEY}_right"]
            .rename(child_key)
        )
        parent2child.index.name = SHAPE_ID_KEY

        # Add empty category to shape_key if not already present
        parent2child = _add_empty_category(parent2child)
        set_shape_metadata(sdata, shape_key=parent_key, metadata=parent2child)

        # Reverse mapping
        child2parent = (
            parent2child.reset_index()
            .rename(columns={child_key: SHAPE_ID_KEY, SHAPE_ID_KEY: parent_key})
            .set_index(SHAPE_ID_KEY)
            .reindex(child_shape[SHAPE_ID_KEY])[parent_key]
            .dropna()
        )
        child2parent = _add_empty_category(child2parent)

        # Remove child shapes that are not mapped to parent shape
        child_shape = (
            child_shape.set_index(SHAPE_ID_KEY).loc[child2parent.index].reset_index()
        )
        child_shape = ShapesModel.parse(child_shape)
        child_shape.attrs = child_attrs
        sdata.shapes[child_key] = child_shape

        set_shape_metadata(sdata, shape_key=child_key, metadata=child2parent)

    return sdata


def _add_empty_category(series: pd.Series) -> pd.Series:
    if series.dtype == "category" and "" not in series.cat.categories:
        series = series.cat.add_categories([""])
    return series.fillna("")
