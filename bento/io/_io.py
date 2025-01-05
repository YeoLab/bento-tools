import warnings
from typing import List, Union, Dict

import emoji
import spatialdata as sd
from anndata import AnnData
from spatialdata import SpatialData
from spatialdata.models import TableModel
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from .._utils import _sync_points, get_cell_key
from ._index import _sjoin_points, _sjoin_shapes
from .._constants import SHAPE_GRAPH_KEY, IS_CELL_KEY, SHAPE_ID_KEY

warnings.filterwarnings("ignore")


def prep(
    sdata: SpatialData,
    cell_key: str,
    points_key: str = "transcripts",
    feature_key: str = "feature_name",
    shape_map: List[Dict] = None,
    map_type: Union[dict, str] = "1to1",
) -> SpatialData:
    """Computes spatial indices for elements in SpatialData; required to use the Bento API.

    The function performs the following steps:
    1. Preprocesses input shapes and points.
        - Forces points to 2D
        - Gives every shape an id column
    2. Creates a shape graph to represent parent-child relationships between shapes.
    3. Indexes points to shapes. (Points outside of {cell_key} shape are filtered out.)
    4. Computes a count table for each shape.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object
    cell_key : str
        Key for the cell shape
    points_key : str
        Key for the points to index
    feature_key : str
        Key for the feature to count
    shape_map : List[Dict]
        List of dictionaries representing parent-child relationships between shapes
    map_type : Union[dict, str]
        How parent-child relationships are defined.
        - "1to1": each parent has at most one child i.e. at most one nucleus per cell
        - "1tomany": each parent can have one or more children. MultiPolygons are used to represent this. i.e. multi-nucleated cells

    Returns
    -------
    SpatialData
        Updated SpatialData object with shape graph and count tables. Mapped shapes have
        new columns for each parent/child relationship.
    """

    shape_keys = list(sdata.shapes.keys())
    shape_keys.sort()

    sdata = _preprocess_input(sdata, points_key, feature_key)

    # Create shape graph
    sdata = _create_shape_graph(sdata, cell_key, shape_map, shape_keys)

    # sindex points and sjoin shapes if they have not been indexed or joined
    sdata = _map_shapes(sdata, shape_map, map_type)
    sdata = _sjoin_points(sdata, points_key, shape_keys)

    # Only keep points within instance_key shape
    _sync_points(sdata, points_key)

    # Recompute count table
    sdata = _compute_counts(sdata, points_key, feature_key, shape_keys)

    return sdata


def _preprocess_input(sdata, points_key, feature_key):
    """Preprocess input shapes and points. Forces points to 2D."""
    transform = {  # Assume points are in global coordinate system
        "global": sd.transformations.get_transformation(sdata.points[points_key])
    }
    if "global" in sdata.points[points_key].attrs["transform"]:
        # Force points to 2D for Xenium data
        if isinstance(transform["global"], sd.transformations.Scale):
            transform = {
                "global": sd.transformations.Scale(
                    scale=transform.to_scale_vector(["x", "y"]), axes=["x", "y"]
                )
            }
    sdata.points[points_key] = sd.models.PointsModel.parse(
        sdata.points[points_key].compute().reset_index(drop=True),
        coordinates={"x": "x", "y": "y"},
        feature_key=feature_key,
        transformations=transform,
    )

    # Give every shape an id column
    for shape_key in sdata.shapes.keys():
        shape = sdata.shapes[shape_key]
        attrs = sdata.shapes[shape_key].attrs
        shape[SHAPE_ID_KEY] = [f"S-{i}" for i in range(len(shape))]  # Add id column
        shape = shape.reset_index(drop=True)  # Remove index
        shape.attrs = attrs  # Add attrs
        sdata.shapes[shape_key] = sd.models.ShapesModel.parse(shape)
    return sdata


def _create_shape_graph(sdata, cell_key, shape_map, shape_keys):
    map_obs = pd.DataFrame(index=shape_keys)
    map_obs[IS_CELL_KEY] = map_obs.index == cell_key

    # Create spares square matrix of shape parent-child relationships
    # upper triangle is parent-child, lower triangle is child-parent
    map_obsp = pd.DataFrame(0, index=shape_keys, columns=shape_keys, dtype=np.uint8)

    if shape_map is not None:
        for parent_key in shape_map:
            for child_key in shape_map[parent_key]:
                map_obsp.loc[parent_key, child_key] = 1

    sdata[SHAPE_GRAPH_KEY] = AnnData(obs=map_obs, obsp={SHAPE_GRAPH_KEY: map_obsp})

    return sdata


def _map_shapes(sdata, shape_map, map_type):
    """Iteratively map shapes to shapes according to shape_map relationships."""
    if shape_map is None:
        return sdata

    for parent_key in shape_map:
        # Make sure all child keys are in sdata.shapes
        child_keys = shape_map[parent_key]
        child_keys = [
            child_key for child_key in child_keys if child_key in sdata.shapes
        ]

        sdata = _sjoin_shapes(sdata, parent_key, child_keys, map_type)

    return sdata


def _compute_counts(sdata, points_key, feature_key, shape_keys):
    # TODO refactor all references to sdata['table'] to sdata["{shape_key}_table"]

    for shape_key in shape_keys:
        table = TableModel.parse(
            sdata.aggregate(
                values=points_key,
                instance_key=shape_key,
                by=shape_key,
                value_key=feature_key,
                aggfunc="count",
            ).tables["table"]
        )
        sdata[f"{shape_key}_table"] = table

    return sdata


def _print_shape_graph(sdata):
    """Print a visual representation of the parent/child relationships between shapes."""
    # Get the shape relationships from the graph
    shape_graph = sdata[SHAPE_GRAPH_KEY].obsp[SHAPE_GRAPH_KEY]
    shape_names = sdata[SHAPE_GRAPH_KEY].obs_names

    # Build dictionary of parent->children relationships
    relationships = {}
    for i, parent in enumerate(shape_names):
        children = shape_names[shape_graph[i].nonzero()[0]]
        relationships[parent] = sorted(children)

    # Helper function to print tree recursively
    def print_tree(node, prefix="", is_last=True):
        stack = [(node, prefix, is_last)]

        print("Shape Graph:")
        while stack:
            current, current_prefix, current_is_last = stack.pop()
            connector = "└── " if current_is_last else "├── "
            print(f"{current_prefix}{connector}{current}")

            children = relationships.get(current, [])
            child_prefix = current_prefix + ("    " if current_is_last else "│   ")

            # Add children to stack in reverse order to maintain same output order
            for i in range(len(children) - 1, -1, -1):
                child = children[i]
                child_is_last = i == len(children) - 1
                stack.append((child, child_prefix, child_is_last))

    # Get root node using IS_CELL_KEY
    root = get_cell_key(sdata)
    print_tree(root)
