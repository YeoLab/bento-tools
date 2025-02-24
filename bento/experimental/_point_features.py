import spatialdata as sd
from spatialdata import SpatialData

import dask

# dask.config.set({'dataframe.query-planning': False})
from typing import Union, List, Optional
import pandas as pd
import geopandas as gpd
from tqdm.dask import TqdmCallback

from bento._utils import get_points


def offset(points, shape):
    geo_points = gpd.GeoSeries.from_xy(points["x"], points["y"])
    centroid = shape.centroid  # Shapely Point
    return {"offset": geo_points.distance(centroid).mean()}


point_features = {"offset": offset}


def analyze_points_v2(
    sdata: SpatialData,
    shape_keys: Union[str, List[str]],
    feature_names: Union[str, List[str]],
    points_key: str = "transcripts",
    instance_key: str = "cell_boundaries",
    groupby: Optional[Union[str, List[str]]] = None,
    num_workers: int = 1,
) -> pd.DataFrame:
    """
    Refactored analyze_points to better leverage Dask parallelization
    """
    # 1. Data Preparation
    points_df = get_points(sdata, points_key=points_key, astype="dask", sync=True)
    shapes_df = sdata.shapes[instance_key]

    # 2. Create feature calculators
    feature_calculators = [point_features[f] for f in feature_names for s in shape_keys]

    # 3. Partition points by cells
    points_grouped = points_df.groupby(instance_key, sort=True)
    cell_names = shapes_df.index

    # 4. Create processing bags
    args = [
        (
            points_grouped.get_group(cell),
            shapes_df.loc[cell].geometry,
            feature_calculators,
        )
        for cell in cell_names
    ]

    # 5. Process in parallel using Dask bags
    import dask.bag as db

    bags = db.from_sequence(args).map(process_cell_features)

    # 6. Compute results
    with (
        TqdmCallback(desc="Processing cells"),
        dask.config.set(num_workers=num_workers),
    ):
        results = bags.compute()

    return pd.concat(results)


def process_cell_features(args):
    """Process features for a single cell partition"""
    cell_points, cell_shape, calculators = args

    features = {}
    for calculator in calculators:
        features.update(calculator(cell_points, cell_shape))

    # Create DataFrame with index to avoid ValueError when values are scalar
    output = pd.DataFrame.from_dict(features, orient="index").T
    return output

