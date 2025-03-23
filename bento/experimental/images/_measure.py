from functools import wraps
from typing import Callable, List, Union

import dask
import dask.bag as db
import dask.dataframe as dd
import geopandas as gpd
import pandas as pd
from anndata import AnnData
from shapely.geometry import Polygon, MultiPolygon
from spatialdata import SpatialData
from tqdm.dask import TqdmCallback

from bento._utils import get_points, get_feature_key


def _enable_gene_groups(func: Callable) -> Callable:
    """Enable gene groups for point feature functions."""

    @wraps(func)
    def wrapper(
        points: dd.DataFrame,
        shape: Union[Polygon, MultiPolygon],
        gene_key: str = None,
    ) -> dict:
        points = points.compute()  # Convert from dask to pandas
        if gene_key:
            result = points.groupby(gene_key, observed=True).apply(
                lambda x: func(x, shape)
            )
        else:
            result = func(points, shape)
        return result

    return wrapper


def _apply_func(
    sdata: SpatialData,
    func: Callable,
    image_key: str,
    shape_key: str = "cell_boundaries",
    by_gene: bool = False,
    num_workers: int = 1,
) -> pd.DataFrame:
    """Internal function to process point calculations with parallel processing."""
    # Get points data
    image = sdata.images[image_key]

    # TODO: figure out how to pass image crops masked by shape to function

    # Get shape data
    shape = sdata.shapes[shape_key].sort_index().geometry
    shape_names = shape.index.tolist()

    points_by_shape = points.groupby(shape_key)

    if by_gene:
        gene_key = get_feature_key(sdata)
    else:
        gene_key = None

    point_feature_func = _enable_gene_groups(func)

    # Create processing bags
    args = [
        (points_by_shape.get_group(s), shape.iloc[i], gene_key)
        for i, s in enumerate(shape_names)
    ]
    bags = db.from_sequence(args).map(lambda x: point_feature_func(*x))

    core_text = "cores" if num_workers > 1 else "core"
    # Compute results with progress bar
    with (
        TqdmCallback(desc=f"Using {num_workers} {core_text}"),
        dask.config.set(
            num_workers=num_workers,
            threads_per_worker=1,
        ),
    ):
        result = bags.compute()

    if not by_gene:
        result = pd.Series(result, index=shape_names)
    else:
        result = pd.DataFrame(result, index=shape_names)

    return result


def measure(
    sdata: SpatialData,
    func: Callable,
    image_key: str = "transcripts",
    shape_key: str = "cell_boundaries",
    channels: Union[str, List[str]] = None,
    result_key_prefix: str = None,
    num_workers: int = 1,
) -> pd.DataFrame:
    """Process image raster clipped by each shape.

    This function applies a function to the image pixel intensities clipped by each shape in the SpatialData object.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object
    func : Callable
        Function to apply to each shape
    image_key : str, optional
        Key to use for points
    shape_key : str, optional
        Key to use for shapes
    channels : str or list of str, optional
        Channels to use for the function
    result_key_prefix : str, optional
        Prefix for result keys
    num_workers : int, optional
        Number of workers to use for parallel processing

    Modifies
    -------
    sdata : SpatialData
        SpatialData object with results added to shapes for each channel:
        - `shapes[shape_key][result_key_prefix + '_' + channel]`
    """

    # Process calculation
    result = _apply_func(
        sdata=sdata,
        func=func,
        image_key=image_key,
        shape_key=shape_key,
        channels=channels,
        num_workers=num_workers,
    )

    # Save results

    col_names = [f"{result_key_prefix}_{c}" for c in channels]
    sdata.shapes[shape_key][col_names] = result
    print(f"""Saved to: sdata['{shape_key}']
          columns: {col_names}""")
