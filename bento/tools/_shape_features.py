import warnings

from shapely.errors import ShapelyDeprecationWarning

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
warnings.filterwarnings("ignore")


from typing import Callable, Dict, List, Union, Optional

import matplotlib.path as mplPath
import numpy as np
import pandas as pd
from scipy.spatial import distance, distance_matrix
from shapely.geometry import MultiPolygon, Point
from spatialdata._core.spatialdata import SpatialData
from spatialdata.models import PointsModel, ShapesModel
from tqdm.auto import tqdm
import copy

from .._utils import get_points, get_shape, set_shape_metadata


def area(sdata: SpatialData, shape_key: str, recompute: bool = False) -> None:
    """Compute the area of each shape.

    Parameters
    ----------
    sdata : SpatialData
        Input SpatialData object
    shape_key : str
        Key in sdata.shapes containing shape geometries
    recompute : bool, default False
        Whether to force recomputation if feature exists

    Returns
    -------
    None
        Updates sdata.shapes[shape_key] with:
        - '{shape_key}_area': Area of each polygon
    """

    feature_key = f"{shape_key}_area"
    if feature_key in sdata.shapes[shape_key].columns and not recompute:
        return

    # Calculate pixel-wise area
    area = get_shape(sdata=sdata, shape_key=shape_key, sync=False).area
    set_shape_metadata(
        sdata=sdata, shape_key=shape_key, metadata=area, column_names=feature_key
    )


def _poly_aspect_ratio(poly: Union[MultiPolygon, None]) -> float:
    """Compute aspect ratio of minimum rotated rectangle containing a polygon.

    Parameters
    ----------
    poly : MultiPolygon or None
        Input polygon geometry

    Returns
    -------
    float
        Ratio of longest to shortest side of minimum bounding rectangle,
        or np.nan if polygon is None
    """

    if not poly:
        return np.nan

    # get coordinates of min bounding box vertices around polygon
    x, y = poly.minimum_rotated_rectangle.exterior.coords.xy

    # get length of bound box sides
    edge_length = (
        Point(x[0], y[0]).distance(Point(x[1], y[1])),
        Point(x[1], y[1]).distance(Point(x[2], y[2])),
    )

    # length = longest side, width = shortest side
    length, width = max(edge_length), min(edge_length)

    # return long / short ratio
    return length / width


def aspect_ratio(sdata: SpatialData, shape_key: str, recompute: bool = False) -> None:
    """Compute aspect ratio of minimum rotated rectangle containing each shape.

    The aspect ratio is defined as the ratio of the longest to shortest side
    of the minimum rotated rectangle that contains the shape.

    Parameters
    ----------
    sdata : SpatialData
        Input SpatialData object
    shape_key : str
        Key in sdata.shapes containing shape geometries
    recompute : bool, default False
        Whether to force recomputation if feature exists

    Returns
    -------
    None
        Updates sdata.shapes[shape_key] with:
        - '{shape_key}_aspect_ratio': Ratio of major to minor axis
    """

    feature_key = f"{shape_key}_aspect_ratio"
    if feature_key in sdata.shapes[shape_key].keys() and not recompute:
        return

    ar = get_shape(sdata, shape_key, sync=False).apply(_poly_aspect_ratio)
    set_shape_metadata(
        sdata=sdata, shape_key=shape_key, metadata=ar, column_names=feature_key
    )


def bounds(sdata: SpatialData, shape_key: str, recompute: bool = False) -> None:
    """Compute bounding box coordinates for each shape.

    Parameters
    ----------
    sdata : SpatialData
        Input SpatialData object
    shape_key : str
        Key in sdata.shapes containing shape geometries
    recompute : bool, default False
        Whether to force recomputation if feature exists

    Returns
    -------
    None
        Updates sdata.shapes[shape_key] with:
        - '{shape_key}_minx': x-axis lower bound
        - '{shape_key}_miny': y-axis lower bound
        - '{shape_key}_maxx': x-axis upper bound
        - '{shape_key}_maxy': y-axis upper bound
    """

    feat_names = ["minx", "miny", "maxx", "maxy"]
    feature_keys = [f"{shape_key}_{k}" for k in feat_names]
    if (
        all([k in sdata.shapes[shape_key].keys() for k in feature_keys])
        and not recompute
    ):
        return

    bounds = get_shape(sdata, shape_key, sync=False).bounds

    set_shape_metadata(
        sdata=sdata,
        shape_key=shape_key,
        metadata=bounds[feat_names],
        column_names=feature_keys,
    )


def density(sdata: SpatialData, shape_key: str, recompute: bool = False) -> None:
    """Compute RNA density (molecules per area) for each shape.

    Parameters
    ----------
    sdata : SpatialData
        Input SpatialData object
    shape_key : str
        Key in sdata.shapes containing shape geometries
    recompute : bool, default False
        Whether to force recomputation if feature exists

    Returns
    -------
    None
        Updates sdata.shapes[shape_key] with:
        - '{shape_key}_density': Number of molecules divided by shape area
    """

    feature_key = f"{shape_key}_density"
    if feature_key in sdata.shapes[shape_key].keys() and not recompute:
        return

    count = (
        get_points(sdata, astype="dask", sync=False)
        .query(f"{shape_key} != 'None'")[shape_key]
        .value_counts()
        .compute()
        .reindex_like(sdata.shapes[shape_key])
    )
    area(sdata, shape_key)

    set_shape_metadata(
        sdata=sdata,
        shape_key=shape_key,
        metadata=count / sdata.shapes[shape_key][f"{shape_key}_area"],
        column_names=feature_key,
    )


def opening(
    sdata: SpatialData, shape_key: str, proportion: float, recompute: bool = False
) -> None:
    """Compute morphological opening of each shape.

    The opening operation erodes the shape by distance d and then dilates by d,
    where d = proportion * shape radius.

    Parameters
    ----------
    sdata : SpatialData
        Input SpatialData object
    shape_key : str
        Key in sdata.shapes containing shape geometries
    proportion : float
        Fraction of shape radius to use as opening distance
    recompute : bool, default False
        Whether to force recomputation if feature exists

    Returns
    -------
    None
        Updates sdata.shapes[shape_key] with:
        - '{shape_key}_open_{proportion}_shape': Opened shape geometries
    """

    feature_key = f"{shape_key}_open_{proportion}_shape"

    if feature_key in sdata.shapes[shape_key].keys() and not recompute:
        return

    radius(sdata, shape_key)

    shapes = get_shape(sdata, shape_key, sync=False)
    d = proportion * sdata.shapes[shape_key][f"{shape_key}_radius"]
    set_shape_metadata(
        sdata=sdata,
        shape_key=shape_key,
        metadata=shapes.buffer(-d).buffer(d),
        column_names=feature_key,
    )


def _second_moment_polygon(centroid: Point, pts: np.ndarray) -> Optional[float]:
    """Calculate second moment of points relative to a centroid.

    Parameters
    ----------
    centroid : Point
        Reference point for moment calculation
    pts : np.ndarray
        Array of point coordinates, shape (n, 2)

    Returns
    -------
    float or None
        Second moment value, or None if inputs are invalid
    """

    if not centroid or not isinstance(pts, np.ndarray):
        return
    centroid = np.array(centroid.coords).reshape(1, 2)
    radii = distance.cdist(centroid, pts)
    second_moment = np.sum(radii * radii / len(pts))
    return second_moment


def second_moment(sdata: SpatialData, shape_key: str, recompute: bool = False) -> None:
    """Compute second moment of each shape relative to its centroid.

    The second moment measures the spread of points in the shape around its center.

    Parameters
    ----------
    sdata : SpatialData
        Input SpatialData object
    shape_key : str
        Key in sdata.shapes containing shape geometries
    recompute : bool, default False
        Whether to force recomputation if feature exists

    Returns
    -------
    None
        Updates sdata.shapes[shape_key] with:
        - '{shape_key}_moment': Second moment value for each shape
    """

    feature_key = f"{shape_key}_moment"
    if feature_key in sdata.shapes[shape_key].keys() and not recompute:
        return

    raster(sdata, shape_key, recompute=recompute)

    rasters = sdata.shapes[shape_key][f"{shape_key}_raster"]
    shape_centroids = get_shape(sdata, shape_key, sync=False).centroid

    moments = [
        _second_moment_polygon(centroid, r)
        for centroid, r in zip(shape_centroids, rasters)
    ]

    set_shape_metadata(
        sdata=sdata, shape_key=shape_key, metadata=moments, column_names=feature_key
    )


def _raster_polygon(poly: Union[MultiPolygon, None], step: int = 1) -> Optional[np.ndarray]:
    """Generate grid of points contained within a polygon.

    Parameters
    ----------
    poly : MultiPolygon or None
        Input polygon geometry
    step : int, default 1
        Grid spacing between points

    Returns
    -------
    np.ndarray or None
        Array of grid point coordinates, shape (n, 2),
        or None if polygon is invalid
    """

    if not poly:
        return
    minx, miny, maxx, maxy = [int(i) for i in poly.bounds]
    x, y = np.meshgrid(
        np.arange(minx, maxx + step, step=step),
        np.arange(miny, maxy + step, step=step),
    )
    x = x.flatten()
    y = y.flatten()
    xy = np.array([x, y]).T

    poly_cell_mask = np.ones(xy.shape[0], dtype=bool)

    # Add all points within the polygon; handle MultiPolygons
    if isinstance(poly, MultiPolygon):
        for p in poly:
            poly_path = mplPath.Path(np.array(p.exterior.xy).T)
            poly_cell_mask = poly_cell_mask & poly_path.contains_points(xy)
    else:
        poly_path = mplPath.Path(np.array(poly.exterior.xy).T)
        poly_cell_mask = poly_path.contains_points(xy)
    xy = xy[poly_cell_mask]

    # Make sure at least a single point is returned
    if xy.shape[0] == 0:
        return np.array(poly.centroid.xy).reshape(1, 2)
    return xy


def raster(
    sdata: SpatialData,
    shape_key: str,
    points_key: str = "transcripts",
    step: int = 1,
    recompute: bool = False,
) -> None:
    """Generate grid of points within each shape.

    Creates a regular grid of points with spacing 'step' that covers each shape.
    Points outside the shape are excluded.

    Parameters
    ----------
    sdata : SpatialData
        Input SpatialData object
    shape_key : str
        Key in sdata.shapes containing shape geometries
    points_key : str, default "transcripts"
        Key for points in sdata.points
    step : int, default 1
        Grid spacing between points
    recompute : bool, default False
        Whether to force recomputation if feature exists

    Returns
    -------
    None
        Updates:
        - sdata.shapes[shape_key]['{shape_key}_raster']: Array of grid points per shape
        - sdata.points['{shape_key}_raster']: All grid points as point cloud
    """

    shape_feature_key = f"{shape_key}_raster"

    if shape_feature_key in sdata.shapes[shape_key].keys() and not recompute:
        return

    shapes = get_shape(sdata, shape_key, sync=False)
    raster = shapes.apply(lambda poly: _raster_polygon(poly, step=step))

    raster_all = []
    for s, r in raster.items():
        raster_df = pd.DataFrame(r, columns=["x", "y"])
        raster_df[shape_key] = s
        raster_all.append(raster_df)

    # Add raster to sdata.shapes as 2d array per cell (for point_features compatibility)
    set_shape_metadata(
        sdata=sdata,
        shape_key=shape_key,
        metadata=[df[["x", "y"]].values for df in raster_all],
        column_names=shape_feature_key,
    )

    # Add raster to sdata.points as long dataframe (for flux compatibility)
    raster_all = pd.concat(raster_all).reset_index(drop=True)

    sdata.points[shape_feature_key] = PointsModel.parse(
        raster_all, coordinates={"x": "x", "y": "y"}
    )

    transform = copy.deepcopy(sdata.points[points_key].attrs)
    if "feature_key" in transform["spatialdata_attrs"]:
        del transform["spatialdata_attrs"]["feature_key"]
    sdata.points[shape_feature_key].attrs = transform


def perimeter(sdata: SpatialData, shape_key: str, recompute: bool = False) -> None:
    """Compute perimeter length of each shape.

    Parameters
    ----------
    sdata : SpatialData
        Input SpatialData object
    shape_key : str
        Key in sdata.shapes containing shape geometries
    recompute : bool, default False
        Whether to force recomputation if feature exists

    Returns
    -------
    None
        Updates sdata.shapes[shape_key] with:
        - '{shape_key}_perimeter': Perimeter length of each shape
    """

    feature_key = f"{shape_key}_perimeter"
    if feature_key in sdata.shapes[shape_key].keys() and not recompute:
        return

    set_shape_metadata(
        sdata=sdata,
        shape_key=shape_key,
        metadata=get_shape(sdata, shape_key, sync=False).length,
        column_names=feature_key,
    )


def radius(sdata: SpatialData, shape_key: str, recompute: bool = False) -> None:
    """Compute average radius of each shape.

    The radius is calculated as the mean distance from the shape's centroid
    to points on its boundary.

    Parameters
    ----------
    sdata : SpatialData
        Input SpatialData object
    shape_key : str
        Key in sdata.shapes containing shape geometries
    recompute : bool, default False
        Whether to force recomputation if feature exists

    Returns
    -------
    None
        Updates sdata.shapes[shape_key] with:
        - '{shape_key}_radius': Average radius of each shape
    """

    feature_key = f"{shape_key}_radius"
    if feature_key in sdata.shapes[shape_key].keys() and not recompute:
        return

    shapes = get_shape(sdata, shape_key, sync=False)

    # Get average distance from boundary to centroid
    shape_radius = shapes.apply(_shape_radius)
    set_shape_metadata(
        sdata=sdata,
        shape_key=shape_key,
        metadata=shape_radius,
        column_names=feature_key,
    )


def _shape_radius(poly: Union[MultiPolygon, None]) -> float:
    """Compute average radius of a polygon.

    Calculates mean distance from centroid to boundary points.

    Parameters
    ----------
    poly : MultiPolygon or None
        Input polygon geometry

    Returns
    -------
    float
        Average radius, or np.nan if polygon is None
    """

    if not poly:
        return np.nan

    return distance.cdist(
        np.array(poly.centroid.coords).reshape(1, 2), np.array(poly.exterior.xy).T
    ).mean()


def span(sdata: SpatialData, shape_key: str, recompute: bool = False) -> None:
    """Compute maximum diameter of each shape.

    The span is the length of the longest line segment that fits within the shape.

    Parameters
    ----------
    sdata : SpatialData
        Input SpatialData object
    shape_key : str
        Key in sdata.shapes containing shape geometries
    recompute : bool, default False
        Whether to force recomputation if feature exists

    Returns
    -------
    None
        Updates sdata.shapes[shape_key] with:
        - '{shape_key}_span': Maximum diameter of each shape
    """

    feature_key = f"{shape_key}_span"

    if feature_key in sdata.shapes[shape_key].keys() and not recompute:
        return

    def get_span(poly):
        if not poly:
            return np.nan

        shape_coo = np.array(poly.coords.xy).T
        return int(distance_matrix(shape_coo, shape_coo).max())

    span = get_shape(sdata, shape_key, sync=False).exterior.apply(get_span)
    set_shape_metadata(
        sdata=sdata, shape_key=shape_key, metadata=span, column_names=feature_key
    )


def list_shape_features() -> Dict[str, str]:
    """List available shape feature calculations.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping feature names to their descriptions
    """

    # Get shape feature descriptions from docstrings
    df = dict()
    for k, v in shape_features.items():
        description = v.__doc__.split("Parameters")[0].strip()
        df[k] = description

    return df


shape_features = dict(
    area=area,
    aspect_ratio=aspect_ratio,
    bounds=bounds,
    density=density,
    opening=opening,
    perimeter=perimeter,
    radius=radius,
    raster=raster,
    second_moment=second_moment,
    span=span,
)


def shape_stats(
    sdata: SpatialData,
    feature_names: List[str] = ["area", "aspect_ratio", "density"]
) -> None:
    """Compute common shape statistics.

    Wrapper around analyze_shapes() for frequently used features.

    Parameters
    ----------
    sdata : SpatialData
        Input SpatialData object
    feature_names : List[str], default ["area", "aspect_ratio", "density"]
        Features to compute

    Returns
    -------
    None
        Updates sdata.shapes with computed features
    """

    # Compute features
    analyze_shapes(sdata, "cell_boundaries", feature_names)
    if "nucleus_boundaries" in sdata.shapes.keys():
        analyze_shapes(sdata, "nucleus_boundaries", feature_names)


def analyze_shapes(
    sdata: SpatialData,
    shape_keys: Union[str, List[str]],
    feature_names: Union[str, List[str]],
    feature_kws: Optional[Dict[str, Dict]] = None,
    recompute: bool = False,
    progress: bool = True,
) -> None:
    """Compute multiple shape features.

    Parameters
    ----------
    sdata : SpatialData
        Input SpatialData object
    shape_keys : str or list of str
        Keys in sdata.shapes to analyze
    feature_names : str or list of str
        Names of features to compute
    feature_kws : dict, optional
        Additional keyword arguments for each feature
    recompute : bool, default False
        Whether to force recomputation if features exist
    progress : bool, default True
        Whether to show progress bar

    Returns
    -------
    None
        Updates sdata.shapes with computed features
    """

    # Cast to list if not already
    if isinstance(shape_keys, str):
        shape_keys = [shape_keys]

    # Cast to list if not already
    if isinstance(feature_names, str):
        feature_names = [feature_names]

    # Generate feature x shape combinations
    combos = [(f, s) for f in feature_names for s in shape_keys]

    # Set up progress bar
    if progress:
        combos = tqdm(combos)

    # Analyze each feature x shape combination
    for feature, shape in combos:
        kws = dict(recompute=recompute)
        if feature_kws and feature in feature_kws:
            kws.update(feature_kws[feature])

        shape_features[feature](sdata, shape, **kws)


def register_shape_feature(name: str, func: Callable[[SpatialData, str], None]) -> None:
    """Register a new shape feature calculation function.

    Parameters
    ----------
    name : str
        Name to register the feature as
    func : Callable[[SpatialData, str], None]
        Function that takes SpatialData and shape_key as arguments
        and modifies SpatialData in-place

    Returns
    -------
    None
        Updates global shape_features dictionary
    """
    shape_features[name] = func

    # TODO perform some checks on the function

    print(f"Registered shape feature '{name}' to `bento.tl.shape_features`.")
