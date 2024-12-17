import warnings

import dask
import emoji
from shapely.errors import ShapelyDeprecationWarning

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

import re
from abc import ABCMeta, abstractmethod
from math import isnan
from typing import List, Optional, Union, Type, Dict

import numpy as np
import pandas as pd
from astropy.stats.spatial import RipleysKEstimator
from scipy.spatial import distance
from scipy.stats import spearmanr
from spatialdata._core.spatialdata import SpatialData
from tqdm.dask import TqdmCallback

from .. import tools as tl
from .._utils import get_points


def analyze_points(
    sdata: SpatialData,
    shape_keys: Union[str, List[str]],
    feature_names: Union[str, List[str]],
    points_key: str = "transcripts",
    instance_key: str = "cell_boundaries",
    groupby: Optional[Union[str, List[str]]] = None,
    recompute: bool = False,
    progress: bool = False,
    num_workers: int = 1,
) -> None:
    """Calculate features for point groups within cells.

    Efficiently avoids recomputing cell-level features by compiling and computing
    once the set of required cell-level features and attributes for each feature.

    Parameters
    ----------
    sdata : SpatialData
        Input SpatialData object
    shape_keys : str or list of str
        Names of shapes to analyze in sdata.shapes
    feature_names : str or list of str
        Names of features to compute; list available features with `bt.tl.list_point_features()`
    points_key : str, default "transcripts"
        Key for points in sdata.points
    instance_key : str, default "cell_boundaries"
        Key for cell boundaries in sdata.shapes
    groupby : str or list of str, optional
        Column(s) in sdata.points[points_key] to group by
    recompute : bool, default False
        Whether to force recomputation of features
    progress : bool, default False
        Whether to show progress bars
    num_workers : int, default 1
        Number of parallel workers

    Returns
    -------
    None
        Updates sdata.tables["table"].uns["{instance_key}_{groupby}_features"] with computed features

    Raises
    ------
    KeyError
        If required shape keys or groupby columns are not found
    """

    # Cast to list if not already
    if isinstance(shape_keys, str):
        shape_keys = [shape_keys]

    # Cast to list if not already
    if isinstance(feature_names, str):
        feature_names = [feature_names]

    # Make sure groupby is a list
    if isinstance(groupby, str):
        groupby = [f"{instance_key}_index", groupby]
    elif isinstance(groupby, list):
        groupby = [f"{instance_key}_index"] + groupby
    else:
        groupby = [f"{instance_key}_index"]

    # Make sure points are sjoined to all shapes in shape_keys
    shapes_found = set(shape_keys).intersection(set(sdata.points[points_key].columns))
    if len(shapes_found) != len(shape_keys):
        raise KeyError(
            f"sdata.points[{points_key}] does not have all columns: {shape_keys}. Please run sjoin_points first."
        )

    # Make sure all groupby keys are in point columns
    for g in groupby:
        if (
            g != f"{instance_key}_index"
            and g
            not in get_points(
                sdata, points_key=points_key, astype="dask", sync=True
            ).columns
        ):
            raise ValueError(f"Groupby key {g} not found in point columns.")

    # Generate feature x shape combinations
    feature_combos = [
        point_features[f](instance_key, s) for f in feature_names for s in shape_keys
    ]

    # Compile dependency set of features and attributes
    cell_features = set()
    obs_attrs = set()
    for f in feature_combos:
        cell_features.update(f.cell_features)
        obs_attrs.update(f.attributes)

    cell_features = list(cell_features)
    obs_attrs = list(obs_attrs)

    tl.analyze_shapes(
        sdata=sdata,
        shape_keys=instance_key,
        feature_names=cell_features,
        progress=progress,
        recompute=recompute,
    )

    # Make sure points are sjoined to all shapes in shape_keys
    attributes = [attr for attr in obs_attrs if attr not in shape_keys]
    attributes.append(instance_key)
    attrs_found = set(attributes).intersection(
        set(sdata.shapes[instance_key].columns.tolist())
    )
    if len(attrs_found) != len(attributes):
        raise KeyError(f"df does not have all columns: {obs_attrs}.")

    points_df = get_points(
        sdata, points_key=points_key, astype="geopandas", sync=True
    ).set_index(instance_key)
    # Pull all shape polygons into the points dataframe
    for shape in list(
        set(obs_attrs).intersection(set([x for x in shape_keys if x != instance_key]))
    ):
        points_df = points_df.join(
            sdata.shapes[shape].set_index(instance_key),
            on=instance_key,
            lsuffix="",
            rsuffix=f"_{shape}",
        ).rename(columns={shape: f"{shape}_index", f"geometry_{shape}": shape})

    # Pull cell_boundaries shape features into the points dataframe
    points_df = (
        points_df.join(sdata.shapes[instance_key][attributes])
        .rename_axis(f"{instance_key}_index")
        .reset_index()
    )

    for g in groupby:
        points_df[g] = points_df[g].astype("category")

    # Calculate features for a sample
    def process_sample(df):
        sample_output = {}
        for f in feature_combos:
            sample_output.update(f.extract(df))
        return sample_output

    # Process all samples in a partition
    def process_partition(bag):
        df, groupby, process_sample = bag
        # Groupby by cell and groupby keys and process each sample
        out = df.groupby(groupby, observed=True).apply(process_sample)
        return pd.DataFrame.from_records(out.values, index=out.index)

    import dask.bag as db

    points_grouped = points_df.groupby(f"{instance_key}_index")
    cells = list(points_grouped.groups.keys())

    args = [(points_grouped.get_group(cell), groupby, process_sample) for cell in cells]
    bags = db.from_sequence(args)

    output = bags.map(process_partition)

    with TqdmCallback(desc="Batches"), dask.config.set(num_workers=num_workers):
        output = output.compute()

    output = pd.concat(output)

    groupby[groupby.index(f"{instance_key}_index")] = instance_key
    output_key = "_".join([*groupby, "features"])
    if output_key in sdata.tables["table"].uns:
        sdata.tables["table"].uns[output_key][output.columns] = output.reset_index(
            drop=True
        ).rename(columns={f"{instance_key}_index": instance_key})
    else:
        sdata.tables["table"].uns[output_key] = output.reset_index().rename(
            columns={f"{instance_key}_index": instance_key}
        )

    print(emoji.emojize("Done :bento_box:"))


class PointFeature(metaclass=ABCMeta):
    """Base class for point feature calculations.

    Parameters
    ----------
    instance_key : str
        Key for cell boundaries in sdata.shapes
    shape_key : str, optional
        Key for shape to analyze relative to

    Attributes
    ----------
    cell_features : set
        Required cell-level features
    attributes : set
        Required shape attributes
    """

    def __init__(self, instance_key: str, shape_key: Optional[str] = None):
        self.cell_features = set()
        self.attributes = set()
        self.instance_key = instance_key
        
        if shape_key:
            self.attributes.add(shape_key)
            self.shape_key = shape_key

    @abstractmethod
    def extract(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate features for a group of points.

        Parameters
        ----------
        df : pd.DataFrame
            Points data with required columns

        Returns
        -------
        Dict[str, float]
            Computed feature values
        """
        return df


class ShapeProximity(PointFeature):
    """Compute proximity of points relative to a shape boundary.

    Parameters
    ----------
    instance_key : str
        Key for cell boundaries in sdata.shapes
    shape_key : str
        Key for shape to analyze relative to

    Attributes
    ----------
    cell_features : set
        Required cell-level features
    attributes : set
        Required shape attributes

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - {shape_key}_inner_proximity: Proximity of points inside shape (0-1)
        - {shape_key}_outer_proximity: Proximity of points outside shape (0-1)
        Values closer to 0 indicate farther from boundary, closer to 1 indicate nearer
    """

    def __init__(self, instance_key: str, shape_key: str):
        super().__init__(instance_key, shape_key)
        self.cell_features.add("radius")
        self.attributes.add(f"{self.instance_key}_radius")

    def extract(self, df):
        df = super().extract(df)

        # Get shape polygon
        shape = df[self.shape_key].values[0]

        # Skip if no shape or if shape is nan
        try:
            if isnan(shape):
                return {
                    f"{self.shape_key}_inner_proximity": np.nan,
                    f"{self.shape_key}_outer_proximity": np.nan,
                }
        except:
            pass

        if not shape:
            return {
                f"{self.shape_key}_inner_proximity": np.nan,
                f"{self.shape_key}_outer_proximity": np.nan,
            }

        # Get points
        points_geo = df["geometry"]

        # Check for points within shape, assume all are intracellular
        if self.shape_key == self.instance_key:
            inner = np.array([True] * len(df))
        else:
            inner = df[f"{self.shape_key}_index"] != ""
        outer = ~inner

        inner_dist = np.nan
        outer_dist = np.nan

        if inner.sum() > 0:
            inner_dist = points_geo[inner].distance(shape.boundary).mean()

        if outer.sum() > 0:
            outer_dist = points_geo[outer].distance(shape.boundary).mean()

        # Scale from [0, 1], where 1 is close and 0 is far.
        cell_radius = df[f"{self.instance_key}_radius"].values[0]
        inner_proximity = (cell_radius - inner_dist) / cell_radius
        outer_proximity = (cell_radius - outer_dist) / cell_radius

        if np.isnan(inner_proximity):
            inner_proximity = 0

        if np.isnan(outer_proximity):
            outer_proximity = 0

        return {
            f"{self.shape_key}_inner_proximity": inner_proximity,
            f"{self.shape_key}_outer_proximity": outer_proximity,
        }


class ShapeAsymmetry(PointFeature):
    """Compute asymmetry of points relative to a shape centroid.

    Parameters
    ----------
    instance_key : str
        Key for cell boundaries in sdata.shapes
    shape_key : str
        Key for shape to analyze relative to


    Attributes
    ----------
    cell_features : set
        Required cell-level features
    attributes : set
        Required shape attributes

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - {shape_key}_inner_asymmetry: Asymmetry of points inside shape (0-1)
        - {shape_key}_outer_asymmetry: Asymmetry of points outside shape (0-1)
        Values closer to 0 indicate symmetry, closer to 1 indicate asymmetry
    """

    def __init__(self, instance_key: str, shape_key: str):
        super().__init__(instance_key, shape_key)
        self.cell_features.add("radius")
        self.attributes.add(f"{self.instance_key}_radius")

    def extract(self, df):
        df = super().extract(df)

        # Get shape polygon
        shape = df[self.shape_key].values[0]

        # Skip if no shape or shape is nan
        try:
            if isnan(shape):
                return {
                    f"{self.shape_key}_inner_asymmetry": np.nan,
                    f"{self.shape_key}_outer_asymmetry": np.nan,
                }
        except:
            pass

        if shape is None:
            return {
                f"{self.shape_key}_inner_asymmetry": np.nan,
                f"{self.shape_key}_outer_asymmetry": np.nan,
            }

        # Get points
        points_geo = df["geometry"]

        # Check for points within shape, assume all are intracellular
        if self.shape_key == self.instance_key:
            inner = np.array([True] * len(df))
        else:
            inner = df[f"{self.shape_key}_index"] != ""
        outer = ~inner

        inner_to_centroid = np.nan
        outer_to_centroid = np.nan

        if inner.sum() > 0:
            inner_to_centroid = points_geo[inner].distance(shape.centroid).mean()

        if outer.sum() > 0:
            outer_to_centroid = points_geo[outer].distance(shape.centroid).mean()

        # Values [0, 1], where 1 is asymmetrical and 0 is symmetrical.
        cell_radius = df[f"{self.instance_key}_radius"].values[0]
        inner_asymmetry = inner_to_centroid / cell_radius
        outer_asymmetry = outer_to_centroid / cell_radius

        if np.isnan(inner_asymmetry):
            inner_asymmetry = 0

        if np.isnan(outer_asymmetry):
            outer_asymmetry = 0

        return {
            f"{self.shape_key}_inner_asymmetry": inner_asymmetry,
            f"{self.shape_key}_outer_asymmetry": outer_asymmetry,
        }


class PointDispersionNorm(PointFeature):
    """Compute normalized dispersion of points relative to RNA signal centroid.

    Parameters
    ----------
    instance_key : str
        Key for cell boundaries in sdata.shapes
    shape_key : str
        Key for shape to analyze relative to

    Attributes
    ----------
    cell_features : set
        Required cell-level features
    attributes : set
        Required shape attributes

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - point_dispersion_norm: Second moment of points normalized by 
          second moment of uniform distribution
    """

    def __init__(self, instance_key: str, shape_key: str):
        super().__init__(instance_key, shape_key)
        self.cell_features.add("raster")
        self.attributes.add(f"{self.instance_key}_raster")

    def extract(self, df):
        df = super().extract(df)

        # Get precomputed cell centroid and raster
        pt_centroid = df[["x", "y"]].values.mean(axis=0).reshape(1, 2)
        cell_raster = df[f"{self.instance_key}_raster"].values[0]

        # Skip if no raster
        if not np.array(cell_raster).flatten().any():
            return {"point_dispersion_norm": np.nan}

        # calculate points moment
        point_moment = _second_moment(pt_centroid, df[["x", "y"]].values)
        cell_moment = _second_moment(pt_centroid, cell_raster)

        # Normalize by cell moment
        norm_moment = point_moment / cell_moment

        return {"point_dispersion_norm": norm_moment}


class ShapeDispersionNorm(PointFeature):
    """Compute normalized dispersion of points relative to a shape centroid.

    Parameters
    ----------
    instance_key : str
        Key for cell boundaries in sdata.shapes
    shape_key : str
        Key for shape to analyze relative to

    Attributes
    ----------
    cell_features : set
        Required cell-level features
    attributes : set
        Required shape attributes

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - {shape_key}_dispersion_norm: Second moment of points normalized by
          second moment of uniform distribution
    """

    def __init__(self, instance_key: str, shape_key: str):
        super().__init__(instance_key, shape_key)
        self.cell_features.add("raster")
        self.attributes.add(f"{self.instance_key}_raster")

    def extract(self, df):
        df = super().extract(df)

        # Get shape polygon
        shape = df[self.shape_key].values[0]

        # Skip if no shape or if shape is nan
        try:
            if isnan(shape):
                return {f"{self.shape_key}_dispersion_norm": np.nan}
        except:
            pass

        if not shape:
            return {f"{self.shape_key}_dispersion_norm": np.nan}

        # Get precomputed shape centroid and raster
        cell_raster = df[f"{self.instance_key}_raster"].values[0]

        # calculate points moment
        point_moment = _second_moment(shape.centroid, df[["x", "y"]].values)
        cell_moment = _second_moment(shape.centroid, cell_raster)

        # Normalize by cell moment
        norm_moment = point_moment / cell_moment

        return {f"{self.shape_key}_dispersion_norm": norm_moment}


class ShapeDistance(PointFeature):
    """Compute absolute distances between points and a shape boundary.

    Parameters
    ----------
    instance_key : str
        Key for cell boundaries in sdata.shapes
    shape_key : str
        Key for shape to analyze relative to

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - {shape_key}_inner_distance: Mean distance of points inside shape to boundary
        - {shape_key}_outer_distance: Mean distance of points outside shape to boundary
    """

    def __init__(self, instance_key: str, shape_key: str):
        super().__init__(instance_key, shape_key)

    def extract(self, df):
        df = super().extract(df)

        # Get shape polygon
        shape = df[self.shape_key].values[0]

        # Skip if no shape or if shape is nan
        try:
            if isnan(shape):
                return {
                    f"{self.shape_key}_inner_distance": np.nan,
                    f"{self.shape_key}_outer_distance": np.nan,
                }
        except:
            pass

        if not shape:
            return {
                f"{self.shape_key}_inner_distance": np.nan,
                f"{self.shape_key}_outer_distance": np.nan,
            }

        # Get points
        points_geo = df["geometry"].values

        # Check for points within shape, assume all are intracellular
        if self.shape_key == self.instance_key:
            inner = np.array([True] * len(df))
        else:
            inner = df[f"{self.shape_key}_index"] != ""
        outer = ~inner

        if inner.sum() > 0:
            inner_dist = points_geo[inner].distance(shape.boundary).mean()
        else:
            inner_dist = np.nan

        if outer.sum() > 0:
            outer_dist = points_geo[outer].distance(shape.boundary).mean()
        else:
            outer_dist = np.nan

        return {
            f"{self.shape_key}_inner_distance": inner_dist,
            f"{self.shape_key}_outer_distance": outer_dist,
        }


class ShapeOffset(PointFeature):
    """Compute distances between point centroids and a shape centroid.

    Parameters
    ----------
    instance_key : str
        Key for cell boundaries in sdata.shapes
    shape_key : str
        Key for shape to analyze relative to

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - {shape_key}_inner_offset: Mean distance from inner points to shape centroid
        - {shape_key}_outer_offset: Mean distance from outer points to shape centroid
    """

    def __init__(self, instance_key: str, shape_key: str):
        super().__init__(instance_key, shape_key)

    def extract(self, df):
        df = super().extract(df)

        # Get shape polygon
        shape = df[self.shape_key].values[0]

        # Skip if no shape
        try:
            if isnan(shape):
                return {
                    f"{self.shape_key}_inner_offset": np.nan,
                    f"{self.shape_key}_outer_offset": np.nan,
                }
        except:
            pass

        if not shape:
            return {
                f"{self.shape_key}_inner_offset": np.nan,
                f"{self.shape_key}_outer_offset": np.nan,
            }

        # Get points
        points_geo = df["geometry"].values

        # Check for points within shape, assume all are intracellular
        if self.shape_key == self.instance_key:
            inner = np.array([True] * len(df))
        else:
            inner = df[f"{self.shape_key}_index"] != ""
        outer = ~inner

        if inner.sum() > 0:
            inner_to_centroid = points_geo[inner].distance(shape.centroid).mean()
        else:
            inner_to_centroid = np.nan

        if outer.sum() > 0:
            outer_to_centroid = points_geo[outer].distance(shape.centroid).mean()
        else:
            outer_to_centroid = np.nan

        return {
            f"{self.shape_key}_inner_offset": inner_to_centroid,
            f"{self.shape_key}_outer_offset": outer_to_centroid,
        }


class PointDispersion(PointFeature):
    """Compute second moment of points relative to RNA signal centroid.

    Parameters
    ----------
    instance_key : str
        Key for cell boundaries in sdata.shapes
    shape_key : Optional[str]
        Not used, included for API consistency

    Attributes
    ----------
    cell_features : set
        Required cell-level features
    attributes : set
        Required shape attributes

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - point_dispersion: Second moment of points relative to RNA centroid
    """

    def __init__(self, instance_key: str, shape_key: Optional[str] = None):
        super().__init__(instance_key, shape_key)

    def extract(self, df):
        df = super().extract(df)

        # Get precomputed cell centroid and raster
        pt_centroid = df[["x", "y"]].values.mean(axis=0).reshape(1, 2)

        # calculate points moment
        point_moment = _second_moment(pt_centroid, df[["x", "y"]].values)

        return {"point_dispersion": point_moment}


class ShapeDispersion(PointFeature):
    """Compute second moment of points relative to a shape centroid.

    Parameters
    ----------
    instance_key : str
        Key for cell boundaries in sdata.shapes
    shape_key : str
        Key for shape to analyze relative to

    Attributes
    ----------
    cell_features : set
        Required cell-level features
    attributes : set
        Required shape attributes

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - {shape_key}_dispersion: Second moment of points relative to shape centroid
    """

    def __init__(self, instance_key: str, shape_key: str):
        super().__init__(instance_key, shape_key)

    def extract(self, df):
        df = super().extract(df)

        # Get shape polygon
        shape = df[self.shape_key].values[0]

        # Skip if no shape or if shape is nan
        try:
            if isnan(shape):
                return {f"{self.shape_key}_dispersion": np.nan}
        except:
            pass

        if not shape:
            return {f"{self.shape_key}_dispersion": np.nan}

        # calculate points moment
        point_moment = _second_moment(shape.centroid, df[["x", "y"]].values)

        return {f"{self.shape_key}_dispersion": point_moment}


class RipleyStats(PointFeature):
    """Compute Ripley's L-function statistics for point patterns.

    The L-function is evaluated at r=[1,d], where d is half the cell's maximum diameter.

    Parameters
    ----------
    instance_key : str
        Key for cell boundaries in sdata.shapes
    shape_key : Optional[str]
        Not used, included for API consistency

    Attributes
    ----------
    cell_features : set
        Required cell-level features
    attributes : set
        Required shape attributes

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - l_max: Maximum value of L-function 
        - l_max_gradient: Maximum gradient of L-function
        - l_min_gradient: Minimum gradient of L-function
        - l_monotony: Spearman correlation between L-function and radius
        - l_half_radius: L-function value at quarter cell diameter
    """

    def __init__(self, instance_key: str, shape_key: Optional[str] = None):
        super().__init__(instance_key, shape_key)
        self.cell_features.update(["span", "bounds", "area"])
        self.attributes.update([f"{instance_key}_span", f"{instance_key}_minx", 
                              f"{instance_key}_miny", f"{instance_key}_maxx",
                              f"{instance_key}_maxy", f"{instance_key}_area"])

    def extract(self, df):
        df = super().extract(df)

        # Get precomputed centroid and cell moment
        cell_span = df[f"{self.instance_key}_span"].values[0]
        cell_minx = df[f"{self.instance_key}_minx"].values[0]
        cell_miny = df[f"{self.instance_key}_miny"].values[0]
        cell_maxx = df[f"{self.instance_key}_maxx"].values[0]
        cell_maxy = df[f"{self.instance_key}_maxy"].values[0]
        cell_area = df[f"{self.instance_key}_area"].values[0]

        estimator = RipleysKEstimator(
            area=cell_area,
            x_min=cell_minx,
            y_min=cell_miny,
            x_max=cell_maxx,
            y_max=cell_maxy,
        )

        quarter_span = cell_span / 4
        radii = np.linspace(1, quarter_span * 2, num=int(quarter_span * 2))

        # Get points
        points_geo = df["geometry"].values
        points_geo = np.array([points_geo.x, points_geo.y]).T

        # Compute ripley function stats
        stats = estimator.Hfunction(data=points_geo, radii=radii, mode="none")

        # Max value of the L-function
        l_max = max(stats)

        # Max and min value of the gradient of L
        ripley_smooth = pd.Series(stats).rolling(5).mean()
        ripley_smooth.dropna(inplace=True)

        # Can't take gradient of single number
        if len(ripley_smooth) < 2:
            ripley_smooth = np.array([0, 0])

        ripley_gradient = np.gradient(ripley_smooth)
        l_max_gradient = ripley_gradient.max()
        l_min_gradient = ripley_gradient.min()

        # Monotony of L-function in the interval
        l_monotony = spearmanr(radii, stats)[0]

        # L-function at L/4 where length of the cell L is max dist between 2 points on polygon defining cell border
        l_half_radius = estimator.Hfunction(
            data=points_geo, radii=[quarter_span], mode="none"
        )[0]

        result = {
            "l_max": l_max,
            "l_max_gradient": l_max_gradient,
            "l_min_gradient": l_min_gradient,
            "l_monotony": l_monotony,
            "l_half_radius": l_half_radius,
        }

        return result


class ShapeEnrichment(PointFeature):
    """Compute fraction of points within a shape boundary.

    Parameters
    ----------
    instance_key : str
        Key for cell boundaries in sdata.shapes
    shape_key : str
        Key for shape to analyze relative to

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - {shape_key}_enrichment: Fraction of points inside shape (0-1)
    """

    def __init__(self, instance_key: str, shape_key: str):
        super().__init__(instance_key, shape_key)

    def extract(self, df):
        df = super().extract(df)

        # Get points outside shape
        points_geo = df["geometry"]

        # Check for points within shape, assume all are intracellular
        if self.shape_key == self.instance_key:
            enrichment = 1.0
        else:
            inner_count = (df[f"{self.shape_key}_index"] != "").sum()
            enrichment = inner_count / float(len(points_geo))

        return {f"{self.shape_key}_enrichment": enrichment}


def _second_moment(centroid: np.ndarray, pts: np.ndarray) -> float:
    """Calculate second moment of points relative to a centroid.

    Parameters
    ----------
    centroid : np.ndarray
        Reference point coordinates, shape (1, 2)
    pts : np.ndarray
        Point coordinates, shape (n, 2)

    Returns
    -------
    float
        Second moment value
    """
    if type(centroid) is not np.ndarray:
        centroid = centroid.coords
    centroid = np.array(centroid).reshape(1, 2)
    radii = distance.cdist(centroid, pts)
    second_moment = np.sum(radii * radii / len(pts))
    return second_moment


def list_point_features() -> pd.DataFrame:
    """List available point feature calculations.

    Returns
    -------
    pd.DataFrame
        DataFrame with feature names as index and descriptions from docstrings
    """

    # Get point feature descriptions from docstrings
    df = dict()
    for k, v in point_features.items():
        description = v.__doc__.split("Attributes")[0].strip()
        description = re.sub("\s +", " ", description)
        df[k] = description

    return df


point_features = dict(
    proximity=ShapeProximity,
    asymmetry=ShapeAsymmetry,
    point_dispersion_norm=PointDispersionNorm,
    shape_dispersion_norm=ShapeDispersionNorm,
    distance=ShapeDistance,
    offset=ShapeOffset,
    point_dispersion=PointDispersion,
    shape_dispersion=ShapeDispersion,
    ripley=RipleyStats,
    shape_enrichment=ShapeEnrichment,
)


def register_point_feature(name: str, FeatureClass: Type[PointFeature]) -> None:
    """Register a new point feature calculation class.

    Parameters
    ----------
    name : str
        Name to register the feature as
    FeatureClass : Type[PointFeature]
        Class that extends PointFeature base class

    Returns
    -------
    None
        Updates global point_features dictionary
    """

    point_features[name] = FeatureClass

    print(f"Registered point feature '{name}' to `bento.tl.shape_features`.")
