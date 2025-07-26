import pandas as pd
import geopandas as gpd


def compute_polarity(
    points: gpd.GeoDataFrame,
    shapes: gpd.GeoDataFrame,
    shape_radii: pd.Series,
    shape_key: str,
    feature_key: str,
) -> pd.DataFrame:
    """
    Compute polarity values for each feature-shape combination using vectorized operations.

    Polarity measures the displacement of the center of mass from the shape centroid,
    normalized by the shape radius.

    Parameters
    ----------
    points : gpd.GeoSeries
        GeoSeries with point geometries
    shapes : gpd.GeoDataFrame
        GeoDataFrame with shape geometries
    shape_radii : pd.Series
        Series of radii for shapes
    shape_key : str
        Key for shapes
    feature_key : str
        Key for features

    Returns
    -------
    polarity : DataFrame
        DataFrame with shape_key, feature_key, and polarity

    Raises
    ------
    ValueError
        If points or shapes are not GeoSeries or GeoDataFrame
    """

    # Dissolve points to get centroids for each shape-feature combination
    # This gives a GeoSeries with MultiIndex (shape_key, feature_key)
    point_centroids = points.dissolve(by=[shape_key, feature_key], observed=True).centroid.reset_index()

    # Align shape centroids and radii to the shape_key in point_centroids_df
    shape_centroids = shapes.centroid.reindex(point_centroids[shape_key].values).reset_index(drop=True)
    shape_radii_aligned = shape_radii.reindex(point_centroids[shape_key].values).reset_index(drop=True)

    # Compute distances between point centroids and shape centroids, then normalize by radius
    polarity = point_centroids.geometry.distance(shape_centroids) / shape_radii_aligned

    # Prepare output DataFrame with shape_key, feature_key, and polarity
    result = pd.DataFrame(
        {
            shape_key: point_centroids[shape_key].values,
            feature_key: point_centroids[feature_key].values,
            "polarity": polarity.values,
        }
    )

    return result
