import pandas as pd
import geopandas as gpd
from spatialdata import SpatialData

from bento._logging import logger
from bento._utils import get_points


def _compute_polarity(points: gpd.GeoDataFrame, shapes: gpd.GeoDataFrame, shape_radii: pd.Series, shape_key: str, feature_key: str) -> pd.DataFrame:
    point_centroids = points.dissolve(by=[shape_key, feature_key], observed=True).centroid.reset_index()
    shape_centroids = shapes.centroid.reindex(point_centroids[shape_key].values).reset_index(drop=True)
    shape_radii_aligned = shape_radii.reindex(point_centroids[shape_key].values).reset_index(drop=True)
    polarity = point_centroids.geometry.distance(shape_centroids) / shape_radii_aligned
    result = pd.DataFrame(
        {shape_key: point_centroids[shape_key].values, feature_key: point_centroids[feature_key].values, "polarity": polarity.values}
    )
    return result


def polarity(
    sdata: SpatialData,
    points_key: str = "transcripts",
    shape_key: str = "cell_boundaries",
    feature_key: str = "feature_name",
    table_key: str = "table",
    recompute: bool = True,
    progress: bool = True,
) -> None:
    if table_key not in sdata.tables:
        logger.error(f"Table '{table_key}' not found. Please run bt.io.prep() first to create the base table.")
        return

    table = sdata.tables[table_key]
    if not recompute and "polarity" in table.layers:
        logger.info("Polarity layer already exists. Set recompute=True to recalculate.")
        return

    logger.info("Starting polarity computation...")

    from bento.shapes import radius as _radius

    points = get_points(sdata, points_key=points_key, astype="geopandas")
    shapes = sdata.shapes[shape_key]

    if "radius" not in shapes.columns or recompute:
        _radius(sdata, shape_key=shape_key, recompute=recompute)

    shape_radii = shapes["radius"]
    polarity_values = _compute_polarity(points=points, shapes=shapes, shape_radii=shape_radii, shape_key=shape_key, feature_key=feature_key)

    polarity_df = polarity_values.pivot(index=shape_key, columns=feature_key, values="polarity")
    polarity_df = polarity_df.reindex(index=table.obs_names, columns=table.var_names, fill_value=0.0)
    table.layers["polarity"] = polarity_df.values
    logger.info("Polarity saved as layer: polarity")


