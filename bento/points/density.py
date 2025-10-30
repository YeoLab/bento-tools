import numpy as np
from spatialdata import SpatialData

from bento._logging import logger


def density(
    sdata: SpatialData,
    points_key: str = "transcripts",
    feature_key: str = "feature_name",
    shape_key: str = "cell_boundaries",
    table_key: str = "table",
    recompute: bool = True,
    progress: bool = True,
    leave: bool = True,
) -> None:
    if table_key not in sdata.tables:
        logger.error(f"Table '{table_key}' not found. Please run bt.io.prep() first to create the base table.")
        return

    table = sdata.tables[table_key]
    if not recompute and "density" in table.layers:
        logger.info("Density layer already exists. Set recompute=True to recalculate.")
        return

    logger.info("Starting density computation...")

    count_matrix = table.X.toarray() if hasattr(table.X, "toarray") else table.X
    if table.obs["region"].iloc[0] != shape_key:
        logger.error(f"Table region '{table.obs['region'].iloc[0]}' doesn't match shape_key '{shape_key}'")
        return

    shapes = sdata.shapes[shape_key]
    n_shapes, n_features = count_matrix.shape
    if progress:
        logger.info(f"Computing density for {n_features} features across {n_shapes} shapes")

    from bento.shapes import area as _area

    area_col = "area"
    if area_col not in shapes.columns or recompute:
        _area(sdata=sdata, shape_key=shape_key, recompute=recompute)
        logger.info(f"Computed areas for {shapes.shape[0]} shapes")

    areas = shapes.loc[table.obs_names, area_col].values.reshape(-1, 1)
    invalid_areas = (areas <= 0) | np.isnan(areas)
    if np.any(invalid_areas):
        logger.warning(f"Found {np.sum(invalid_areas)} shapes with invalid areas (<=0 or NaN)")

    density_matrix = np.divide(
        count_matrix, areas, out=np.full_like(count_matrix, np.nan, dtype=float), where=~invalid_areas
    )

    table.layers["density"] = density_matrix
    logger.info(f"Density saved as layer. Non-zero densities: {np.sum(density_matrix > 0)}")


