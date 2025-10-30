import numpy as np
import pandas as pd
from spatialdata import SpatialData

from bento._logging import logger
from ._measure import measure


def _ripley(points: pd.DataFrame, shape) -> dict:
    from astropy.stats import RipleysKEstimator
    from scipy.spatial import distance_matrix
    from scipy.stats import spearmanr

    if not shape or len(points) < 2:
        return {"l_max": np.nan, "l_max_gradient": np.nan, "l_min_gradient": np.nan, "l_monotony": np.nan, "l_half_radius": np.nan}

    shape_coo = np.array(shape.exterior.coords.xy).T
    cell_span = distance_matrix(shape_coo, shape_coo).max()
    cell_minx, cell_miny, cell_maxx, cell_maxy = shape.bounds
    cell_area = shape.area
    estimator = RipleysKEstimator(area=cell_area, x_min=cell_minx, y_min=cell_miny, x_max=cell_maxx, y_max=cell_maxy)
    quarter_span = cell_span / 4
    radii = np.linspace(1, quarter_span * 2, num=int(quarter_span * 2))
    points_geo = points[["x", "y"]].values
    stats = estimator.Hfunction(data=points_geo, radii=radii, mode="none")
    l_max = max(stats)
    ripley_smooth = pd.Series(stats).rolling(5).mean()
    ripley_smooth.dropna(inplace=True)
    if len(ripley_smooth) < 2:
        ripley_smooth = np.array([0, 0])
    ripley_gradient = np.gradient(ripley_smooth)
    l_max_gradient = ripley_gradient.max()
    l_min_gradient = ripley_gradient.min()
    l_monotony = spearmanr(radii, stats)[0]
    l_half_radius = estimator.Hfunction(data=points_geo, radii=[quarter_span], mode="none")[0]
    return {"l_max": l_max, "l_max_gradient": l_max_gradient, "l_min_gradient": l_min_gradient, "l_monotony": l_monotony, "l_half_radius": l_half_radius}


def ripley(
    sdata: SpatialData,
    points_key: str = "transcripts",
    shape_key: str = "cell_boundaries",
    feature_key: str = "feature_name",
    table_key: str = "table",
    n_jobs: int = 1,
    recompute: bool = True,
    progress: bool = True,
    leave: bool = True,
) -> None:
    if table_key not in sdata.tables:
        logger.error(f"Table '{table_key}' not found. Please run bt.io.prep() first to create the base table.")
        return

    ripley_layers = ["ripley_k", "ripley_l"]
    table = sdata.tables[table_key]
    if not recompute and any(layer in table.layers for layer in ripley_layers):
        logger.info("Ripley layers already exist. Set recompute=True to recalculate.")
        return

    temp_table_key = f"temp_{table_key}_ripley"

    measure(
        sdata=sdata,
        func=_ripley,
        points_key=points_key,
        shape_key=shape_key,
        feature_key=feature_key,
        result_key=temp_table_key,
        n_jobs=n_jobs,
        recompute=recompute,
        progress=progress,
        leave=leave,
    )

    if temp_table_key in sdata.tables:
        temp_table = sdata.tables[temp_table_key]
        if len(temp_table.layers) > 0:
            for layer_name, layer_data in temp_table.layers.items():
                result_matrix = layer_data.toarray() if hasattr(layer_data, "toarray") else layer_data
                aligned_matrix = np.full((len(table.obs_names), len(table.var_names)), np.nan)
                obs_mapping = {name: i for i, name in enumerate(table.obs_names)}
                var_mapping = {name: i for i, name in enumerate(table.var_names)}
                for i, obs_name in enumerate(temp_table.obs_names):
                    if obs_name in obs_mapping:
                        for j, var_name in enumerate(temp_table.var_names):
                            if var_name in var_mapping:
                                aligned_matrix[obs_mapping[obs_name], var_mapping[var_name]] = result_matrix[i, j]
                final_layer_name = layer_name if layer_name.startswith("ripley_") else f"ripley_{layer_name}"
                table.layers[final_layer_name] = aligned_matrix
        else:
            result_matrix = temp_table.X.toarray() if hasattr(temp_table.X, "toarray") else temp_table.X
            aligned_matrix = np.full((len(table.obs_names), len(table.var_names)), np.nan)
            obs_mapping = {name: i for i, name in enumerate(table.obs_names)}
            var_mapping = {name: i for i, name in enumerate(table.var_names)}
            for i, obs_name in enumerate(temp_table.obs_names):
                if obs_name in obs_mapping:
                    for j, var_name in enumerate(temp_table.var_names):
                        if var_name in var_mapping:
                            aligned_matrix[obs_mapping[obs_name], var_mapping[var_name]] = result_matrix[i, j]
            table.layers["ripley"] = aligned_matrix

        del sdata.tables[temp_table_key]
        logger.info(f"Ripley statistics saved as layers in table: {table_key}")


