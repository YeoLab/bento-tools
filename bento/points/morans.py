import numpy as np
import pandas as pd
from spatialdata import SpatialData

from bento._logging import logger
from ._measure import measure


def _morans_i(points: pd.DataFrame, shape, attribute: str = None, k: int = 8) -> dict:
    from scipy.spatial import distance_matrix
    from sklearn.neighbors import NearestNeighbors
    from scipy.stats import norm

    if len(points) < 3:
        return {"morans_i": np.nan, "morans_p": np.nan, "morans_z": np.nan}

    coords = points[["x", "y"]].values
    n = len(coords)
    distances = distance_matrix(coords, coords)
    weights = np.zeros((n, n))
    for i in range(n):
        neighbor_indices = np.argsort(distances[i])[1 : k + 1]
        weights[i, neighbor_indices] = 1
    weights = (weights + weights.T) > 0
    weights = weights.astype(float)
    row_sums = weights.sum(axis=1)
    row_sums[row_sums == 0] = 1
    weights = weights / row_sums[:, np.newaxis]
    if attribute and attribute in points.columns:
        values = points[attribute].values
    else:
        nbrs = NearestNeighbors(n_neighbors=min(k, n - 1)).fit(coords)
        distances_nn, _ = nbrs.kneighbors(coords)
        values = 1 / (distances_nn.mean(axis=1) + 1e-10)
    mean_val = np.mean(values)
    deviations = values - mean_val
    numerator = 0
    for i in range(n):
        for j in range(n):
            numerator += weights[i, j] * deviations[i] * deviations[j]
    denominator = np.sum(deviations**2)
    w_sum = np.sum(weights)
    if denominator == 0 or w_sum == 0:
        return {"morans_i": np.nan, "morans_p": np.nan, "morans_z": np.nan}
    morans_i_val = (n / w_sum) * (numerator / denominator)
    expected_i = -1 / (n - 1)
    s2 = denominator / n
    b2 = (np.sum(deviations**4) / n) / (s2**2)
    w2_sum = np.sum(weights**2)
    var_i = ((n * ((n**2 - 3 * n + 3) * w2_sum - n * w_sum + 3 * (w_sum**2)) - b2 * ((n**2 - n) * w2_sum - 2 * n * w_sum + 6 * (w_sum**2))) / ((n - 1) * (n - 2) * (n - 3) * (w_sum**2))) - expected_i**2
    if var_i <= 0:
        z_score = np.nan
        p_value = np.nan
    else:
        z_score = (morans_i_val - expected_i) / np.sqrt(var_i)
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
    return {"morans_i": morans_i_val, "morans_p": p_value, "morans_z": z_score}


def morans_i(
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

    table = sdata.tables[table_key]
    if not recompute and "morans_i" in table.layers:
        logger.info("Moran's I layer already exists. Set recompute=True to recalculate.")
        return

    temp_table_key = f"temp_{table_key}_morans"

    measure(
        sdata=sdata,
        func=_morans_i,
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
        result_matrix = temp_table.X.toarray() if hasattr(temp_table.X, "toarray") else temp_table.X
        aligned_matrix = np.full((len(table.obs_names), len(table.var_names)), np.nan)
        obs_mapping = {name: i for i, name in enumerate(table.obs_names)}
        var_mapping = {name: i for i, name in enumerate(table.var_names)}
        for i, obs_name in enumerate(temp_table.obs_names):
            if obs_name in obs_mapping:
                for j, var_name in enumerate(temp_table.var_names):
                    if var_name in var_mapping:
                        aligned_matrix[obs_mapping[obs_name], var_mapping[var_name]] = result_matrix[i, j]
        table.layers["morans_i"] = aligned_matrix
        del sdata.tables[temp_table_key]
        logger.info("Moran's I saved as layer: morans_i")


