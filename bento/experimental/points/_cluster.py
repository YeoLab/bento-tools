# Perform unsupervised clustering on table(s) of features
# Save cluster labels to table.obs

import spatialdata as sd
import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from ..._logging import logger


# pseudocode
# get table(s) of features
# concat into single sample x feature table
# look up molecule count for each sample from sdata['table']
# samples to compute on = samples molecule count > 0
# training set = samples molecule count >= 5
# standard scale features, fit only on training set
# neighbors
# pca
# leiden clustering
# save cluster labels to table layer


def cluster(
    sdata: sd.SpatialData,
    table_key: str,
    count_table_key: str,
    points_key: str,
    min_count: int = 5,
    resolution: float = 1,
    result_key: str = "cluster_labels",
):
    """Cluster points in table.

    Parameters
    ----------
    sdata : sd.SpatialData
        SpatialData object
    table_key : str
        Key to table
    count_table_key : str
        Key to count table
    points_key : str
        Key to points table
    min_count : int, optional
        Minimum number of molecules per sample, by default 5
    result_key : str, optional
        Key to save cluster labels in sdata[table_key].obs
    """
    table = sdata[table_key]
    count_mask = sdata[count_table_key].to_df().reindex_like(table.to_df()) < min_count

    X = []
    for layer in table.layers:
        if layer == result_key or layer == "cluster_labels":
            continue
        layer_df = table.to_df(layer=layer).copy()
        logger.info(f"table[{table_key}][{layer}]: {layer_df.shape} - valid: {(~layer_df.isna()).sum().sum()}")
        layer_df[count_mask] = np.nan
        layer_df = layer_df.reset_index(names="obs").melt(
            id_vars="obs", var_name="var", value_name=layer
        )
        if len(X) == 0:
            X.append(layer_df)  # this includes obs, var, and layer values
        else:
            X.append(layer_df[[layer]])  # this includes just layer values

    X = pd.concat(X, axis=1)
    logger.info(f"X: {X.shape}")

    X = X.dropna()
    logger.info(f"X: {X.shape}")

    feature_names = X.columns.drop(["obs", "var"]).tolist()
    logger.info(f"Feature names: {feature_names}")

    # standard scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[feature_names])

    # quantile scale features
    # scaler = QuantileTransformer(output_distribution="normal")
    # X[feature_names] = scaler.fit_transform(X[feature_names])

    adata = sc.AnnData(X_scaled, obs=X[["obs", "var"]])
    adata.var_names = feature_names

    # neighbors
    sc.pp.neighbors(adata)
    sc.tl.pca(adata)
    sc.tl.leiden(adata, resolution=resolution, key_added=result_key)
    logger.info(f"Clusters found: {adata.obs[result_key].nunique()}")

    # save cluster labels to table layer
    sdata[table_key].layers[result_key] = adata.obs.pivot(
        index="obs", columns="var", values=result_key
    ).reindex_like(table.to_df())

    # Save cluster labels to points table
    attrs = sdata[points_key].attrs
    points = (
        sdata[points_key].compute().drop(columns=["cluster_labels"], errors="ignore")
    )
    adata.obs.rename(
        columns={"obs": "cell_boundaries", "var": "feature_name"}, inplace=True
    )
    points = (
        points.set_index(["cell_boundaries", "feature_name"])
        .join(adata.obs.set_index(["cell_boundaries", "feature_name"]))
        .reset_index()
    )
    points["cluster_labels"] = (
        points["cluster_labels"]
        .astype(float)
        .fillna(-1)
        .astype(int)
        .astype(str)
        .astype("category")
    )
    sdata[points_key] = sd.models.PointsModel.parse(points)
    sdata[points_key].attrs = attrs
    logger.info(f"Clusters saved to sdata[{table_key}].layers[{result_key}]")

    return adata


# spatial plot the clustering results
# for each cluster
# - Get points corresponding to cellxgene sample label
# - Rasterize points + gaussian blur
# - save to sdata as image with cluster labels as channels
# - save summary image as max projection of channels; pixel value = top cluster label
