"""Analysis outputs preserve numerical counts and group identities."""

from pathlib import Path

import dask
import numpy as np
import pandas as pd
import pytest
import spatialdata as sd

import bento as bt


@pytest.fixture
def analysis_data():
    data = sd.read_zarr(Path(bt.__file__).parent / "datasets" / "six_cell_data.zarr")
    bt.io.prep(
        data,
        points_key="transcripts",
        feature_key="feature_name",
        instance_key="cell_boundaries",
        shape_keys=["cell_boundaries", "nucleus_boundaries"],
    )
    return data


def test_knn_flux_uses_requested_number_of_neighbors(analysis_data):
    # Keep worker calculations in this process for deterministic unit execution.
    with dask.config.set(scheduler="synchronous"):
        bt.tl.flux(analysis_data, method="knn", n_neighbors=3, res=0.2, random_state=11)
    points = bt.ut.get_points(analysis_data, "cell_boundaries_raster", sync=False)
    np.testing.assert_allclose(points["flux_counts"], 3)
    embeddings = points.filter(regex=r"^flux_embed_").to_numpy()
    assert embeddings.shape[0] == len(points)
    assert embeddings.shape[1] > 0
    assert np.isfinite(embeddings).all()
    assert set(analysis_data.tables["table"].uns["flux_genes"]) == set(
        analysis_data.tables["table"].var_names
    )
    before = points.copy(deep=True)
    bt.tl.flux(analysis_data, method="knn", n_neighbors=3, res=0.2)
    pd.testing.assert_frame_equal(
        before, bt.ut.get_points(analysis_data, "cell_boundaries_raster", sync=False)
    )


def test_point_dispersion_matches_per_cell_second_moment(analysis_data):
    points = bt.ut.get_points(analysis_data)
    expected = points.groupby("cell_boundaries", observed=True).apply(
        lambda frame: (
            np.square(
                frame[["x", "y"]].to_numpy() - frame[["x", "y"]].mean().to_numpy()
            )
            .sum(axis=1)
            .mean()
        )
    )
    with dask.config.set(scheduler="synchronous"):
        bt.tl.analyze_points(
            analysis_data,
            "cell_boundaries",
            "point_dispersion",
            groupby=None,
            progress=False,
        )
        result = analysis_data.tables["table"].uns["cell_boundaries_features"]
        measured = result.set_index("cell_boundaries")["point_dispersion"].reindex(
            expected.index
        )
        np.testing.assert_allclose(measured, expected)
        # Repeating the analysis updates the existing feature table without adding groups.
        bt.tl.analyze_points(
            analysis_data,
            "cell_boundaries",
            "point_dispersion",
            groupby=None,
            progress=False,
        )
    repeated = analysis_data.tables["table"].uns["cell_boundaries_features"]
    assert len(repeated) == len(expected)
    np.testing.assert_allclose(repeated["point_dispersion"], result["point_dispersion"])


def test_point_analysis_rejects_shapes_missing_from_points(analysis_data):
    with pytest.raises(KeyError, match="does not have all columns"):
        bt.tl.analyze_points(analysis_data, "missing_shape", "point_dispersion")


@pytest.mark.parametrize("radius", [None, 0.5])
def test_radius_flux_counts_match_euclidean_neighborhoods(analysis_data, radius):
    molecules = bt.ut.get_points(analysis_data)
    with dask.config.set(scheduler="synchronous"):
        bt.tl.flux(
            analysis_data, method="radius", radius=radius, res=0.2, random_state=11
        )
    mean_radius = analysis_data.shapes["cell_boundaries"][
        "cell_boundaries_radius"
    ].mean()
    effective_radius = mean_radius / 3 if radius is None else radius * mean_radius
    raster = bt.ut.get_points(analysis_data, "cell_boundaries_raster", sync=False)
    expected = []
    for _, pixel in raster.iterrows():
        cell_points = molecules[
            molecules["cell_boundaries"] == pixel["cell_boundaries"]
        ]
        offsets = cell_points[["x", "y"]].to_numpy() - [pixel["x"], pixel["y"]]
        expected.append((np.square(offsets).sum(axis=1) <= effective_radius**2).sum())
    np.testing.assert_array_equal(raster["flux_counts"], expected)
