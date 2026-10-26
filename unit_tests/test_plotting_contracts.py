"""Plot artists encode the requested coordinates, labels, and loadings."""

from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection, QuadMesh
import numpy as np
import pandas as pd
import pytest
import spatialdata as sd

import bento as bt


@pytest.fixture(scope="module")
def prepared_plot_data():
    path = Path(bt.__file__).parent / "datasets" / "six_cell_data.zarr"
    data = sd.read_zarr(path)
    return bt.io.prep(
        data,
        points_key="transcripts",
        feature_key="feature_name",
        instance_key="cell_boundaries",
        shape_keys=["cell_boundaries", "nucleus_boundaries"],
    )


@pytest.fixture
def plot_data(prepared_plot_data):
    return deepcopy(prepared_plot_data)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("hide_outside", [True, False])
def test_points_plot_preserves_coordinates(plot_data, hide_outside):
    expected = bt.ut.get_points(plot_data, sync=hide_outside)[["x", "y"]].to_numpy()
    _, ax = plt.subplots()
    bt.pl.points(plot_data, hide_outside=hide_outside, ax=ax, title="Coordinates")
    scatter = next(c for c in ax.collections if isinstance(c, PathCollection))
    np.testing.assert_allclose(scatter.get_offsets(), expected)
    assert ax.get_title() == "Coordinates"


def test_points_hue_subset_contains_only_selected_gene(plot_data):
    points = bt.ut.get_points(plot_data)
    gene = points["feature_name"].iloc[0]
    expected = points[points["feature_name"] == gene][["x", "y"]].to_numpy()
    _, ax = plt.subplots()
    bt.pl.points(plot_data, hue="feature_name", hue_order=[gene], ax=ax)
    scatter = next(c for c in ax.collections if isinstance(c, PathCollection))
    np.testing.assert_allclose(scatter.get_offsets(), expected)


@pytest.mark.parametrize("style", ["outline", "fill"])
def test_shape_layers_produce_paths_and_requested_title(plot_data, style):
    _, ax = plt.subplots()
    bt.pl.shapes(
        plot_data,
        shapes="cell_boundaries",
        color="red",
        color_style=style,
        hide_outside=False,
        title="Cells",
        ax=ax,
    )
    assert ax.get_title() == "Cells"
    assert sum(len(c.get_paths()) for c in ax.collections) == len(
        plot_data.shapes["cell_boundaries"]
    )


@pytest.mark.parametrize("kind", ["hist", "kde"])
def test_density_draws_data_in_supplied_axes(plot_data, kind):
    _, ax = plt.subplots()
    bt.pl.density(plot_data, kind=kind, ax=ax, title="Density")
    assert ax.get_title() == "Density"
    assert len(ax.collections) > 0


def test_points_save_nonempty_png(plot_data, tmp_path):
    path = tmp_path / "points.png"
    bt.pl.points(plot_data, fname=str(path))
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert path.stat().st_size > 1000


@pytest.mark.parametrize(
    "options",
    [
        {"n_top": None, "show_labels": True, "cluster": False},
        {"n_top": [None, 1], "show_labels": [True, True], "cluster": [False, False]},
    ],
)
def test_factor_heatmap_contains_requested_loadings(options):
    factors = [
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[0.2, 0.8], [0.9, 0.1], [0.3, 0.4]]),
    ]
    labels = {"layer": ["a", "b"], "gene": ["g1", "g2", "g3"]}
    bt.pl.factor(factors, labels, ["layer", "gene"], **options)
    mesh = next(c for c in plt.gcf().axes[0].collections if isinstance(c, QuadMesh))
    np.testing.assert_allclose(
        np.asarray(mesh.get_array()).ravel(), factors[0].T.ravel()
    )
    assert {t.get_text() for t in plt.gcf().axes[0].get_xticklabels()} == {"a", "b"}


def test_fluxmap_without_results_gives_actionable_message(plot_data, capsys):
    bt.pl.fluxmap(plot_data)
    assert "run bento.tl.fluxmap() first" in capsys.readouterr().out


@pytest.mark.parametrize(
    "self_pairs,expected_labels",
    [
        (False, ["a_b", "b_a"]),
        (True, ["a_a", "a_b", "b_a", "b_b"]),
        ("only", ["a", "b"]),
    ],
)
def test_colocation_heatmap_filters_self_pairs(plot_data, self_pairs, expected_labels):
    uns = plot_data.tables["table"].uns
    uns["factors"] = {
        2: [
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.ones((3, 2)),
            np.arange(8.0).reshape(4, 2),
        ]
    }
    uns["tensor_labels"] = {
        "layer": np.array(["x", "y"]),
        "cell": np.array(["c1", "c2", "c3"]),
        "pair": np.array(["a_a", "a_b", "b_a", "b_b"]),
    }
    uns["tensor_names"] = ["layer", "cell", "pair"]
    bt.pl.colocation(
        plot_data,
        rank=2,
        self_pairs=self_pairs,
        n_top=None,
        z_score=False,
        cluster=False,
        show_labels=True,
    )
    labels = [t.get_text() for t in plt.gcf().axes[2].get_yticklabels()]
    assert set(labels) == set(expected_labels)
    assert list(uns["tensor_labels"]["pair"]) == ["a_a", "a_b", "b_a", "b_b"]


@pytest.fixture
def raster_data(plot_data):
    from spatialdata.models import PointsModel

    cell = plot_data.shapes["cell_boundaries"].index[0]
    points = PointsModel.parse(
        pd.DataFrame(
            {
                "x": [0.0, 1.0, 0.0, 1.0],
                "y": [0.0, 0.0, 1.0, 1.0],
                "cell_boundaries": [cell] * 4,
                "flux_color": ["#ff0000", "#00ff00", "#0000ff", "#ffffff"],
                "score": [0.0, 1.0, 2.0, 3.0],
            }
        )
    )
    plot_data.points["cell_boundaries_raster"] = points
    return plot_data


@pytest.mark.parametrize("alpha", [False, True])
def test_flux_raster_encodes_rgb_channels(raster_data, alpha):
    _, ax = plt.subplots()
    bt.pl.flux(raster_data, alpha=alpha, hide_outside=False, shapes=[], ax=ax)
    image = np.asarray(ax.images[0].get_array())
    assert image.shape == (2, 2, 4 if alpha else 3)
    np.testing.assert_allclose(image[0, 0, :3], [1, 0, 0])
    np.testing.assert_allclose(image[1, 0, :3], [0, 0, 1])
    np.testing.assert_allclose(image[0, 1, :3], [0, 1, 0])
    if alpha:
        np.testing.assert_allclose(image[:, :, 3], 1)


def test_enrichment_raster_encodes_numeric_scores(raster_data):
    _, ax = plt.subplots()
    bt.pl.fe(raster_data, gs="score", hide_outside=False, shapes=[], ax=ax, cbar=True)
    np.testing.assert_allclose(
        np.asarray(ax.images[0].get_array()).squeeze(), [[0, 1], [2, 3]]
    )
    assert len(ax.child_axes) + len(ax.figure.axes) >= 2


@pytest.mark.parametrize("palette", ["tab10", {"fluxmap1": "red"}])
def test_fluxmap_legend_describes_result_shapes(plot_data, palette):
    plot_data.shapes["fluxmap1"] = plot_data.shapes["cell_boundaries"].copy()
    _, ax = plt.subplots()
    bt.pl.fluxmap(plot_data, palette=palette, ax=ax)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["fluxmap1"]


@pytest.mark.parametrize("dim_order", [None, "auto", ["cell", "nucleus", "cytoplasm"]])
@pytest.mark.parametrize("annotate", [None, 2, ["gene0", "gene2"]])
def test_composition_plot_labels_dimensions_and_requested_genes(
    plot_data, annotate, dim_order
):
    stats = pd.DataFrame(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.4, 0.5, 0.1],
            [0.3, 0.2, 0.5],
        ],
        index=[f"gene{i}" for i in range(5)],
        columns=["cell", "nucleus", "cytoplasm"],
    )
    stats["logcounts"] = [1.0, 2.0, 3.0, 4.0, 5.0]
    stats["cell_fraction"] = [10.0, 20.0, 30.0, 40.0, 50.0]
    plot_data.tables["table"].uns["comp_stats"] = stats
    bt.pl.comp(plot_data, annotate=annotate, adjust=False, dim_order=dim_order)
    labels = {t.get_text() for t in plt.gca().texts}
    assert {"cell", "nucleus", "cytoplasm"} <= labels
    if isinstance(annotate, list):
        assert set(annotate) <= labels
    if isinstance(annotate, int):
        assert len(labels & set(stats.index)) == annotate


def test_shape_statistics_plot_has_one_panel_per_statistic_and_shape(plot_data):
    bt.tl.shape_stats(plot_data)
    bt.pl.shape_stats(plot_data, rug=True)
    assert len(plt.gcf().axes) == 6
    assert all(ax.get_lines() for ax in plt.gcf().axes)


def test_pattern_combinations_plot_reports_sample_count(plot_data):
    from bento._constants import PATTERN_NAMES

    plot_data.tables["table"].uns["lp"] = pd.DataFrame(
        np.eye(5, dtype=int), columns=PATTERN_NAMES
    )
    bt.pl.lp_dist(plot_data, show_counts="%d")
    assert "5 samples" in plt.gcf()._suptitle.get_text()


def test_differential_pattern_plot_draws_reference_thresholds(plot_data):
    from bento._constants import PATTERN_NAMES

    plot_data.tables["table"].uns["diff_treatment"] = pd.DataFrame(
        {
            "log2fc": [-3.0, -1.0, 0.0, 1.0, 3.0],
            "-log10padj": [3.0, 2.0, 1.0, 2.0, 3.0],
            "pattern": PATTERN_NAMES,
            "phenotype": ["treated"] * 5,
        }
    )
    bt.pl.lp_diff_discrete(plot_data, phenotype="treatment")
    ax = plt.gcf().axes[0]
    vertical = [
        line.get_xdata()[0]
        for line in ax.lines
        if len(line.get_xdata()) == 2 and line.get_xdata()[0] == line.get_xdata()[1]
    ]
    assert set(vertical) == {-2, 0, 2}


def test_gene_pattern_plot_labels_patterns_and_requested_gene(plot_data):
    from bento._constants import PATTERN_NAMES

    points = bt.ut.get_points(plot_data)
    labels = (
        points[["cell_boundaries", "feature_name"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    for i, pattern in enumerate(PATTERN_NAMES):
        labels[pattern] = (np.arange(len(labels)) % len(PATTERN_NAMES) == i).astype(int)
    probabilities = labels.copy()
    probabilities[PATTERN_NAMES] = labels[PATTERN_NAMES] * 0.8 + 0.1
    plot_data.tables["table"].uns["lp"] = labels
    plot_data.tables["table"].uns["lpp"] = probabilities
    gene = labels["feature_name"].iloc[0]
    _, ax = plt.subplots()
    bt.pl.lp_genes(plot_data, annotate=[gene], ax=ax)
    text = {t.get_text() for t in ax.texts}
    assert set(PATTERN_NAMES) <= text
    assert gene in text
