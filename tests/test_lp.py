import pytest


import bento as bt

from tests import conftest


@pytest.fixture(scope="module")
def lp_data():
    data = bt.ds.sample_data()
    data = bt.io.prep(
        data,
        points_key="transcripts",
        feature_key="feature_name",
        instance_key="cell_boundaries",
        shape_keys=["cell_boundaries", "nucleus_boundaries"],
    )
    bt.tl.lp(
        sdata=data,
        instance_key="cell_boundaries",
        nucleus_key="nucleus_boundaries",
        groupby="feature_name",
    )
    bt.tl.lp_stats(sdata=data)

    return data


@pytest.fixture(scope="module")
def lp_small_data(small_data):
    bt.tl.lp(
        sdata=small_data,
        instance_key="cell_boundaries",
        nucleus_key="nucleus_boundaries",
        groupby="feature_name",
    )
    bt.tl.lp_stats(sdata=small_data)

    return small_data


@pytest.fixture()
def lp_diff_discrete_data(lp_data):
    # Assign random category to each cell
    category = ["A", "B"]
    phenotype = [category[i % 2] for i in range(lp_data["cell_boundaries"].shape[0])]
    lp_data.shapes["cell_boundaries"]["cell_stage"] = phenotype

    # Assign random category to each cell
    error_category = [0, 1]
    error_phenotype = [
        error_category[i % 2] for i in range(lp_data["cell_boundaries"].shape[0])
    ]
    lp_data.shapes["cell_boundaries"]["error_test"] = error_phenotype

    # Calculate lp_diff_discrete
    bt.tl.lp_diff_discrete(
        sdata=lp_data, instance_key="cell_boundaries", phenotype="cell_stage"
    )

    return lp_data


@pytest.fixture()
def lp_diff_discrete_small_data(lp_small_data):
    # Assign random category to each cell
    n_cells = lp_small_data["cell_boundaries"].shape[0]
    n_a = n_cells // 2
    n_b = n_cells - n_a
    phenotype = ["A"] * n_a + ["B"] * n_b
    lp_small_data.shapes["cell_boundaries"]["cell_stage"] = phenotype

    return lp_small_data


@pytest.fixture()
def lp_diff_continuous_data(lp_data):
    bt.tl.lp_diff_continuous(lp_data, phenotype="cell_boundaries_area")

    return lp_data


def test_lp(lp_data):
    # Check lp and lpp dataframes in sdata.tables["table"].uns
    assert all(
        column in lp_data.tables["table"].uns["lp"].columns
        for column in conftest.LP_COLUMNS
    )
    assert all(
        column in lp_data.tables["table"].uns["lpp"].columns
        for column in conftest.LP_COLUMNS
    )


def test_lp_stats(lp_data):
    # Check lp_stats index in sdata.tables["table"].uns
    assert lp_data.tables["table"].uns["lp_stats"].index.name == "feature_name"

    # Check lp_stats dataframe in sdata.tables["table"].uns
    for column in conftest.LP_STATS_COLUMNS:
        assert column in lp_data.tables["table"].uns["lp_stats"].columns


def test_lp_diff_discrete(lp_diff_discrete_data):
    # Check lp_diff_discrete dataframe in sdata.tables["table"].uns
    assert all(
        column in lp_diff_discrete_data.tables["table"].uns["diff_cell_stage"].columns
        for column in conftest.LP_DIFF_DISCRETE_COLUMNS
    )


def test_lp_diff_discrete_warn_continuous(lp_diff_discrete_data):
    # Check that UserWarning is raised when phenotype is numeric
    with pytest.warns(UserWarning):
        bt.tl.lp_diff_discrete(
            sdata=lp_diff_discrete_data,
            instance_key="cell_boundaries",
            phenotype="error_test",
        )


def test_lp_diff_discrete_small_data(lp_diff_discrete_small_data):
    # Check that UserWarning is raised when no significant patterns are found when lp is run
    with pytest.warns(UserWarning):
        bt.tl.lp_diff_discrete(
            sdata=lp_diff_discrete_small_data,
            instance_key="cell_boundaries",
            phenotype="cell_stage",
        )


def test_lp_diff_continuous(lp_diff_continuous_data):
    # Check lp_diff_continuous dataframe in sdata.tables["table"].uns
    assert all(
        column
        in lp_diff_continuous_data.tables["table"]
        .uns["diff_cell_boundaries_area"]
        .columns
        for column in conftest.LP_DIFF_CONTINUOUS_COLUMNS
    )

    # def test_lp_dist_plot(self):
    #     plt.figure()
    #     bt.pl.lp_dist(small_data, fname=f"{self.imgdir}/lp_dist.png")

    # def test_lp_genes_plot(self):
    #     plt.figure()
    #     bt.pl.lp_genes(
    #         small_data,
    #         groupby="feature_name",
    #         points_key="transcripts",
    #         instance_key="cell_boundaries",
    #         fname=f"{self.imgdir}/lp_genes.png",
    #     )

    # def test_lp_diff_discrete_plot(self):
    #     area_binary = []
    #     median = small_data.shapes["cell_boundaries"]["cell_boundaries_area"].median()
    #     for i in range(len(small_data.shapes["cell_boundaries"])):
    #         cell_boundaries_area = small_data.shapes["cell_boundaries"][
    #             "cell_boundaries_area"
    #         ][i]
    #         if cell_boundaries_area > median:
    #             area_binary.append("above")
    #         else:
    #             area_binary.append("below")
    #     small_data.shapes["cell_boundaries"]["area_binary"] = area_binary

    #     bt.tl.lp_diff_discrete(small_data, phenotype="area_binary")

    #     plt.figure()
    #     bt.pl.lp_diff_discrete(
    #         small_data,
    #         phenotype="area_binary",
    #         fname=f"{self.imgdir}/lp_diff_discrete.png",
    #     )
