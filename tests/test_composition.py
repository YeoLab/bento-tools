import bento as bt
import pandas as pd


def test_comp(small_data):
    # Set up test data
    shape_names = ["cell_boundaries", "nucleus_boundaries"]
    points_key = "transcripts"

    # Call the comp function
    bt.tl.comp(sdata=small_data, points_key=points_key, shape_names=shape_names)

    # Check if comp_stats is updated in sdata.tables["table"].uns
    assert "comp_stats" in small_data.tables["table"].uns

    # Check the type of comp_stats
    assert type(small_data.tables["table"].uns["comp_stats"]) == pd.DataFrame

    # Check if the shape_names are present in comp_stats
    assert all(
        shape_name in small_data.tables["table"].uns["comp_stats"]
        for shape_name in shape_names
    )


def test_comp_plot_annotate():
    import matplotlib

    matplotlib.use("Agg")
    sdata = bt.ds.sample_data()
    bt.geo.overlay(
        sdata,
        s1="cell_boundaries",
        s2="nucleus_boundaries",
        name="cytoplasm",
        how="difference",
    )
    bt.tl.comp(
        sdata=sdata,
        points_key="transcripts",
        shape_names=["cytoplasm", "nucleus_boundaries"],
    )

    # annotate with adjust=True (the default) runs adjustText over the points
    bt.pl.comp(sdata, annotate=10)
