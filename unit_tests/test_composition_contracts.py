"""Composition distances agree for identical groups."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import spatialdata as sd
import bento as bt


@pytest.fixture
def paired_groups():
    data = sd.read_zarr(Path(bt.__file__).parent / "datasets" / "six_cell_data.zarr")
    bt.io.prep(
        data,
        points_key="transcripts",
        feature_key="feature_name",
        instance_key="cell_boundaries",
        shape_keys=["cell_boundaries", "nucleus_boundaries"],
    )
    points = bt.ut.get_points(data)
    # Duplicate every molecule into a second batch, so both distributions agree.
    import pandas as pd
    from spatialdata.models import PointsModel

    paired = pd.concat(
        [points.assign(batch="control"), points.assign(batch="treated")],
        ignore_index=True,
    )
    attrs = deepcopy(data.points["transcripts"].attrs)
    data.points["transcripts"] = PointsModel.parse(paired)
    data.points["transcripts"].attrs = attrs
    return data


@pytest.mark.xfail(
    strict=True,
    raises=KeyError,
    reason="https://github.com/YeoLab/bento-tools/issues/197",
)
def test_composition_distance_vanishes_for_identical_batches(paired_groups):
    bt.tl.comp_diff(
        paired_groups,
        "transcripts",
        ["cell_boundaries", "nucleus_boundaries"],
        "batch",
        "control",
    )
    stats = paired_groups.tables["table"].uns["batch_comp_stats"]
    assert set(stats) == {"control", "treated"}
    np.testing.assert_allclose(stats["treated"]["treated_diff"], 0, atol=1e-12)
    np.testing.assert_allclose(
        stats["control"][["cell", "nucleus"]], stats["treated"][["cell", "nucleus"]]
    )


def test_composition_distance_with_custom_shape_names(paired_groups):
    from spatialdata.models import PointsModel

    points = bt.ut.get_points(paired_groups)
    points["cell"] = points["cell_boundaries"]
    points["nucleus"] = points["nucleus_boundaries"]
    attrs = deepcopy(paired_groups.points["transcripts"].attrs)
    paired_groups.points["transcripts"] = PointsModel.parse(points)
    paired_groups.points["transcripts"].attrs = attrs
    for alias, original in [
        ("cell", "cell_boundaries"),
        ("nucleus", "nucleus_boundaries"),
    ]:
        paired_groups.shapes[alias] = paired_groups.shapes[original].copy()
    bt.tl.comp_diff(
        paired_groups, "transcripts", ["cell", "nucleus"], "batch", "control"
    )
    result = paired_groups.tables["table"].uns["batch_comp_stats"]
    np.testing.assert_allclose(result["treated"]["treated_diff"], 0, atol=1e-12)
