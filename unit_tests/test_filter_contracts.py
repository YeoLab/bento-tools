"""Gene filtering keeps count tables, points and categories synchronized."""

from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from spatialdata import SpatialData
from spatialdata.models import PointsModel

from bento import ut


@pytest.mark.parametrize(
    "points_key,feature_key", [("transcripts", "feature_name"), ("molecules", "gene")]
)
@pytest.mark.parametrize(
    "threshold,genes",
    [(1, ["high", "low"]), (2, ["high", "low"]), (3, ["high"]), (4, [])],
)
def test_filter_by_gene_threshold_and_custom_keys(
    points_key, feature_key, threshold, genes
):
    counts = ad.AnnData(
        X=np.array([[3.0, 1.0, 0.0], [0.0, 2.0, 0.0]]),
        obs=pd.DataFrame(index=["a", "b"]),
        var=pd.DataFrame(index=["high", "low", "absent"]),
    )
    points = PointsModel.parse(
        pd.DataFrame(
            {
                "x": [0.0, 1.0, 2.0],
                "y": [0.0, 1.0, 2.0],
                feature_key: pd.Categorical(["high", "low", "absent"]),
            }
        )
    )
    original_attrs = points.attrs.copy()
    sdata = SpatialData(points={points_key: points}, tables={"table": counts})
    assert (
        ut.filter_by_gene(
            sdata, min_count=threshold, points_key=points_key, feature_key=feature_key
        )
        is sdata
    )
    assert list(sdata.tables["table"].var_names) == genes
    result = sdata.points[points_key].compute()
    assert list(result[feature_key]) == genes
    assert set(result[feature_key].cat.categories) == set(genes)
    assert sdata.points[points_key].attrs == original_attrs
    assert sdata.tables["table"].shape == (2, len(genes))


@pytest.mark.parametrize("astype", ["array", "numpy", "invalid"])
def test_get_points_rejects_unknown_return_type(astype):
    sdata = SpatialData(
        points={
            "transcripts": PointsModel.parse(pd.DataFrame({"x": [0.0], "y": [0.0]}))
        }
    )
    with pytest.raises(ValueError, match="astype must be one of"):
        ut.get_points(sdata, astype=astype, sync=False)


def test_get_points_rejects_missing_points_key():
    with pytest.raises(ValueError, match="not found"):
        ut.get_points(SpatialData(), points_key="missing", sync=False)


@pytest.mark.parametrize(
    "operation",
    [
        lambda sdata, key: ut.get_shape(sdata, key, sync=False),
        lambda sdata, key: ut.set_shape_metadata(sdata, key, [1], "value"),
    ],
)
def test_shape_access_rejects_missing_shape(operation):
    with pytest.raises(ValueError, match="not found"):
        operation(
            SimpleNamespace(
                shapes={},
                tables={
                    "table": SimpleNamespace(
                        uns={"spatialdata_attrs": {"instance_key": "cells"}}
                    )
                },
            ),
            "missing",
        )
