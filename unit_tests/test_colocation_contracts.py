"""Check the labelled tensor passed to colocation decomposition."""

from unittest.mock import Mock

import numpy as np
import pandas as pd

import bento as bt
from bento.tools import _colocation


def test_colocation_preserves_axis_labels_and_fills_unobserved_pairs(
    rectangles, monkeypatch
):
    uns = rectangles.tables["table"].uns
    uns["clq"] = {
        "nucleus": pd.DataFrame(
            {
                "cells": ["wide", "square"],
                "gene": ["b", "a"],
                "neighbor": ["a", "b"],
                "log_clq": [2.5, -1.0],
            }
        ),
        "cytoplasm": pd.DataFrame(
            {"cells": ["wide"], "gene": ["a"], "neighbor": ["b"], "log_clq": [4.0]}
        ),
    }
    factors = {2: "factorization"}
    errors = pd.DataFrame({"rank": [2], "rmse": [0.25]})
    decompose = Mock(return_value=(factors, errors))
    monkeypatch.setattr(_colocation, "decompose", decompose)

    bt.tl.colocation(
        rectangles,
        ranks=[2],
        instance_key="cells",
        feature_key="gene",
        iterations=5,
        plot_error=False,
    )

    assert uns["tensor_names"] == ["compartment", "cells", "pair"]
    assert {axis: labels.tolist() for axis, labels in uns["tensor_labels"].items()} == {
        "compartment": ["cytoplasm", "nucleus"],
        "cells": ["square", "wide"],
        "pair": ["a_b", "b_a"],
    }
    expected = np.array([[[0, 0], [4, 0]], [[-1, 0], [0, 2.5]]])
    np.testing.assert_array_equal(uns["tensor"], expected)
    decompose.assert_called_once()
    tensor, ranks = decompose.call_args.args
    np.testing.assert_array_equal(tensor, expected)
    assert ranks == [2]
    assert decompose.call_args.kwargs == {"iterations": 5}
    assert uns["factors"] == factors
    pd.testing.assert_frame_equal(uns["factors_error"], errors)
