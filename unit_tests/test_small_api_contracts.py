"""Small public API contracts independent of bundled analysis datasets."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bento import tl, ut
from bento.tools import _point_features, _shape_features
from bento.tools._flux import vec2color


@pytest.mark.parametrize(
    "key,getter",
    [("instance_key", ut.get_instance_key), ("feature_key", ut.get_feature_key)],
)
def test_metadata_key_lookup(key, getter):
    sdata = SimpleNamespace(
        points={
            "transcripts": SimpleNamespace(
                attrs={"spatialdata_attrs": {key: "custom_name"}}
            )
        }
    )
    assert getter(sdata) == "custom_name"


@pytest.mark.parametrize("getter", [ut.get_instance_key, ut.get_feature_key])
@pytest.mark.parametrize("level", ["points", "attributes", "key"])
def test_metadata_key_missing_raises_keyerror(getter, level):
    points = (
        {}
        if level == "points"
        else {
            "transcripts": SimpleNamespace(
                attrs={} if level == "attributes" else {"spatialdata_attrs": {}}
            )
        }
    )
    with pytest.raises(KeyError, match="key attribute not found"):
        getter(SimpleNamespace(points=points))


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
def test_vector_colors_scale_channels_and_pad_missing(channels):
    vectors = np.tile(np.arange(3.0)[:, None], (1, channels))
    original = vectors.copy()
    colors = vec2color(vectors, fmt="rgb")
    expected = np.zeros((3, 3))
    expected[:, : min(channels, 3)] = np.array([0.0, 0.5, 1.0])[:, None]
    np.testing.assert_allclose(colors, expected)
    np.testing.assert_array_equal(vectors, original)


def test_vector_colors_respect_scaling_and_normalize_alpha():
    colors = vec2color(
        np.array([[1.0], [2.0], [3.0]]),
        alpha_vec=np.array([0.0, 2.0, 4.0]),
        fmt="rgb",
        vmin=0.2,
        vmax=0.8,
    )
    np.testing.assert_allclose(
        colors, [[0.2, 0, 0, 0], [0.5, 0, 0, 0.5], [0.8, 0, 0, 1]]
    )


def test_vector_colors_default_to_hex():
    colors = vec2color(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    assert list(colors) == ["#000000ff", "#ffffffff"]


@pytest.mark.parametrize("name", ["fazal2019", "xia2019"])
def test_packaged_gene_sets_have_nonempty_network(name):
    result = tl.load_gene_sets(name)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert {"source", "target", "weight"} <= set(result.columns)
    assert result[["source", "target", "weight"]].notna().all().all()
    assert pd.api.types.is_numeric_dtype(result["weight"])
    result.iloc[0, 0] = "changed_in_memory"
    assert tl.load_gene_sets(name).iloc[0, 0] != "changed_in_memory"


def test_unknown_gene_set_raises_keyerror():
    with pytest.raises(KeyError):
        tl.load_gene_sets("unknown_gene_set")


def test_register_shape_feature_is_discoverable_and_callable(monkeypatch):
    monkeypatch.setattr(
        _shape_features, "shape_features", _shape_features.shape_features.copy()
    )

    def marker(sdata, shape_key):
        """Record the selected shape."""
        sdata.append(shape_key)

    tl.register_shape_feature("unit_marker", marker)
    assert "unit_marker" in tl.list_shape_features()
    result = []
    _shape_features.shape_features["unit_marker"](result, "cells")
    assert result == ["cells"]


def test_register_point_feature_is_discoverable(monkeypatch):
    monkeypatch.setattr(
        _point_features, "point_features", _point_features.point_features.copy()
    )

    class Marker(tl.PointFeature):
        """A point count feature."""

        def extract(self, df):
            return {"count": len(df)}

    tl.register_point_feature("unit_marker", Marker)
    assert "unit_marker" in tl.list_point_features()
    assert _point_features.point_features["unit_marker"]("cells").extract(
        pd.DataFrame(index=range(3))
    ) == {"count": 3}


@pytest.mark.xfail(
    strict=True, reason="https://github.com/YeoLab/bento-tools/issues/195"
)
def test_rmse_uses_root_of_mean_squared_error():
    from bento.tools._decomposition import rmse

    assert rmse(np.array([1.0, 3.0]), np.zeros(2)) == pytest.approx(np.sqrt(5.0))
