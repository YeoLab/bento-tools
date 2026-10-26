"""Synthetic spatial data shared by numerical unit tests."""

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box
from spatialdata import SpatialData
from spatialdata.models import PointsModel, ShapesModel, TableModel


@pytest.fixture
def rectangles():
    shapes = ShapesModel.parse(
        gpd.GeoDataFrame(
            geometry=[box(0, 0, 4, 3), box(10, 10, 12, 12)],
            index=pd.Index(["wide", "square"]),
        )
    )
    points = PointsModel.parse(
        pd.DataFrame(
            {
                "x": [1.0, 2.0, 11.0],
                "y": [1.0, 1.0, 11.0],
                "cells": ["wide", "wide", "square"],
                "gene": pd.Categorical(["a", "a", "a"]),
            }
        )
    )
    points.attrs["spatialdata_attrs"] = {"instance_key": "cells", "feature_key": "gene"}
    table = TableModel.parse(
        ad.AnnData(
            X=np.array([[2.0], [1.0]]),
            obs=pd.DataFrame(
                {
                    "cell_id": ["wide", "square"],
                    "region": pd.Categorical(["cells", "cells"]),
                },
                index=["wide", "square"],
            ),
            var=pd.DataFrame(index=["a"]),
        ),
        region="cells",
        region_key="region",
        instance_key="cell_id",
    )
    return SpatialData(
        shapes={"cells": shapes},
        points={"transcripts": points},
        tables={"table": table},
    )
