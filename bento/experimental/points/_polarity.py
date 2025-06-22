from typing import Union
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from scipy.spatial import distance


def _polarity(
    points: pd.DataFrame,
    shape: Union[Polygon, MultiPolygon],
    radius: float,
    x: float,
    y: float,
) -> float:
    """Calculate relative displacement of points within shape using center of mass.

    Parameters
    ----------
    points : pd.DataFrame
        Points with x,y coordinates
    shape : Union[Polygon, MultiPolygon]
        Shape to calculate polarity within
    radius : float
        Radius of shape
    x : float
        shape centroid x
    y : float
        shape centroid y

    Returns
    -------
    dict
        dist: Distance from points to shape center of mass normalized by the shape radius
    """
    if not shape or not radius or not x or not y:
        return {"polarity": np.nan}

    points = points[["x", "y"]].values
    points_com = np.mean(points, axis=0)
    polarity = np.linalg.norm(points_com - (x, y)) / radius
    return {"polarity": polarity}
