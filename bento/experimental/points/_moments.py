from typing import Union
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon


def _central_moment(points: np.ndarray, com: np.ndarray, p: int, q: int) -> float:
    """Calculate central moment of points relative to shape center of mass."""
    return np.mean((points[:, 0] - com[0]) ** p * (points[:, 1] - com[1]) ** q)


def _norm_central_moment(points: np.ndarray, com: np.ndarray, p: int, q: int) -> float:
    """Calculate normalized central moment of points relative to shape center of mass."""
    return _central_moment(points, com, p, q) / _central_moment(points, com, 0, 0) ** (
        (p + q) / 2 + 1
    )


def _hu_moment(points: np.ndarray, com: np.ndarray) -> tuple:
    """Calculate Hu moments of points relative to shape center of mass."""
    eta_20 = _norm_central_moment(points, com, 2, 0)
    eta_02 = _norm_central_moment(points, com, 0, 2)
    eta_11 = _norm_central_moment(points, com, 1, 1)
    eta_30 = _norm_central_moment(points, com, 3, 0)
    eta_21 = _norm_central_moment(points, com, 2, 1)
    eta_12 = _norm_central_moment(points, com, 1, 2)
    eta_03 = _norm_central_moment(points, com, 0, 3)

    hu_1 = eta_20 + eta_02
    hu_2 = (eta_20 - eta_02) ** 2 + 4 * eta_11**2
    hu_3 = (eta_30 - 3 * eta_12) ** 2 + (3 * eta_21 - eta_03) ** 2
    hu_4 = (eta_30 + eta_12) ** 2 + (eta_21 + eta_03) ** 2
    hu_5 = (eta_30 - 3 * eta_12) * (eta_30 + eta_12) * (
        (eta_30 + eta_12) ** 2 - 3 * (eta_21 + eta_03) ** 2
    ) + (3 * eta_21 - eta_03) * (eta_21 + eta_03) * (
        3 * (eta_30 + eta_12) ** 2 - (eta_21 + eta_03) ** 2
    )
    hu_6 = (eta_20 - eta_02) * (
        (eta_30 + eta_12) ** 2 - (eta_21 + eta_03) ** 2
    ) + 4 * eta_11 * (eta_30 + eta_12) * (eta_21 + eta_03)
    # hu_7 = (3 * eta_21 - eta_03) * (eta_30 + eta_12) * (
    #     (eta_30 + eta_12) ** 2 - 3 * (eta_21 + eta_03) ** 2
    # ) - (eta_30 - 3 * eta_12) * (eta_21 + eta_03) * (
    #     3 * (eta_30 + eta_12) ** 2 - (eta_21 + eta_03) ** 2
    # )

    # log scale all moments
    hu_1 = -(np.sign(hu_1) * np.log(abs(hu_1)))
    hu_2 = -(np.sign(hu_2) * np.log(abs(hu_2)))
    hu_3 = -(np.sign(hu_3) * np.log(abs(hu_3)))
    hu_4 = -(np.sign(hu_4) * np.log(abs(hu_4)))
    hu_5 = -(np.sign(hu_5) * np.log(abs(hu_5)))
    hu_6 = -(np.sign(hu_6) * np.log(abs(hu_6)))
    # hu_7 = -(np.sign(hu_7) * np.log(abs(hu_7)))
    return hu_1, hu_2, hu_3, hu_4, hu_5, hu_6


def _hu_moments(points: pd.DataFrame, shape: Union[Polygon, MultiPolygon], x: float, y: float) -> dict:
    """Calculate moment-based features for points within shape.

    Parameters
    ----------
    points : pd.DataFrame
        Points with x,y coordinates
    shape : Union[Polygon, MultiPolygon]
        Shape to calculate moment-based features within
    x : float
        shape centroid x
    y : float
        shape centroid y

    Returns
    -------
    dict
        hum_1: First moment of points relative to shape center of mass
        hum_2: Second moment of points relative to shape center of mass
        hum_3: Third moment of points relative to shape center of mass
        hum_4: Fourth moment of points relative to shape center of mass
        hum_5: Fifth moment of points relative to shape center of mass
        hum_6: Sixth moment of points relative to shape center of mass
        # hum_7: Seventh moment of points relative to shape center of mass
    """
    points = points[["x", "y"]].values

    hu_1, hu_2, hu_3, hu_4, hu_5, hu_6 = _hu_moment(points, (x, y))

    return {
        "hum_1": hu_1,
        "hum_2": hu_2,
        "hum_3": hu_3,
        "hum_4": hu_4,
        "hum_5": hu_5,
        "hum_6": hu_6,
        # "hum_7": hu_7,
    }
