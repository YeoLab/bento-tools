from typing import Union
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon

def _density(points: pd.DataFrame, shape: Union[Polygon, MultiPolygon]) -> dict:
    """Calculate density of points within shape.

    Parameters
    ----------
    points : pd.DataFrame
        Points with x,y coordinates
    shape : Union[Polygon, MultiPolygon]
        Shape to calculate density within

    Returns
    -------
    dict
        density: Number of points divided by shape area
    """
    return {"density": len(points) / shape.area} 