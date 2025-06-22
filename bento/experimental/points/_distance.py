from typing import Union
import polars as pl
import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from numba import njit, float64, int64, guvectorize

# @guvectorize(
#     [(float64[:], float64[:], float64[:], float64[:], float64, float64[:])],
#     "(n),(n),(n),(n),()->(n)",
#     # nopython=True,
# )
def _calculate_distances(x_points, y_points, x_shape, y_shape, radius, out):
    """Calculate minimum distances from points to shape segments using numba.

    Parameters
    ----------
    x_points : float64[:]
        x coordinates of points
    y_points : float64[:]
        y coordinates of points
    x_shape : float64[:]
        x coordinates of shape vertices
    y_shape : float64[:]
        y coordinates of shape vertices
    radius : float64
        Radius for normalization
    out : float64[:]
        Output array for mean and variance [mean, var]
    """
    n_points = len(x_points)
    n_shape = len(x_shape)
    min_distances = np.full(n_points, np.inf)

    # For each point
    for i in range(n_points):
        px, py = x_points[i], y_points[i]

        # For each shape segment
        for j in range(n_shape - 1):
            # Get segment endpoints
            x1, y1 = x_shape[j], y_shape[j]
            x2, y2 = x_shape[j + 1], y_shape[j + 1]

            # Vector from point to segment start
            A = px - x1
            B = py - y1
            C = x2 - x1
            D = y2 - y1

            # Calculate dot product and length squared
            dot = A * C + B * D
            len_sq = C * C + D * D

            if len_sq == 0:
                # Degenerate line segment
                dist = np.sqrt(A * A + B * B)
            else:
                param = dot / len_sq
                if param < 0:
                    # Closest point is first vertex
                    dist = np.sqrt(A * A + B * B)
                elif param > 1:
                    # Closest point is second vertex
                    xx = px - x2
                    yy = py - y2
                    dist = np.sqrt(xx * xx + yy * yy)
                else:
                    # Closest point is on the line segment
                    xx = px - (x1 + param * C)
                    yy = py - (y1 + param * D)
                    dist = np.sqrt(xx * xx + yy * yy)

            # Update minimum distance for this point
            if dist < min_distances[i]:
                min_distances[i] = dist

    # Normalize distances by radius
    min_distances = min_distances / radius

    # Calculate mean and variance
    out[0] = np.mean(min_distances)
    out[1] = np.var(min_distances)


def _distances(
    points: pl.DataFrame,
    groupby_col: str,
    radius: float,
    shape_x_col: str,
    shape_y_col: str,
) -> pl.DataFrame:
    """Calculate distance stats from points to shape using Polars expressions.

    Parameters
    ----------
    points : pl.DataFrame
        Points DataFrame with x,y coordinates, a groupby column, and shape coordinates
    groupby_col : str
        Column name to group by
    radius : float
        Radius of shape
    shape_x_col : str
        Column name containing x coordinates of the shape
    shape_y_col : str
        Column name containing y coordinates of the shape

    Returns
    -------
    pl.DataFrame
        DataFrame with distance statistics per group:
        - dist_mean: Mean distance from points to shape
        - dist_var: Variance of distance from points to shape
    """
    if not (radius > 0):
        return pl.DataFrame(
            {
                groupby_col: points[groupby_col].unique(),
                "dist_mean": [np.nan],
                "dist_var": [np.nan],
            }
        )

    # Group by and apply the distance calculation
    result = points.group_by(groupby_col).agg(
        pl.struct(["x", "y", shape_x_col, shape_y_col])
        .map_batches(
            lambda x: _calculate_distances(
                x.struct.field("x"),
                x.struct.field("y"),
                x.struct.field(shape_x_col),
                x.struct.field(shape_y_col),
                radius,
            )
        )
        .alias("stats")
    )

    # Split the stats column into mean and variance
    result = result.with_columns(
        [
            pl.col("stats").list.get(0).alias("dist_mean"),
            pl.col("stats").list.get(1).alias("dist_var"),
        ]
    ).drop("stats")

    return result
