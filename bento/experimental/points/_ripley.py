from typing import Union
import numpy as np
import pandas as pd
from astropy.stats import RipleysKEstimator
from scipy.spatial import distance_matrix
from scipy.stats import spearmanr
from shapely.geometry import MultiPolygon, Polygon

# TODO define set of functions under `autocorr`. see other packages for inspiration

def _morans_i(
    points: pd.DataFrame,
    shape: Union[Polygon, MultiPolygon],
    attribute: str = None,
    k: int = 8,
) -> dict:
    """Calculate Moran's I spatial autocorrelation statistic for point patterns.
    
    Parameters
    ----------
    points : pd.DataFrame
        Points with x,y coordinates and optional attribute values
    shape : Union[Polygon, MultiPolygon]
        Shape containing the points
    attribute : str, optional
        Column name for attribute values. If None, uses point density
    k : int, default 8
        Number of nearest neighbors for spatial weights
        
    Returns
    -------
    dict
        morans_i: Moran's I statistic (-1 to 1)
        morans_p: P-value for significance test
        morans_z: Z-score for significance test
    """
    if len(points) < 3:
        return {
            "morans_i": np.nan,
            "morans_p": np.nan,
            "morans_z": np.nan,
        }
    
    # Get point coordinates
    coords = points[["x", "y"]].values
    n = len(coords)
    
    # Create spatial weights matrix using k-nearest neighbors
    distances = distance_matrix(coords, coords)
    weights = np.zeros((n, n))
    
    for i in range(n):
        # Get k nearest neighbors (excluding self)
        neighbor_indices = np.argsort(distances[i])[1:k+1]
        weights[i, neighbor_indices] = 1
    
    # Make weights symmetric
    weights = (weights + weights.T) > 0
    weights = weights.astype(float)
    
    # Row standardize weights
    row_sums = weights.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    weights = weights / row_sums[:, np.newaxis]
    
    # Get attribute values
    if attribute and attribute in points.columns:
        values = points[attribute].values
    else:
        # Use local point density as attribute
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=min(k, n-1)).fit(coords)
        distances_nn, _ = nbrs.kneighbors(coords)
        # Use inverse of mean distance to k nearest neighbors as density proxy
        values = 1 / (distances_nn.mean(axis=1) + 1e-10)
    
    # Calculate Moran's I
    mean_val = np.mean(values)
    deviations = values - mean_val
    
    # Numerator: sum of weighted cross-products
    numerator = 0
    for i in range(n):
        for j in range(n):
            numerator += weights[i, j] * deviations[i] * deviations[j]
    
    # Denominator: sum of squared deviations
    denominator = np.sum(deviations ** 2)
    
    # Sum of weights
    w_sum = np.sum(weights)
    
    if denominator == 0 or w_sum == 0:
        return {
            "morans_i": np.nan,
            "morans_p": np.nan,
            "morans_z": np.nan,
        }
    
    # Moran's I statistic
    morans_i = (n / w_sum) * (numerator / denominator)
    
    # Calculate expected value and variance for significance testing
    expected_i = -1 / (n - 1)
    
    # Simplified variance calculation (assumes normality)
    s2 = denominator / n
    b2 = (np.sum(deviations ** 4) / n) / (s2 ** 2)
    
    # Sum of squared weights
    w2_sum = np.sum(weights ** 2)
    
    # Variance of Moran's I under normality assumption
    var_i = ((n * ((n ** 2 - 3 * n + 3) * w2_sum - n * w_sum + 3 * (w_sum ** 2)) - 
              b2 * ((n ** 2 - n) * w2_sum - 2 * n * w_sum + 6 * (w_sum ** 2))) / 
             ((n - 1) * (n - 2) * (n - 3) * (w_sum ** 2))) - expected_i ** 2
    
    if var_i <= 0:
        z_score = np.nan
        p_value = np.nan
    else:
        # Z-score
        z_score = (morans_i - expected_i) / np.sqrt(var_i)
        
        # P-value (two-tailed test)
        from scipy.stats import norm
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
    
    return {
        "morans_i": morans_i,
        "morans_p": p_value,
        "morans_z": z_score,
    }



def _ripley(
    points: pd.DataFrame,
    shape: Union[Polygon, MultiPolygon],
) -> dict:
    """Calculate Ripley's L-function statistics for point patterns."""
    if not shape or len(points) < 2:
        return {
            "l_max": np.nan,
            "l_max_gradient": np.nan,
            "l_min_gradient": np.nan,
            "l_monotony": np.nan,
            "l_half_radius": np.nan,
        }

    # Get cell properties
    shape_coo = np.array(shape.exterior.coords.xy).T
    cell_span = distance_matrix(shape_coo, shape_coo).max()
    cell_minx, cell_miny, cell_maxx, cell_maxy = shape.bounds
    cell_area = shape.area

    estimator = RipleysKEstimator(
        area=cell_area,
        x_min=cell_minx,
        y_min=cell_miny,
        x_max=cell_maxx,
        y_max=cell_maxy,
    )

    quarter_span = cell_span / 4
    radii = np.linspace(1, quarter_span * 2, num=int(quarter_span * 2))

    # Get points coordinates
    points_geo = points[["x", "y"]].values

    # Compute ripley function stats
    stats = estimator.Hfunction(data=points_geo, radii=radii, mode="none")

    # Max value of the L-function
    l_max = max(stats)

    # Max and min value of the gradient of L
    ripley_smooth = pd.Series(stats).rolling(5).mean()
    ripley_smooth.dropna(inplace=True)

    # Can't take gradient of single number
    if len(ripley_smooth) < 2:
        ripley_smooth = np.array([0, 0])

    ripley_gradient = np.gradient(ripley_smooth)
    l_max_gradient = ripley_gradient.max()
    l_min_gradient = ripley_gradient.min()

    # Monotony of L-function in the interval
    l_monotony = spearmanr(radii, stats)[0]

    # L-function at L/4 where length of the cell L is max dist between 2 points on polygon defining cell border
    l_half_radius = estimator.Hfunction(
        data=points_geo, radii=[quarter_span], mode="none"
    )[0]

    return {
        "l_max": l_max,
        "l_max_gradient": l_max_gradient,
        "l_min_gradient": l_min_gradient,
        "l_monotony": l_monotony,
        "l_half_radius": l_half_radius,
    }


