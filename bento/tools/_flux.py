from typing import Iterable, Literal, Optional, Union, List, Tuple

import dask.array as da
import emoji
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import shapely
from kneed import KneeLocator
from minisom import MiniSom
from scipy.sparse import csr_matrix, vstack
from shapely import Polygon
from shapely.geometry import Point
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, minmax_scale, quantile_transform
from sklearn.utils import resample
from spatialdata._core.spatialdata import SpatialData
from spatialdata.models import ShapesModel, Image2DModel
from spatialdata.transformations import Identity
from tqdm.auto import tqdm
import zarr

from .._utils import (
    get_points,
    get_shape_metadata,
    set_points_metadata,
)
from ..io._index import _sjoin_points, _sjoin_shapes
from ..tools._neighborhoods import _count_neighbors
from ..shapes import radius as compute_radius


def _create_raster_grid(
    sdata: SpatialData,
    instance_key: str,
    res: float,
) -> Tuple[pd.DataFrame, int, int, Tuple[float, float, float, float]]:
    """
    Create raster grid and assign points to cells.
    
    Parameters
    ----------
    sdata : SpatialData
        SpatialData object.
    instance_key : str
        Key for cell_boundaries instances.
    res : float
        Resolution to use for rendering embedding.
    
    Returns
    -------
    Tuple[pd.DataFrame, int, int, Tuple[float, float, float, float]]
        Tuple containing:
        - raster_points: DataFrame with raster points and cell assignments
        - grid_height: Height of the grid in pixels
        - grid_width: Width of the grid in pixels
        - bounds: Spatial bounds (minx, miny, maxx, maxy)
    """
    # Determine spatial extent from shapes
    shapes = sdata.shapes[instance_key]
    bounds = shapes.total_bounds  # [minx, miny, maxx, maxy]
    
    # Create grid coordinates based on resolution
    step = 1 / res
    x_coords = np.arange(bounds[0], bounds[2], step)
    y_coords = np.arange(bounds[1], bounds[3], step)
    
    # Create grid height and width
    grid_height = len(y_coords)
    grid_width = len(x_coords)
    
    # Create meshgrid for raster points
    xx, yy = np.meshgrid(x_coords, y_coords)
    raster_coords = pd.DataFrame({
        'x': xx.ravel(),
        'y': yy.ravel()
    })
    
    # Assign each raster point to a cell using spatial join
    raster_geom = gpd.GeoDataFrame(
        raster_coords,
        geometry=[Point(x, y) for x, y in zip(raster_coords['x'], raster_coords['y'])],
        crs=shapes.crs
    )
    raster_with_cells = gpd.sjoin(raster_geom, shapes, how='left', predicate='within')
    
    # Filter to only points within cells and add index mapping
    raster_points = raster_with_cells[raster_with_cells['index_right'].notna()].copy()
    raster_points[instance_key] = raster_points['index_right']
    raster_points['grid_x'] = ((raster_points['x'] - bounds[0]) / step).astype(int)
    raster_points['grid_y'] = ((raster_points['y'] - bounds[1]) / step).astype(int)
    
    return raster_points, grid_height, grid_width, bounds


def _compute_cell_flux(
    cpoints: pd.DataFrame,
    rpoints: pd.DataFrame,
    n_genes: int,
    method: Literal["knn", "radius"],
    n_neighbors: Optional[int],
    radius_param: Optional[float],
    cell_composition: np.ndarray,
) -> Tuple[csr_matrix, np.ndarray, List]:
    """
    Compute flux values for a single cell.
    
    Parameters
    ----------
    cpoints : pd.DataFrame
        Points dataframe for the cell.
    rpoints : pd.DataFrame
        Raster points dataframe for the cell.
    n_genes : int
        Number of genes.
    method : Literal["knn", "radius"]
        Method to use for local neighborhood.
    n_neighbors : Optional[int]
        Number of neighbors for knn method.
    radius_param : Optional[float]
        Radius for radius method.
    cell_composition : np.ndarray
        Cell composition vector (normalized).
    
    Returns
    -------
    Tuple[csr_matrix, np.ndarray, List]
        Tuple containing:
        - cflux: Sparse matrix of flux values
        - total_count: Array of total counts per raster point
        - rpoint_index: List of raster point indices
    """
    rpoint_index = rpoints.index.tolist()
    if method == "knn":
        gene_count = _count_neighbors(
            cpoints,
            n_genes,
            rpoints,
            n_neighbors=n_neighbors,
            agg=None,
        )
    elif method == "radius":
        gene_count = _count_neighbors(
            cpoints,
            n_genes,
            rpoints,
            radius=radius_param,
            agg=None,
        )
    
    # Work with sparse matrices for memory efficiency
    # Count points in each neighborhood using sparse operations
    total_count = np.asarray(gene_count.sum(axis=1)).ravel()
    
    # Compute row sums for normalization, avoid division by zero
    row_sums = np.asarray(gene_count.sum(axis=1)).ravel()
    row_sums_safe = row_sums.copy()
    row_sums_safe[row_sums_safe == 0] = 1  # Prevent division by zero
    
    # Compute composition using efficient sparse scaling
    flux_composition = gene_count.multiply(1.0 / row_sums_safe[:, np.newaxis])
    
    # Convert to dense only for the subtraction and scaling operations
    # This is necessary because we subtract a dense vector from each row
    flux_dense = flux_composition.toarray()
    cflux_dense = flux_dense - cell_composition
    
    # Apply StandardScaler with with_mean=False for consistent scaling
    cflux_scaled = StandardScaler(with_mean=False).fit_transform(cflux_dense)
    
    # Convert result back to sparse CSR format for efficient storage
    cflux = csr_matrix(cflux_scaled)

    return cflux, total_count, rpoint_index


def _train_svd_model(
    train_data: csr_matrix,
    n_components: int,
    random_state: int,
) -> Tuple[TruncatedSVD, np.ndarray]:
    """
    Train TruncatedSVD model on training data.
    
    Parameters
    ----------
    train_data : csr_matrix
        Training data (sparse matrix).
    n_components : int
        Number of components for SVD.
    random_state : int
        Random state for reproducibility.
    
    Returns
    -------
    Tuple[TruncatedSVD, np.ndarray]
        Tuple containing:
        - model: Trained TruncatedSVD model
        - variance_ratio: Explained variance ratio for each component
    """
    model = TruncatedSVD(
        n_components=n_components, algorithm="randomized", random_state=random_state
    ).fit(train_data)
    variance_ratio = model.explained_variance_ratio_
    return model, variance_ratio


def _transform_flux_embeddings(
    model: TruncatedSVD,
    flux_data: csr_matrix,
) -> np.ndarray:
    """
    Apply SVD transformation to flux data.
    
    Parameters
    ----------
    model : TruncatedSVD
        Trained SVD model.
    flux_data : csr_matrix
        Flux data to transform.
    
    Returns
    -------
    np.ndarray
        Transformed embeddings.
    """
    return model.transform(flux_data)


def _compute_flux_colors(
    flux_embed: np.ndarray,
    rpoints_counts: np.ndarray,
) -> np.ndarray:
    """
    Convert flux embeddings to RGB color representation.
    
    Parameters
    ----------
    flux_embed : np.ndarray
        Flux embeddings.
    rpoints_counts : np.ndarray
        Counts per raster point.
    
    Returns
    -------
    np.ndarray
        RGB color values (N, 3) in range [0, 1].
    """
    flux_color = vec2color(flux_embed, alpha_vec=rpoints_counts)
    
    # Convert flux color hex strings to numeric representation
    # Extract RGB values from hex colors for image storage
    flux_color_rgb = np.array([
        [int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)] 
        if isinstance(c, str) and c.startswith('#') else [0, 0, 0]
        for c in flux_color
    ], dtype=np.float32) / 255.0
    
    return flux_color_rgb


def _initialize_zarr_image(
    sdata: SpatialData,
    image_key: str,
    n_channels: int,
    grid_height: int,
    grid_width: int,
    channel_names: List[str],
    chunk_size: int,
) -> zarr.Array:
    """
    Initialize zarr array for flux image with proper chunking.
    
    Parameters
    ----------
    sdata : SpatialData
        SpatialData object (must be zarr-backed).
    image_key : str
        Key for the image in sdata.images.
    n_channels : int
        Number of channels.
    grid_height : int
        Height of the grid.
    grid_width : int
        Width of the grid.
    channel_names : List[str]
        Names of channels.
    chunk_size : int
        Chunk size for spatial dimensions.
    
    Returns
    -------
    zarr.Array
        Initialized zarr array.
    
    Raises
    ------
    ValueError
        If sdata is not zarr-backed.
    """
    if not (hasattr(sdata, 'path') and sdata.path is not None):
        raise ValueError(
            "SpatialData object must be zarr-backed for streaming flux computation. "
            "Please save your SpatialData object to a zarr store first using sdata.write(path)."
        )
    
    # Open zarr store
    store = zarr.open(sdata.path, mode='a')
    
    # Create zarr array with appropriate chunking
    # Chunks: (n_channels, min(chunk_size, grid_height), min(chunk_size, grid_width))
    chunks = (
        n_channels,
        min(chunk_size, grid_height),
        min(chunk_size, grid_width)
    )
    
    # Create array path in zarr store
    array_path = f"images/{image_key}/data"
    
    # Remove existing array if it exists
    if array_path in store:
        del store[array_path]
    
    # Create new zarr array
    zarr_array = store.create(
        array_path,
        shape=(n_channels, grid_height, grid_width),
        dtype=np.float32,
        chunks=chunks,
        fill_value=0.0,
    )
    
    return zarr_array


def _write_batch_to_zarr(
    zarr_array: zarr.Array,
    batch_flux: csr_matrix,
    batch_embeddings: np.ndarray,
    batch_colors: np.ndarray,
    batch_counts: np.ndarray,
    batch_indices: np.ndarray,
    raster_points: pd.DataFrame,
    n_genes: int,
    n_components: int,
    grid_height: int,
    grid_width: int,
) -> None:
    """
    Write batch results to zarr array at correct grid coordinates.
    
    Parameters
    ----------
    zarr_array : zarr.Array
        Zarr array to write to.
    batch_flux : csr_matrix
        Flux values for the batch (sparse).
    batch_embeddings : np.ndarray
        Embeddings for the batch.
    batch_colors : np.ndarray
        RGB colors for the batch.
    batch_counts : np.ndarray
        Counts for the batch.
    batch_indices : np.ndarray
        Raster point indices for the batch.
    raster_points : pd.DataFrame
        Raster points dataframe with grid coordinates.
    n_genes : int
        Number of genes.
    n_components : int
        Number of embedding components.
    grid_height : int
        Height of the grid.
    grid_width : int
        Width of the grid.
    """
    # Create mapping from raster point index to grid coordinates
    if 'flux_index' not in raster_points.columns:
        raster_points['flux_index'] = range(len(raster_points))
    
    for i, idx in enumerate(batch_indices):
        if idx >= len(raster_points):
            continue
        
        raster_row = raster_points[raster_points['flux_index'] == idx]
        if len(raster_row) == 0:
            continue
        
        grid_y = raster_row['grid_y'].values[0]
        grid_x = raster_row['grid_x'].values[0]
        
        # Ensure indices are within bounds
        if grid_y >= grid_height or grid_x >= grid_width or grid_y < 0 or grid_x < 0:
            continue
        
        # Fill gene values (sparse, so convert row to dense)
        gene_values = batch_flux[i].toarray().ravel()
        zarr_array[:n_genes, grid_y, grid_x] = gene_values
        
        # Fill embedding values
        zarr_array[n_genes:n_genes+n_components, grid_y, grid_x] = batch_embeddings[i]
        
        # Fill color RGB values
        zarr_array[n_genes+n_components:n_genes+n_components+3, grid_y, grid_x] = batch_colors[i]
        
        # Fill counts
        zarr_array[n_genes+n_components+3, grid_y, grid_x] = batch_counts[i]


def flux(
    sdata: SpatialData,
    points_key: str = "transcripts",
    instance_key: str = "cell_boundaries",
    feature_key: str = "feature_name",
    method: Literal["knn", "radius"] = "radius",
    n_neighbors: Optional[int] = None,
    radius: Optional[float] = None,
    res: Optional[float] = 1,
    train_size: Optional[float] = 0.1,
    random_state: int = 11,
    recompute: bool = False,
    num_workers: int = 1,
    chunk_size: int = 10000,
) -> SpatialData:
    """
    Compute RNAflux embeddings of each pixel as local composition normalized by cell composition.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object.
    points_key : str, default "transcripts"
        Key for points element that holds transcript coordinates.
    instance_key : str, default "cell_boundaries"
        Key for cell_boundaries instances.
    feature_key : str, default "feature_name"
        Key for gene instances.
    method : Literal["knn", "radius"], default "radius"
        Method to use for local neighborhood.
    n_neighbors : Optional[int], default None
        Number of neighbors to use for local neighborhood if method is "knn".
    radius : Optional[float], default None
        Fraction of mean cell radius to use for local neighborhood if method is "radius".
        If None, defaults to 1/3 of average cell radius.
    res : Optional[float], default 1
        Resolution to use for rendering embedding.
    train_size : Optional[float], default 0.1
        Fraction of data to use for training.
    random_state : int, default 11
        Random state for reproducibility.
    recompute : bool, default False
        If True, recompute flux even if it already exists.
    num_workers : int, default 1
        Number of workers to use for parallel processing.
    chunk_size : int, default 10000
        Size of chunks for writing data to ome-zarr. Larger values use more memory
        but may improve write performance. Chunked writing enables efficient storage
        and reduces memory usage for large datasets.

    Returns
    -------
    SpatialData
        Updated SpatialData object with:
        - .images["{instance_key}_flux"]: Image2DModel containing flux values as channels:
            * Gene expression channels (one per gene)
            * Embedding channels (flux_embed_0, flux_embed_1, ...)
            * Color channels (flux_color_r, flux_color_g, flux_color_b)
            * Counts channel (flux_counts)
        - .tables["table"].uns["flux_genes"]: List of genes used for embedding.
        - .tables["table"].uns["flux_variance_ratio"]: Array of explained variance ratio for each component.
        - .tables["table"].uns["flux_image_key"]: Key of the flux image in sdata.images.
        - .tables["table"].uns["flux_channel_names"]: List of channel names in the flux image.

    Notes
    -----
    RNAflux requires a minimum of 4 genes per cell to compute all embeddings properly.
    
    This function uses streaming computation to process large datasets efficiently.
    The SpatialData object must be zarr-backed (saved to disk using sdata.write(path))
    to enable incremental writing and prevent loading entire datasets into memory.
    """

    # Check if flux has already been computed
    flux_image_key = f"{instance_key}_flux"
    if flux_image_key in sdata.images and not recompute:
        return sdata

    # Check if zarr-backed (required for streaming)
    if not (hasattr(sdata, 'path') and sdata.path is not None):
        raise ValueError(
            "SpatialData object must be zarr-backed for streaming flux computation. "
            "Please save your SpatialData object to a zarr store first using sdata.write(path)."
        )

    if method == "radius":
        compute_radius(sdata, shape_key=instance_key, recompute=True, progress=False)
        mean_radius = (
            get_shape_metadata(
                sdata, shape_key=instance_key, metadata_keys="radius"
            )
            .mean()
            .values[0]
        )
        # Default radius = 33% of average cell radius
        if radius is None:
            radius = mean_radius / 3
        # If radius is a fraction, use that fraction of average cell radius
        elif radius <= 1:
            radius = radius * mean_radius
        # If radius is an integer, use that as the radius

    # Grab molecules
    points = get_points(sdata, points_key=points_key, astype="pandas", sync=True)
    points = points[[instance_key, feature_key, "x", "y"]]

    # Create raster grid for flux computation
    pbar = tqdm(total=5)
    pbar.set_description(emoji.emojize("Creating raster grid"))
    raster_points, grid_height, grid_width, bounds = _create_raster_grid(
        sdata, instance_key, res
    )
    pbar.update()

    # Extract gene names and codes
    gene_names = points[feature_key].cat.categories.tolist()
    n_genes = len(gene_names)

    points_grouped = points.groupby(instance_key)
    rpoints_grouped = raster_points.groupby(instance_key)
    cells = list(points_grouped.groups.keys())
    cells.sort()

    # Compute cell composition (small, keep in memory)
    cell_composition = sdata.tables["table"][cells, gene_names].X.toarray()
    cell_composition = cell_composition / (cell_composition.sum(axis=1).reshape(-1, 1))
    cell_composition[np.isnan(cell_composition)] = 0

    # Number of components for SVD
    n_components = min(n_genes, 10)
    
    # Number of channels: genes + embeddings + RGB (3) + counts (1)
    n_channels = n_genes + n_components + 3 + 1
    embed_names = [f"flux_embed_{i}" for i in range(n_components)]
    channel_names = (
        gene_names + 
        embed_names + 
        ['flux_color_r', 'flux_color_g', 'flux_color_b'] +
        ['flux_counts']
    )

    # Initialize zarr array for streaming writes
    pbar.set_description(emoji.emojize("Initializing zarr array"))
    zarr_array = _initialize_zarr_image(
        sdata, flux_image_key, n_channels, grid_height, grid_width, channel_names, chunk_size
    )
    pbar.update()

    # Create mapping from raster point index to grid coordinates
    raster_points['flux_index'] = range(len(raster_points))

    # Phase 1: Accumulate training samples and train SVD model
    pbar.set_description(emoji.emojize("Training SVD model"))
    train_n = max(1, int(train_size * len(cells)))  # Ensure at least 1 sample
    train_samples = []
    processed_cell_indices = set()  # Track which cells we've already processed
    
    # Process cells in batches to accumulate training data
    # Use a reasonable batch size for training
    training_batch_size = max(50, min(200, train_n // 5))
    
    for i in range(0, min(len(cells), train_n * 3), training_batch_size):  # Process up to 3x train_n to ensure we get enough
        if len(train_samples) >= train_n:
            break
            
        batch_cells = cells[i:min(i + training_batch_size, len(cells))]
        
        # Process batch of cells
        for cell_idx, cell in enumerate(batch_cells):
            global_cell_idx = i + cell_idx
            if len(train_samples) >= train_n:
                break
                
            cpoints = points_grouped.get_group(cell)
            rpoints = rpoints_grouped.get_group(cell)
            
            cflux, counts, rpoint_idx = _compute_cell_flux(
                cpoints, rpoints, n_genes, method, n_neighbors, radius, 
                cell_composition[global_cell_idx]
            )
            
            # Accumulate training samples
            train_samples.append(cflux)
            processed_cell_indices.add(global_cell_idx)
    
    # Stack training samples and train model
    if len(train_samples) > 0:
        train_data = vstack(train_samples)
        train_data.data = np.nan_to_num(train_data.data, copy=False)
        model, variance_ratio = _train_svd_model(train_data, n_components, random_state)
        del train_data, train_samples
    else:
        raise ValueError("No training samples collected. Check train_size parameter.")
    
    pbar.update()

    # Phase 2: Process all cells in batches, transform, and write to zarr
    pbar.set_description(emoji.emojize("Processing cells and writing to zarr"))
    
    # Determine batch size for processing (based on chunk_size and memory considerations)
    # Process cells in batches to control memory usage
    processing_batch_size = max(50, min(500, len(cells) // 10))  # Adaptive batch size
    
    for batch_start in range(0, len(cells), processing_batch_size):
        batch_end = min(batch_start + processing_batch_size, len(cells))
        batch_cells = cells[batch_start:batch_end]
        
        # Process batch
        batch_flux_list = []
        batch_counts_list = []
        batch_indices_list = []
        
        for cell_idx, cell in enumerate(batch_cells):
            global_cell_idx = batch_start + cell_idx
            
            # Skip if already processed in training phase (will be handled separately)
            if global_cell_idx in processed_cell_indices:
                # Recompute for cells used in training (need full results for writing)
                cpoints = points_grouped.get_group(cell)
                rpoints = rpoints_grouped.get_group(cell)
                
                cflux, counts, rpoint_idx = _compute_cell_flux(
                    cpoints, rpoints, n_genes, method, n_neighbors, radius, 
                    cell_composition[global_cell_idx]
                )
            else:
                # New cell, compute flux
                cpoints = points_grouped.get_group(cell)
                rpoints = rpoints_grouped.get_group(cell)
                
                cflux, counts, rpoint_idx = _compute_cell_flux(
                    cpoints, rpoints, n_genes, method, n_neighbors, radius, 
                    cell_composition[global_cell_idx]
                )
            
            batch_flux_list.append(cflux)
            batch_counts_list.append(counts)
            batch_indices_list.append(rpoint_idx)
        
        # Stack batch results
        batch_flux = vstack(batch_flux_list)
        batch_flux.data = np.nan_to_num(batch_flux.data, copy=False)
        batch_counts = np.hstack(batch_counts_list)
        batch_indices = np.hstack(batch_indices_list)
        
        # Transform with SVD
        batch_embeddings = _transform_flux_embeddings(model, batch_flux)
        
        # Compute colors
        batch_colors = _compute_flux_colors(batch_embeddings, batch_counts)
        
        # Write batch to zarr
        _write_batch_to_zarr(
            zarr_array, batch_flux, batch_embeddings, batch_colors, batch_counts,
            batch_indices, raster_points, n_genes, n_components, grid_height, grid_width
        )
        
        # Release batch memory
        del batch_flux, batch_embeddings, batch_colors, batch_counts, batch_indices
        del batch_flux_list, batch_counts_list, batch_indices_list
    
    pbar.update()

    # Create Image2DModel from zarr array
    pbar.set_description(emoji.emojize("Creating image model"))
    
    # Load zarr array as dask array
    # Use the zarr array directly to create dask array
    flux_image_da = da.from_array(zarr_array, chunks=zarr_array.chunks)
    
    # Calculate appropriate scale factors based on image size
    scale_factors = None
    min_dim = min(grid_height, grid_width)
    if min_dim >= 64:
        scale_factors = [2, 4]
    elif min_dim >= 32:
        scale_factors = [2]
    
    flux_image_xr = Image2DModel.parse(
        flux_image_da,
        dims=['c', 'y', 'x'],
        c_coords=channel_names,
        transformations={"global": Identity()},
        scale_factors=scale_factors
    )
    
    # Store as image in SpatialData
    sdata.images[flux_image_key] = flux_image_xr
    
    # Store metadata in table
    sdata.tables["table"].uns["flux_variance_ratio"] = variance_ratio
    sdata.tables["table"].uns["flux_genes"] = gene_names
    sdata.tables["table"].uns["flux_image_key"] = flux_image_key
    sdata.tables["table"].uns["flux_channel_names"] = channel_names
    
    # Write table metadata
    try:
        sdata.write_element("table", overwrite=True)
    except Exception as e:
        import warnings
        warnings.warn(f"Could not write table to zarr: {e}")

    pbar.set_description(emoji.emojize("Done. :bento_box:"))
    pbar.update()
    pbar.close()
    
    return sdata


def vec2color(
    vec: np.ndarray,
    alpha_vec: Optional[np.ndarray] = None,
    fmt: Literal[
        "rgb",
        "hex",
    ] = "hex",
    vmin: float = 0,
    vmax: float = 1,
) -> Union[np.ndarray, List[str]]:
    """
    Convert vector to color.

    Parameters
    ----------
    vec : np.ndarray
        Input vector to convert to color.
    alpha_vec : Optional[np.ndarray], default None
        Vector of alpha values.
    fmt : Literal["rgb", "hex"], default "hex"
        Output format for colors.
    vmin : float, default 0
        Minimum value for color scaling.
    vmax : float, default 1
        Maximum value for color scaling.

    Returns
    -------
    Union[np.ndarray, List[str]]
        Array of RGB values or list of hex color codes.
    """

    # Grab the first 3 channels
    color = vec[:, :3]
    color = quantile_transform(color[:, :3])
    color = minmax_scale(color, feature_range=(vmin, vmax))

    # If vec has fewer than 3 channels, fill empty channels with 0
    if color.shape[1] < 3:
        color = np.pad(color, ((0, 0), (0, 3 - color.shape[1])), constant_values=0)

    # Replace NaNs with 0
    color[np.isnan(color)] = 0

    # Add alpha channel
    if alpha_vec is not None:
        alpha = alpha_vec.reshape(-1, 1)
        # alpha = quantile_transform(alpha)
        alpha = alpha / alpha.max()
        alpha[np.isnan(alpha)] = 0
        color = np.c_[color, alpha]

    if fmt == "rgb":
        pass
    elif fmt == "hex":
        color = np.apply_along_axis(mpl.colors.to_hex, 1, color, keep_alpha=True)

    ["#00000000" if c is np.nan else c for c in color]

    return color


def fluxmap(
    sdata: SpatialData,
    points_key: str = "transcripts",
    instance_key: str = "cell_boundaries",
    n_clusters: Union[Iterable[int], int] = range(2, 9),
    num_iterations: int = 1000,
    min_count: int = 50,
    train_size: float = 1,
    res: float = 1,
    random_state: int = 11,
    plot_error: bool = False,
):
    """
    Cluster flux embeddings using self-organizing maps (SOMs) and vectorize clusters as Polygon shapes.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object.
    points_key : str, default "transcripts"
        Key for points element that holds transcript coordinates.
    instance_key : str, default "cell_boundaries"
        Key for cell_boundaries instances.
    n_clusters : Union[Iterable[int], int], default range(2, 9)
        Number of clusters to use. If iterable, will pick best number of clusters
        using the elbow heuristic evaluated on the quantization error.
    num_iterations : int, default 1000
        Number of iterations to use for SOM training.
    min_count : int, default 50
        Minimum count for a point to be included in clustering.
    train_size : float, default 1
        Fraction of cells to use for SOM training.
    res : float, default 1
        Resolution used for rendering embedding.
    random_state : int, default 11
        Random state to use for SOM training.
    plot_error : bool, default False
        Whether to plot quantization error.

    Returns
    -------
    SpatialData
        Updated SpatialData object with:
        - .points[f"{instance_key}_raster"]: Added "fluxmap" column denoting cluster membership.
        - .shapes["fluxmap#"]: Added "fluxmap#" columns for each cluster rendered as (Multi)Polygon shapes.
    """

    raster_points = get_points(
        sdata, points_key=f"{instance_key}_raster", astype="pandas", sync=False
    )
    rpoints_index = raster_points.index

    # Check if flux embedding has been computed
    if "flux_embed_0" not in raster_points.columns:
        raise ValueError(
            "Flux embedding has not been computed. Run `bento.tl.flux()` first."
        )

    flux_embed = raster_points.filter(regex=r"^flux_embed").copy()

    # Keep only points with minimum neighborhood count
    flux_counts = raster_points["flux_counts"]
    valid_points = flux_counts >= min_count
    flux_embed = flux_embed[valid_points]
    embed_index = flux_embed.index

    # Sort columns
    flux_embed = flux_embed[sorted(flux_embed.columns.tolist())].to_numpy()

    if isinstance(n_clusters, int):
        n_clusters = [n_clusters]

    if isinstance(n_clusters, range):
        n_clusters = list(n_clusters)

    # Subsample flux embeddings for faster training
    if train_size > 1:
        raise ValueError("train_size must be equal to or less than 1.")
    if train_size == 1:
        flux_train = flux_embed
    if train_size < 1:
        flux_train = resample(
            flux_embed,
            n_samples=int(train_size * flux_embed.shape[0]),
            random_state=random_state,
        )

    # Perform SOM clustering over n_clusters range and pick best number of clusters using elbow heuristic
    pbar = tqdm(total=3)
    pbar.set_description(emoji.emojize("Optimizing # of clusters"))
    som_models = {}
    quantization_errors = []
    for k in tqdm(n_clusters, leave=False):
        som = MiniSom(1, k, flux_train.shape[1], random_seed=random_state)
        som.random_weights_init(flux_train)
        som.train(flux_train, num_iterations, random_order=False, verbose=False)
        som_models[k] = som
        quantization_errors.append(som.quantization_error(flux_embed))

    # Use kneed to find elbow
    if len(n_clusters) > 1:
        kl = KneeLocator(
            n_clusters, quantization_errors, curve="convex", direction="decreasing"
        )
        best_k = kl.elbow

        if plot_error:
            kl.plot_knee()
            plt.show()

        if best_k is None:
            print("No elbow found. Rerun with a fixed k or a different range.")
            return

    else:
        best_k = n_clusters[0]
    # from anndata import AnnData
    # import scanpy as sc

    # Cluster flux_train using scanpy leiden clustering
    # flux_adata = AnnData(flux_embed)
    # sc.pp.pca(flux_adata, n_comps=min(flux_embed.shape[1] - 1, 50))
    # sc.pp.neighbors(flux_adata)
    # sc.tl.leiden(flux_adata, resolution=0.5)
    # sc.tl.umap(flux_adata)
    # sc.pl.umap(flux_adata, color="leiden")

    # pbar.update()

    # pbar.set_description(f"Assigning to {flux_adata.obs['leiden'].nunique()} clusters")
    # fluxmap_values = pd.Series(
    #     flux_adata.obs["leiden"].astype(int).values, index=embed_index
    # ).reindex(rpoints_index, fill_value=0)
    # raster_points["fluxmap"] = fluxmap_values
    # set_points_metadata(
    #     sdata,
    #     points_key=f"{instance_key}_raster",
    #     metadata=list(fluxmap_values),
    #     columns="fluxmap",
    # )

    # Use best k to assign each sample to a cluster
    pbar.set_description(f"Assigning to {best_k} clusters")
    som = som_models[best_k]
    winner_coordinates = np.array([som.winner(x) for x in flux_embed]).T

    # Indices start at 0, so add 1; we will treat 0 as background
    qnt_index = np.ravel_multi_index(winner_coordinates, (1, best_k)) + 1
    qnt_index = pd.Series(qnt_index, index=embed_index).reindex(
        rpoints_index, fill_value=0
    )
    raster_points["fluxmap"] = qnt_index
    set_points_metadata(
        sdata,
        points_key=f"{instance_key}_raster",
        metadata=list(qnt_index),
        columns="fluxmap",
    )
    pbar.update()

    # Vectorize polygons in each cell
    pbar.set_description(emoji.emojize("Vectorizing domains"))
    cells = raster_points[instance_key].unique().tolist()

    # Cast to int
    raster_points[["x", "y", "fluxmap"]] = raster_points[["x", "y", "fluxmap"]].astype(
        int
    )

    rpoints_grouped = raster_points.groupby(instance_key)
    fluxmap_df = dict()
    for cell in tqdm(cells, leave=False):
        rpoints = rpoints_grouped.get_group(cell)

        # Translate so all points are positive and save offsets
        x_offset = rpoints["x"].min()
        y_offset = rpoints["y"].min()
        rpoints["x"] = rpoints["x"] - x_offset
        rpoints["y"] = rpoints["y"] - y_offset

        # Fill in image at each point xy with fluxmap value by casting to dense matrix
        image = (
            csr_matrix(
                (
                    rpoints["fluxmap"],
                    (
                        (rpoints["y"] * res).astype(int),
                        (rpoints["x"] * res).astype(int),
                    ),
                )
            )
            .todense()
            .astype("int16")
        )

        # Find all the contours
        contours = rasterio.features.shapes(image)
        polygons = np.array([(shapely.geometry.shape(p), v) for p, v in contours])
        shapes = gpd.GeoDataFrame(
            polygons[:, 1],
            geometry=gpd.GeoSeries(polygons[:, 0]).T,
            columns=["fluxmap"],
        )

        # Remove background shape
        shapes["fluxmap"] = shapes["fluxmap"].astype(int)
        shapes = shapes[shapes["fluxmap"] != 0]

        # Group same fields as MultiPolygons
        shapes = shapes.dissolve("fluxmap")["geometry"]

        # Upscale to match original resolution, shift back to original coordinates
        shapes = shapes.scale(xfact=1 / res, yfact=1 / res, origin=(0, 0)).translate(
            x_offset, y_offset
        )

        fluxmap_df[cell] = shapes

    fluxmap_df = pd.DataFrame.from_dict(fluxmap_df).T
    fluxmap_df.columns = "fluxmap" + fluxmap_df.columns.astype(str)

    pbar.update()

    pbar.set_description("Saving")

    old_shapes = [k for k in sdata.shapes.keys() if k.startswith("fluxmap")]
    for key in old_shapes:
        del sdata.shapes[key]

    sd_attrs = sdata.shapes[instance_key].attrs
    fluxmap_df = fluxmap_df.reindex(sdata.tables["table"].obs_names).where(
        fluxmap_df.notna(), other=Polygon()
    )
    fluxmap_names = fluxmap_df.columns.tolist()
    for fluxmap in fluxmap_names:
        sdata.shapes[fluxmap] = ShapesModel.parse(
            gpd.GeoDataFrame(geometry=fluxmap_df[fluxmap])
        )
        sdata.shapes[fluxmap].attrs = sd_attrs

    old_cols = sdata.points[points_key].columns[
        sdata.points[points_key].columns.str.startswith("fluxmap")
    ]
    sdata.points[points_key] = sdata.points[points_key].drop(old_cols, axis=1)

    _sjoin_points(sdata=sdata, shape_keys=fluxmap_names, points_key=points_key)
    _sjoin_shapes(sdata=sdata, instance_key=instance_key, shape_keys=fluxmap_names, instance_map_type="1to1")

    pbar.update()
    pbar.set_description("Done")
    pbar.close()
