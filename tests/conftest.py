import pytest
import spatialdata as sd
import numpy as np
import dask.array as da
import geopandas as gpd
import pandas as pd
import dask.dataframe as dd
from spatialdata.models import Image2DModel, Labels2DModel, ShapesModel, PointsModel
from shapely.geometry import Polygon

import bento as bt
import logging

logger = logging.getLogger(__name__)

TEST_ZARR = "small_data.zarr"
CELL_TO_NUCLEUS_MAP = {
    "c0": "",
    "c1": "n4",
    "c2": "n6",
    "c3": "",
    "c4": "n0",
    "c5": "",
}
NUCLEUS_TO_CELL_MAP = {"n0": "c4", "n4": "c1", "n6": "c2"}

FLUX_RES = 0.5
FLUX_RADIUS = 20
FLUXMAP_MIN_COUNT = 1
FLUXMAP_TRAIN_SIZE = 1
FLUXMAP_N_CLUSTERS = 3
FAZAL2019_FEATURES = [
    "Cytosol",
    "ER Lumen",
    "ERM",
    "Lamina",
    "Nuclear Pore",
    "Nucleolus",
    "Nucleus",
    "OMM",
]
XIA2019_FEATURES = ["ER", "Nucleus"]

# Constants for synthetic data
N_SHAPES_SMALL: int = 100
N_SHAPES_MEDIUM: int = 1000
N_GENES: int = 20
N_POINTS_PER_GENE: int = 10


@pytest.fixture(scope="session")
def small_data():
    data = sd.read_zarr(bt.__file__.rsplit("/", 1)[0] + "/datasets/" + TEST_ZARR)
    data = bt.io.prep(
        data,
        points_key="transcripts",
        feature_key="feature_name",
        instance_key="cell_boundaries",
        shape_keys=["cell_boundaries", "nucleus_boundaries"],
    )
    logger.info("Small data loaded:")
    logger.info(data)
    return data


def _create_unified_synthetic_dataset(n_shapes, n_genes, points_per_gene, include_images=True):
    """Create a unified synthetic dataset with shapes, points, and optionally images/labels.
    
    Parameters
    ----------
    n_shapes : int
        Number of shapes to create
    n_genes : int
        Number of genes/features for points
    points_per_gene : int
        Number of points per gene per shape
    include_images : bool, default True
        Whether to include image and label data
    
    Returns
    -------
    SpatialData
        Unified SpatialData object with shapes, points, and optionally images/labels
    """
    np.random.seed(42)

    # Create shapes and points in a grid layout
    shapes_data = {}
    points_data = {"x": [], "y": [], "feature_name": []}

    # Calculate grid layout for shapes to avoid overlap
    grid_size = int(np.ceil(np.sqrt(n_shapes)))
    cell_size = 100.0 / grid_size  # 100x100 area

    for i in range(n_shapes):
        # Calculate grid position
        row = i // grid_size
        col = i % grid_size

        # Create a square shape with some padding
        center_x = col * cell_size + cell_size / 2
        center_y = row * cell_size + cell_size / 2
        size = cell_size * 0.8  # 80% of cell size to avoid overlap

        # Create square polygon
        half_size = size / 2
        coords = [
            (center_x - half_size, center_y - half_size),
            (center_x + half_size, center_y - half_size),
            (center_x + half_size, center_y + half_size),
            (center_x - half_size, center_y + half_size),
            (center_x - half_size, center_y - half_size),  # Close the polygon
        ]

        shapes_data[f"{i}"] = Polygon(coords)

        # Generate points for each gene within this shape
        for gene_idx in range(n_genes):
            # Generate points within the square bounds
            x_coords = np.random.uniform(center_x - half_size, center_x + half_size, points_per_gene)
            y_coords = np.random.uniform(center_y - half_size, center_y + half_size, points_per_gene)

            points_data["x"].extend(x_coords)
            points_data["y"].extend(y_coords)
            points_data["feature_name"].extend([f"gene_{gene_idx}"] * points_per_gene)

    # Create GeoDataFrame for shapes
    shapes_gdf = gpd.GeoDataFrame({"geometry": list(shapes_data.values())}, index=list(shapes_data.keys()))
    shapes = ShapesModel.parse(shapes_gdf)

    # Create points DataFrame
    points_df = pd.DataFrame(points_data)
    points_ddf = dd.from_pandas(points_df, npartitions=1)
    points = PointsModel.parse(points_ddf)

    # Create SpatialData with shapes and points
    data = sd.SpatialData(
        points={"transcripts": points},
        shapes={"cell_boundaries": shapes}
    )

    # Prepare the data (creates table)
    data = bt.io.prep(
        data,
        points_key="transcripts",
        feature_key="feature_name",
        instance_key="cell_boundaries",
        shape_keys=["cell_boundaries"],
    )

    # Add images and labels if requested
    if include_images:
        # Create a 2D label array that matches the spatial extent
        # Use a finer resolution than shapes for better label representation
        height, width = 200, 200
        labels = np.zeros((height, width), dtype=np.uint32)

        # Map each shape to a label region
        # Scale coordinates from [0, 100] to [0, 200]
        scale_factor = height / 100.0
        
        for i, (shape_idx, shape) in enumerate(shapes_gdf.iterrows()):
            bounds = shape.geometry.bounds
            # Convert shape bounds to pixel coordinates
            xmin_px = int(bounds[0] * scale_factor)
            ymin_px = int(bounds[1] * scale_factor)
            xmax_px = int(bounds[2] * scale_factor)
            ymax_px = int(bounds[3] * scale_factor)
            
            # Ensure within bounds
            xmin_px = max(0, min(xmin_px, width - 1))
            ymin_px = max(0, min(ymin_px, height - 1))
            xmax_px = max(xmin_px + 1, min(xmax_px, width))
            ymax_px = max(ymin_px + 1, min(ymax_px, height))
            
            # Fill label region (using shape index + 1, since labels start from 1)
            labels[ymin_px:ymax_px, xmin_px:xmax_px] = i + 1

        # Create a 3D image array with 3 channels (c, y, x)
        n_channels = 3
        image = np.zeros((n_channels, height, width), dtype=np.float32)

        # Generate different intensity patterns for each channel
        for channel_idx in range(n_channels):
            for region_id in range(1, min(n_shapes + 1, 255)):  # Limit to avoid overflow
                mask = labels == region_id
                if np.any(mask):
                    # Each channel has different intensity per region
                    intensity = np.random.uniform(100, 255) * (channel_idx + 1)
                    image[channel_idx, mask] = intensity
                    # Add some noise
                    noise = np.random.normal(0, 10, size=mask.sum())
                    image[channel_idx, mask] += noise

        # Clip to valid range
        image = np.clip(image, 0, 255)

        # Convert to dask arrays
        image_da = da.from_array(image, chunks=(n_channels, height, width))
        labels_da = da.from_array(labels, chunks=(height, width))

        # Create SpatialData models
        image_model = Image2DModel.parse(image_da, dims=["c", "y", "x"])
        labels_model = Labels2DModel.parse(labels_da, dims=["y", "x"])

        # Set channel coordinate names as strings (c0, c1, c2)
        image_model = image_model.assign_coords(c=[f"c{i}" for i in range(n_channels)])

        # Add images and labels to existing SpatialData
        data.images["test_image"] = image_model
        data.labels["test_labels"] = labels_model

    return data


@pytest.fixture(scope="session")
def synthetic_data():
    """Unified fixture providing synthetic data with shapes, points, and images/labels."""
    return _create_unified_synthetic_dataset(
        n_shapes=N_SHAPES_SMALL,
        n_genes=N_GENES,
        points_per_gene=N_POINTS_PER_GENE,
        include_images=True
    )
