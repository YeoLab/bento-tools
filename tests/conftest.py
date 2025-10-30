import pytest
import spatialdata as sd
import numpy as np
import dask.array as da
from spatialdata.models import Image2DModel, Labels2DModel

import bento as bt

TEST_ZARR = "small_data.zarr"
SIX_CELL_ZARR = "six_cell_data.zarr"
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

SHAPE_FEATURES = bt.tl.list_shape_features().keys()
SHAPE_FEATURE_NAMES = [
    "area",
    "aspect_ratio",
    "minx",
    "miny",
    "maxx",
    "maxy",
    "density",
    "open_0.5_shape",
    "perimeter",
    "radius",
    "raster",
    "moment",
    "span",
]
CELL_FEATURES = [f"cell_boundaries_{x}" for x in SHAPE_FEATURE_NAMES]
NUCLEUS_FEATURES = [f"nucleus_boundaries_{x}" for x in SHAPE_FEATURE_NAMES]
OPENING_PARAMS = {"opening": {"proportion": 0.5}}

POINT_FEATURES = bt.tl.list_point_features().keys()
POINT_ONLY_FEATURE_NAMES = [
    "point_dispersion_norm",
    "point_dispersion",
    "l_max",
    "l_max_gradient",
    "l_min_gradient",
    "l_monotony",
    "l_half_radius",
]
POINT_FEATURE_NAMES = [
    "inner_proximity",
    "outer_proximity",
    "inner_asymmetry",
    "outer_asymmetry",
    "dispersion_norm",
    "inner_distance",
    "outer_distance",
    "inner_offset",
    "outer_offset",
    "dispersion",
    "enrichment",
]
POINT_CELL_FEATURE_NAMES = [f"cell_boundaries_{x}" for x in POINT_FEATURE_NAMES]
POINT_NUCLEUS_FEATURE_NAMES = [f"nucleus_boundaries_{x}" for x in POINT_FEATURE_NAMES]

LP_COLUMNS = [
    "cell_boundaries",
    "feature_name",
    "cell_edge",
    "cytoplasmic",
    "none",
    "nuclear",
    "nuclear_edge",
]
LP_STATS_COLUMNS = ["cell_edge", "cytoplasmic", "none", "nuclear", "nuclear_edge"]

LP_DIFF_DISCRETE_COLUMNS = [
    "feature_name",
    "pattern",
    "phenotype",
    "dy/dx",
    "std_err",
    "z",
    "pvalue",
    "ci_low",
    "ci_high",
    "padj",
    "-log10p",
    "-log10padj",
    "log2fc",
]

LP_DIFF_CONTINUOUS_COLUMNS = [
    "feature_name",
    "pattern",
    "pearson_correlation",
]


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
    return data


def _create_synthetic_image_data():
    """Create synthetic image and label data for testing image features."""
    np.random.seed(42)
    
    # Create a 2D label array with distinct regions
    height, width = 200, 200
    labels = np.zeros((height, width), dtype=np.uint32)
    
    # Create 10 distinct circular regions
    n_regions = 10
    region_size = 30
    for i in range(n_regions):
        center_y = np.random.randint(region_size, height - region_size)
        center_x = np.random.randint(region_size, width - region_size)
        y, x = np.ogrid[:height, :width]
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= region_size ** 2
        labels[mask] = i + 1  # Labels start from 1
    
    # Create a 3D image array with 3 channels (c, y, x)
    n_channels = 3
    image = np.zeros((n_channels, height, width), dtype=np.float32)
    
    # Generate different intensity patterns for each channel
    for channel_idx in range(n_channels):
        for region_id in range(1, n_regions + 1):
            mask = labels == region_id
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
    
    # Create SpatialData object
    data = sd.SpatialData(
        images={"test_image": image_model},
        labels={"test_labels": labels_model}
    )
    
    return data


@pytest.fixture(scope="session")
def image_data():
    """Fixture providing synthetic image and label data for testing image features."""
    return _create_synthetic_image_data()
