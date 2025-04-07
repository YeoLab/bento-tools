from functools import lru_cache
from typing import Callable, List, Union, Tuple, Dict, Any, Optional

import spatialdata as sd
import dask
import dask.bag as db
import numpy as np
import pandas as pd
import xarray as xr
from anndata import AnnData
from scipy import ndimage
from spatialdata import SpatialData
from tqdm.dask import TqdmCallback
from tqdm.auto import tqdm
import tempfile
import os

from ..._logging import logger


def get_image_crops(
    image: np.ndarray,
    labels: np.ndarray
) -> Tuple[List[np.ndarray], List[int]]:
    """Get image crops from a numpy array.

    Parameters
    ----------
    image : np.ndarray
        (c, y, x) image to crop
    labels : np.ndarray
        Labels to crop

    Returns
    -------
    Tuple[List[np.ndarray], List[int]]
        A tuple containing:
        - List of cropped images
        - List of unique label indices (excluding 0)
    """
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image array (c, y, x), got shape {image.shape}")
    if labels.ndim != 2:
        raise ValueError(f"Expected 2D labels array (y, x), got shape {labels.shape}")

    # Mask image by label, broadcast over channels
    image_masked = image.copy()
    image_masked[:, labels == 0] = 0

    # Get bounding boxes for each label
    bboxes = ndimage.find_objects(labels)
    bboxes = np.array([bbox for bbox in bboxes if bbox is not None])

    unique_labels = np.unique(labels)
    unique_labels = [label for label in unique_labels if label != 0]

    # Create a list to store image crops
    image_crops = []

    # Iterate through bounding boxes and extract crops
    for bbox in bboxes:
        # Extract the crop using the bounding box
        crop = image_masked[:, bbox[0], bbox[1]]
        image_crops.append(crop)

    return image_crops, unique_labels


def _apply_func(
    sdata: SpatialData,
    func: Callable,
    image_key: str,
    label_key: str,
    img_channels: List[str],
    num_workers: int = 1,
) -> pd.DataFrame:
    """Internal function to process image crops with parallel processing.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object containing images and labels
    func : Callable
        Function to apply to each image crop
    image_key : str
        Key for the image in sdata.images
    label_key : str
        Key for the labels in sdata.images
    img_channels : List[str]
        Channels to use for the function
    num_workers : int
        Number of workers for parallel processing

    Returns
    -------
    pd.DataFrame
        Results of applying func to each image crop

    Raises
    ------
    KeyError
        If image_key or label_key not found in sdata
    """
    if image_key not in sdata.images:
        raise KeyError(f"Image key '{image_key}' not found in sdata")
    if label_key not in sdata.images:
        raise KeyError(f"Label key '{label_key}' not found in sdata")

    image = sd.get_pyramid_levels(sdata[image_key], n=0).sel(c=img_channels).to_numpy()
    labels = sdata[label_key].to_numpy()

    img_crops, unique_labels = get_image_crops(image, labels)

    bags = db.from_sequence(img_crops).map(func)

    with TqdmCallback(desc=f"Using {num_workers} cores"):
        dask.config.set(num_workers=num_workers, threads_per_worker=1)
        result = bags.compute()

    result = pd.DataFrame(result, index=unique_labels)

    return result


def measure(
    sdata: SpatialData,
    func: Callable,
    image_key: str,
    label_key: str,
    result_suffix: str,
    img_channels: Optional[Union[str, List[str]]] = None,
    num_workers: int = 1,
) -> pd.DataFrame:
    """Process image crops masked by labels.

    This function applies a function to image crops that are masked by each label in the SpatialData object.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object
    func : Callable
        Function to apply to each image crop
    image_key : str
        Key for the image in sdata
    label_key : str
        Key for the labels in sdata
    result_suffix : str
        Suffix of keys to store the result in the label annotation table
    img_channels : str or list of str, optional
        Channels to use for the function. If None, infer channels from image_key
    num_workers : int, optional
        Number of workers to use for parallel processing

    Returns
    -------
    pd.DataFrame
        Results of applying func to each image crop

    Raises
    ------
    ValueError
        If img_channels is empty or invalid
    """
    if not img_channels:  # Infer channels from image
        img_channels = sd.models.get_channel_names(sdata[image_key])
        if not img_channels:
            raise ValueError(f"No channels found in image '{image_key}'")

    if isinstance(img_channels, str):
        img_channels = [img_channels]

    # Process calculation
    result = _apply_func(
        sdata=sdata,
        func=func,
        image_key=image_key,
        label_key=label_key,
        img_channels=img_channels,
        num_workers=num_workers,
    )

    table_key = f"{label_key}_bt"
    col_names = [f"{c}_{result_suffix}" for c in img_channels]
    result.columns = col_names

    logger.debug(f"Saving to: sdata['{table_key}']")
    if table_key in sdata.tables.keys():
        sdata[table_key].obs[col_names] = result
    else:
        table = AnnData(obs=result)
        table.obs.index = table.obs.index.astype(str)
        table.obs["region"] = label_key
        table.obs["label_index"] = table.obs.index
        logger.debug(table.obs.columns)
        sdata[table_key] = sd.models.TableModel.parse(table)
        sdata.set_table_annotates_spatialelement(
            table_key, label_key, region_key="region", instance_key="label_index"
        )

    print(f"""Saved to: sdata['{table_key}']
            columns: {col_names}""")

    return result


# ================================ IMAGE FUNCTIONS ================================


def _total_intensity(image: np.ndarray) -> float:
    """Calculate the total intensity of an image.

    Parameters
    ----------
    image : np.ndarray
        Image to calculate the total intensity of

    Returns
    -------
    np.ndarray
        Total intensity of the image per channel
    """
    return image.sum(axis=(1, 2))


# ================================ PUBLIC API WRAPPERS ================================


def total_intensity(
    sdata: SpatialData,
    image_key: str,
    label_key: str,
    img_channels: Union[str, List[str]],
    result_suffix: str = "total",
    num_workers: int = 1,
) -> pd.DataFrame:
    """Calculate the total intensity of each label in the SpatialData object.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object
    image_key : str
        Key for the image in sdata
    label_key : str
        Key for the labels in sdata
    img_channels : str or list of str
        Channels to use for the function
    result_suffix : str, optional
        Suffix of key to store the result in the label annotation table
    num_workers : int, optional
        Number of workers to use for parallel processing

    Returns
    -------
    pd.DataFrame
        Total intensity of each label per channel
    """
    return measure(
        sdata,
        func=_total_intensity,
        image_key=image_key,
        label_key=label_key,
        img_channels=img_channels,
        result_suffix=result_suffix,
        num_workers=num_workers,
    )
