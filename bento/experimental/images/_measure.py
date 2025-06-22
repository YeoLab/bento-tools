from typing import Callable, List, Optional, Tuple, Union


import spatialdata as sd
import anndata as ad
import dask
import dask.array as da
import dask.bag as db
import numpy as np
import pandas as pd
import xarray as xr
from anndata import AnnData
from scipy import ndimage
from scipy.sparse import csr_matrix
from skimage.filters import gaussian
from spatialdata import SpatialData
from tqdm.dask import TqdmCallback
from tqdm.auto import tqdm
from ..._logging import logger


def get_image_crops(
    image: da.Array, labels: np.ndarray
) -> Tuple[List[da.Array], List[int]]:
    """Lazily get image crops for each unique label in the labels array.

    Parameters
    ----------
    image : np.ndarray
        (c, y, x) image to crop
    labels : np.ndarray
        Labels to crop

    Returns
    -------
    Tuple[List[da.Array], List[int]]
        A tuple containing:
        - List of cropped images Dask arrays
        - List of cropped labels
        - List of unique label indices (excluding 0)
    """
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image array (c, y, x), got shape {image.shape}")
    if labels.ndim != 2:
        raise ValueError(f"Expected 2D labels array (y, x), got shape {labels.shape}")

    # Mask image by label, broadcast over channels
    # image_masked = image.copy()
    # image_masked[:, labels == 0] = 0

    # Get bounding boxes for each label
    bboxes = ndimage.find_objects(labels)
    bboxes = np.array([bbox for bbox in bboxes if bbox is not None])

    unique_labels = np.unique(labels)
    unique_labels = [label for label in unique_labels if label != 0]

    # Create a list to store image crops
    image_crops = []
    labels_crops = []
    # Iterate through bounding boxes and extract crops
    for bbox, i in zip(bboxes, unique_labels):
        # Extract the crop using the bounding box
        crop = (
            image[:, bbox[0], bbox[1]]
            * (labels[bbox[0], bbox[1]] > 0).astype(np.uint32)
        ).to_numpy()

        # Apply gaussian smoothing
        crop = gaussian(crop, sigma=2, preserve_range=True)
        image_crops.append(crop)
        labels_crops.append(labels[bbox[0], bbox[1]] == i)
    return image_crops, labels_crops, unique_labels


def measure(
    sdata: SpatialData,
    func: Callable,
    image_key: str,
    label_key: str,
    name: str,
    img_channels: Optional[Union[str, List[str]]] = None,
    num_workers: int = 1,
) -> pd.DataFrame:
    """Process image crops masked by labels.

    This function applies a function (or multiple) to image crops that are masked by each label in the SpatialData object.

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
    name : str
        Used as suffix of table name to store the result in
    img_channels : str or list of str, optional
        Channels to use for the function. If None, infer channels from image_key
    num_workers : int, optional
        Number of workers to use for parallel processing

    Modifies
    -------
    sdata : SpatialData object with results added to:
        - `tables[name]` <n_labels, n_channels> with each metric stored as a layer

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

    if isinstance(sdata[image_key], xr.DataTree):
        image = sd.get_pyramid_levels(sdata[image_key], n=0).sel(c=img_channels)
    else:
        image = sdata[image_key].sel(c=img_channels)
    labels = sdata[label_key].to_numpy()

    # Get <c, y, x> dask arrays for each label
    logger.info("Loading images")
    img_crops, labels_crops, unique_labels = get_image_crops(image, labels)

    results = []

    # Apply function to each channel
    def func_wrapper(inner_func, img, label):
        return [inner_func(img[i], label) for i in range(img.shape[0])]

    # Parallelize across crops
    logger.info("Bagging")
    bags = (
        db.from_sequence([(img, label) for img, label in zip(img_crops, labels_crops)])
        .map(lambda x: func_wrapper(func, *x))
        .repartition(npartitions=min(len(unique_labels), 100))
    )

    logger.info("Computing")
    dask.config.set(num_workers=num_workers, threads_per_worker=1)
    with TqdmCallback():
        results = bags.compute()

    logger.info("Reshaping")
    for i, r in enumerate(results):
        channel_result = pd.DataFrame(r)
        channel_result["label"] = unique_labels[i]
        channel_result["channel"] = img_channels
        results[i] = channel_result
    results = pd.concat(results)

    # Reshape results, one df per metric with labels as index and channels as columns
    result_layers = {}
    metrics = results.columns.drop(["label", "channel"])
    for metric in metrics:
        result_layers[metric] = results.pivot(
            index="label", columns="channel", values=metric
        )

    table_key = f"{label_key}.{image_key}.{name}"

    logger.info(f"Saving to: sdata['{table_key}']")
    if table_key in sdata:
        sdata[table_key].layers = result_layers
    else:
        empty_x = csr_matrix(np.zeros_like(result_layers[metrics[0]]))
        # Set row and column names to match the dataframe
        table = AnnData(X=empty_x, layers=result_layers)
        table.obs_names = unique_labels
        table.var_names = img_channels
        table.obs["region"] = label_key
        table.obs["label_index"] = table.obs.index
        sdata[table_key] = sd.models.TableModel.parse(table)
        sdata.set_table_annotates_spatialelement(
            table_key, label_key, region_key="region", instance_key="label_index"
        )

    logger.info(f"""{sdata[table_key]}""")


# ================================ IMAGE FUNCTIONS ================================


def _total_intensity(image: np.ndarray, label: np.ndarray) -> float:
    """Calculate the total intensity of an image.

    Parameters
    ----------
    image : np.ndarray
        Image to calculate the total intensity of

    Returns
    -------
    dict
        "total_intensity": total intensity of the image
    """
    return {"total_intensity": image.sum(where=label > 0)}


def _mean_intensity(image: np.ndarray, label: np.ndarray) -> float:
    """Calculate the mean intensity of an image.

    Parameters
    ----------
    image : np.ndarray
        Image to calculate the mean intensity of

    Returns
    -------
    dict
        "mean_intensity": mean intensity of the image
    """
    return {"mean_intensity": image.mean(where=label > 0)}


def _regionprops(image: np.ndarray, label: np.ndarray) -> float:
    """Calculate the regionprops of an image.

    Parameters
    ----------
    image : np.ndarray
        Image to calculate the regionprops of
    """
    from skimage.measure import regionprops_table

    props = regionprops_table(
        label_image=label.astype(np.uint8),
        intensity_image=image,
        properties=[
            # "area",
            # "area_convex",
            # "axis_major_length",
            # "axis_minor_length",
            # "eccentricity",
            # "equivalent_diameter_area",
            # "euler_number",
            # "extent",
            # "feret_diameter_max",
            # "inertia_tensor",
            # "inertia_tensor_eigvals",
            # "intensity_max",
            # "intensity_mean",
            # "intensity_min",
            # "intensity_std",
            # "moments",
            # "moments_central",
            # "moments_hu",
            # "moments_normalized",
            # "moments_weighted",
            # "moments_weighted_central",
            "moments_weighted_hu",
            # "moments_weighted_normalized",
            # "num_pixels",
            # "orientation",
            # "perimeter",
            # "perimeter_crofton",
            # "solidity",
        ],
    )
    props = {k: v[0] for k, v in props.items()}  # Unpack the values
    return props


# ================================ PUBLIC API WRAPPERS ================================


def total_intensity(
    sdata: SpatialData,
    image_key: str,
    label_key: str,
    img_channels: Union[str, List[str]] = None,
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
        num_workers=num_workers,
        name="total",
    )


def mean_intensity(
    sdata: SpatialData,
    image_key: str,
    label_key: str,
    img_channels: Union[str, List[str]] = None,
    num_workers: int = 1,
) -> pd.DataFrame:
    """Calculate the mean intensity of each label in the SpatialData object.

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
    num_workers : int, optional
        Number of workers to use for parallel processing

    Returns
    -------
    pd.DataFrame
        Mean intensity of each label per channel
    """
    return measure(
        sdata,
        func=_mean_intensity,
        image_key=image_key,
        label_key=label_key,
        img_channels=img_channels,
        num_workers=num_workers,
        name="mean",
    )


def regionprops(
    sdata: SpatialData,
    image_key: str,
    label_key: str,
    img_channels: Union[str, List[str]] = None,
    num_workers: int = 1,
) -> pd.DataFrame:
    """Calculate the regionprops of each label in the SpatialData object.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object
    """

    return measure(
        sdata,
        func=_regionprops,
        image_key=image_key,
        label_key=label_key,
        img_channels=img_channels,
        name="rprops",
        num_workers=num_workers,
    )
