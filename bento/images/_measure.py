from typing import Callable, List, Optional, Tuple, Union

import spatialdata as sd
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

from bento._logging import logger


def get_image_crops(image: da.Array, labels: np.ndarray) -> Tuple[List[da.Array], List[int]]:
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image array (c, y, x), got shape {image.shape}")
    if labels.ndim != 2:
        raise ValueError(f"Expected 2D labels array (y, x), got shape {labels.shape}")

    bboxes = ndimage.find_objects(labels)
    bboxes = np.array([bbox for bbox in bboxes if bbox is not None])

    unique_labels = np.unique(labels)
    unique_labels = [label for label in unique_labels if label != 0]

    image_crops = []
    labels_crops = []
    for bbox, i in zip(bboxes, unique_labels):
        crop = (image[:, bbox[0], bbox[1]] * (labels[bbox[0], bbox[1]] > 0).astype(np.uint32)).to_numpy()
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
    if not img_channels:
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

    logger.info("Loading images")
    img_crops, labels_crops, unique_labels = get_image_crops(image, labels)

    results = []

    def func_wrapper(inner_func, img, label):
        return [inner_func(img[i], label) for i in range(img.shape[0])]

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

    result_layers = {}
    metrics = results.columns.drop(["label", "channel"])
    for metric in metrics:
        result_layers[metric] = results.pivot(index="label", columns="channel", values=metric)

    table_key = f"{label_key}.{image_key}.{name}"

    logger.info(f"Saving to: sdata['{table_key}']")
    if table_key in sdata:
        sdata[table_key].layers = result_layers
    else:
        empty_x = csr_matrix(np.zeros_like(result_layers[metrics[0]]))
        table = AnnData(X=empty_x, layers=result_layers)
        table.obs_names = [int(x) for x in results["label"].unique().tolist()]
        table.var_names = img_channels
        table.obs["region"] = label_key
        table.obs["label_index"] = table.obs.index
        sdata[table_key] = sd.models.TableModel.parse(table)
        sdata.set_table_annotates_spatialelement(table_key, label_key, region_key="region", instance_key="label_index")

    logger.info(f"{sdata[table_key]}")


