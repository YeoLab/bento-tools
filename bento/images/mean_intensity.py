import numpy as np
from typing import Union, List
from spatialdata import SpatialData

from ._measure import measure


def _mean_intensity(image: np.ndarray, label: np.ndarray) -> dict:
    return {"mean_intensity": image.mean(where=label > 0)}


def mean_intensity(
    sdata: SpatialData,
    image_key: str,
    label_key: str,
    img_channels: Union[List[str], str, None] = None,
    num_workers: int = 1,
):
    return measure(sdata, func=_mean_intensity, image_key=image_key, label_key=label_key, img_channels=img_channels, num_workers=num_workers, name="mean")


