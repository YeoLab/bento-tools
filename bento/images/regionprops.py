import numpy as np
from typing import Union, List
from spatialdata import SpatialData

from ._measure import measure


def _regionprops(image: np.ndarray, label: np.ndarray) -> dict:
    from skimage.measure import regionprops_table

    props = regionprops_table(
        label_image=label.astype(np.uint8),
        intensity_image=image,
        properties=["moments_weighted_hu"],
    )
    props = {k: v[0] for k, v in props.items()}
    return props


def regionprops(
    sdata: SpatialData,
    image_key: str,
    label_key: str,
    img_channels: Union[List[str], str, None] = None,
    num_workers: int = 1,
):
    return measure(sdata, func=_regionprops, image_key=image_key, label_key=label_key, img_channels=img_channels, num_workers=num_workers, name="rprops")


