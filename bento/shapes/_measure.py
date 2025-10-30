from typing import Callable, List, Union

import dask.bag as db
import dask.config
import emoji
import numpy as np
import pandas as pd
from spatialdata import SpatialData
from tqdm.dask import TqdmCallback
from tqdm.auto import tqdm

from bento._utils import get_shape
from bento._logging import logger


def measure(
    sdata: SpatialData,
    func: Callable,
    shape_key: str,
    result_keys: Union[str, List[str]],
    n_jobs: int = 1,
    recompute: bool = True,
    progress: bool = True,
    leave: bool = True,
) -> None:
    if not recompute and all(k in sdata.shapes[shape_key].columns for k in (result_keys if isinstance(result_keys, list) else [result_keys])):
        logger.info(f"Skipping, recompute is False. {result_keys} exists in sdata.shapes[{shape_key}]")
        return

    shapes = get_shape(sdata, shape_key, sync=False).geometry
    shape_names = shapes.index.tolist()

    if n_jobs == 1:
        if progress:
            desc = emoji.emojize(":hourglass_not_done:")
            result = [func(shape) for shape in tqdm(shapes, desc=desc, leave=leave, mininterval=0.5)]
        else:
            result = [func(shape) for shape in shapes]
    else:
        shape_coords = np.array(shapes.apply(lambda x: np.array(x.exterior.xy)))
        bags = db.from_sequence(shape_coords).map(func)
        dask.config.set(num_workers=n_jobs)
        if progress:
            desc = emoji.emojize(":hourglass_not_done:")
            with TqdmCallback(desc=desc, leave=leave):
                result = bags.compute()
        else:
            result = bags.compute()

    result = pd.DataFrame(result, index=shape_names)
    sdata.shapes[shape_key][result_keys] = result
    logger.info(f"`{result_keys}` saved to: sdata['{shape_key}']")


