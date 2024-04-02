"""Utility functions for logging media to TensorBoard or WandB."""

from typing import Any, List

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


def log_images(logger: pl.loggers.logger.Logger, key: str, images: List[Any], **kwags: Any) -> None:
    """Log images to TensorBoard or WandB.
    For TensorBoard:
        images (torch.Tensor, numpy.ndarray, or string/blobname)
        optional kwargs are global_step (int), walltime (float), dataformats (str)

    For WandB:
        images (tensors, numpy arrays, PIL images, or filepaths)
        optional kwargs are lists passed to each image (ex: captions, masks, boxes).
    """
    if not isinstance(images, list):
        raise TypeError(f'Expected a list as "images", found {type(images)}')

    if isinstance(logger, TensorBoardLogger):
        for idx, image in enumerate(images):
            logger.add_image(key + f"/{idx}", image, **kwags)

    elif isinstance(logger, WandbLogger):
        logger.log_image(key, images=images, **kwags)

    else:
        raise TypeError(f"Expected a TensorBoardLogger or WandbLogger, found {type(logger)}")
