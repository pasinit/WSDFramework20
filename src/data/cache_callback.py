from typing import Dict, Any
import os

import numpy as np

from allennlp.training import EpochCallback, GradientDescentTrainer
import torch

from src.misc.wsdlogging import get_info_logger


class DatasetCacheCallback(EpochCallback):
    def __init__(self, path):
        self.path = path
        self.loaded = False

    def __call__(self, trainer: GradientDescentTrainer, metrics: Dict[str, Any], epoch: int, **kwargs):
        if self.loaded:
            return
        if os.path.exists(self.path + ".npz"):
            get_info_logger(__name__).info("found cache file, loading...")
            files = np.load(self.path + ".npz")
            ids = files["ids"]
            vectors = torch.Tensor(files["vectors"])
            trainer.model.cache = dict(zip(ids, vectors))
            self.loaded = True
        elif epoch >= 0:
            get_info_logger(__name__).info("dumping cache to {}...".format(self.path))

            cache = trainer.model.cache
            ids, vectors = zip(*cache.items())
            np.savez_compressed(self.path, ids=ids, vectors=[v.detach().cpu().numpy() for v in vectors])
        else:
            get_info_logger(__name__).info("cache nof found")

