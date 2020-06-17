from typing import Dict, Any
import os

import numpy as np

from allennlp.training import EpochCallback, GradientDescentTrainer
import torch

class DatasetCacheCallback(EpochCallback):
    def __init__(self, path):
        self.path = path
        self.loaded = False

    def __call__(self, trainer: GradientDescentTrainer, metrics: Dict[str, Any], epoch: int, **kwargs):
        if self.loaded:
            return
        if os.path.exists(self.path + ".npz"):
            files = np.load(self.path + ".npz")
            ids = files["ids"]
            vectors = torch.Tensor(files["vectors"])
            trainer.model.cache = dict(zip(ids, vectors))
            self.loaded = True
        elif epoch >= 0:
            cache = trainer.model.cache
            ids, vectors = zip(*cache.items())
            np.savez_compressed(self.path, ids=ids, vectors=[v.detach().cpu().numpy() for v in vectors])
