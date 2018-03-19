import numpy as np
import torch


class TensorToNumpy(object):
    def __call__(self, tensor):
        return tensor.numpy()


class NumpyToTensor(object):
    def __call__(self, tensor):
        return tensor.from_numpy(tensor)