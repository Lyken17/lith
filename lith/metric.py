import math

import collections
from collections import OrderedDict

import numpy as np
import torch
from torch.autograd import Variable

class Metric(object):
    def __init__(self):
        pass

    @property
    def name(self):
        return self.__class__.__name__

    def torchlize(func):
        """
        preprocess the args from numpy into torch tensor
        """
        def convert(*args, **kwargs):
            new_args = []
            for idx, each in enumerate(args):
                if isinstance(each, np.ndarray):
                    each = torch.from_numpy(each)
                if isinstance(each, Variable):
                    each = each.data
                new_args.append(each)
            args = tuple(new_args)

            for key, value in kwargs:
                if isinstance(value, np.ndarray):
                    kwargs[key] = torch.from_numpy(value)
                if isinstance(value, Variable):
                    kwargs[key] = value.data
            return func(*args, **kwargs)

        return convert


class Compose(Metric):
    def __init__(self, *args):
        super(Compose, self).__init__()
        self.metrics = []
        if len(args) == 1 and isinstance(args[0], collections.Iterable):
            self.metrics = args[0]
        else:
            for idx, module in enumerate(args):
                self.metrics.append(module)

    def __call__(self, output, target):
        result = OrderedDict()
        for m in self.metrics:
            key = m.name
            value = m(output, target)
            result[key] = value
        return result


class TopKAccuracy(Metric):
    def __init__(self, k):
        super(TopKAccuracy, self).__init__()
        self.k = k

    @property
    def name(self):
        return "Top-%d-Accuracy" % self.k

    @Metric.torchlize
    def __call__(self, output, target):
        """
        :param output[torch.FloatTensor]: *xC
        :param target[torch.LongTensor]: *x1 or *
        :return:
        """
        if output.dim() - 1 == target.dim():
            target = target.unsqueeze(-1)
        assert output.dim() == target.dim(), \
            '''To compute TopK, `output` must have same shape as `target` 
            Current, output.size(): {} target.size(): {}'''.format(output.size(), target.size())
        sorted, indices = torch.topk(output, self.k, -1)
        comp = indices - target
        correct = comp.eq(0).cpu().sum()
        # print(indices.size(), target.size(), correct, output.size(0))
        # print(indices.size(), target.size(), comp.size(), correct, output.size(0))
        return correct / output.size(0)


class Accuracy(TopKAccuracy):
    def __init__(self):
        super(Accuracy, self).__init__(k=1)

    @property
    def name(self):
        return self.__class__.__name__


class Error(Accuracy):
    def __call__(self, output, target):
        acc = super(Error, self).__call__(output, target)
        return 1 - acc


class MeanSquareError(Metric):
    @Metric.torchlize
    def __call__(self, output, target):
        assert output.size() == target.size(), \
            "To compute MSE, output must have same shape as target"
        return torch.mean((output - target) ** 2)


class IoU(Metric):
    @Metric.torchlize
    def __call__(self, output, target):
        """
        :param output: NxCxHxW
        :param target: NxHxW
        :return:
        """
        assert output.dim() - 1 == target.dim() or output.dim() == target.dim(), \
            '''To compute IoU, output must have one more or same dimension as target's,
            e.g. output: NxCxHxW, target: NxHxW or output: NxCxHxW, target: Nx1xHxW'''
        pred = output.max(1)[1]
        channel = output.size(1)
        IoU_list = []
        for i in range(channel):
            inter = ((pred == i) & (target == i)).cpu().sum()
            union = ((pred == i) | (target == i)).cpu().sum()
            IoU = inter / union
            IoU_list.append(IoU)
        return torch.Tensor(IoU_list)


class MeanIoU(IoU):
    def __init__(self, weight=None):
        super(MeanIoU, self).__init__()
        # assert
        self.weight = weight

    @Metric.torchlize
    def __call__(self, output, target):
        IoU_list = super(MeanIoU, self).__call__(output, target)
        if self.weight is not None:
            return torch.mean(IoU_list)
        else:
            return torch.mean(IoU_list * self.weight)


class ConfusionMatrix(Metric):
    @Metric.torchlize
    def __call__(self, output, target):
        """
        :param output: NxC
        :param target: Nx1
        :return:
        """
        channels = output.size(-1)
        fusionMatrix = [_ for _ in range(channels)]
        pred = output.max(-1)[1]
        for i in range(channels):
            tp = ((pred == i) & (target == i)).cpu().sum()
            fp = ((pred == i) & (target != i)).cpu().sum()
            fn = ((pred != i) & (target == i)).cpu().sum()
            tn = ((pred != i) & (target != i)).cpu().sum()
            fusionMatrix[i] = torch.Tensor(
                [[tp, fp],
                 [fn, tn]]
            )
        return fusionMatrix

