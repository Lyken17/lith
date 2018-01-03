import torch
import torch.optim
import torch.nn.functional as F
from torch.autograd import Variable

from torch.nn.modules import Module


def _assert_no_grad(variable):
    assert not variable.requires_grad, "nn criterions don't compute the gradient w.r.t. targets - please " \
                                       "mark these variables as volatile or not requiring gradients"


def _assert_equal_shape(var1, var2):
    assert var1.size() == var2.size(), "Two variable must have equal shape"


class _Loss(Module):
    def __init__(self, size_average=True, reduce=True):
        super(_Loss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce


class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=True, reduce=True):
        super(_WeightedLoss, self).__init__(size_average, reduce)
        self.register_buffer('weight', weight)


class MaxmiumMeanDiscrepancy(_Loss):
    def __init__(self, size_average=True, reduce=True):
        super(MaxmiumMeanDiscrepancy, self).__init__(size_average, reduce)

    def forward(self, output, target):
        _assert_no_grad(target)
        _assert_equal_shape(output, target)
        mean1 = torch.mean(output, 0)
        mean2 = torch.mean(target, 1)
        res = (mean1 - mean2) ** 2
        if self.reduce:
            return res
        if self.size_average:
            n = res.size(0)
            return torch.sum(res) / n
        return torch.sum(res)


class KLDivergence(_Loss):
    def __init__(self, size_average=True, reduce=True):
        super(KLDivergence, self).__init__(size_average, reduce)

    def forward(self, output, target):
        _assert_no_grad(target)
        _assert_equal_shape(output, target)

        res = target * (torch.log(target) - output)
        res = torch.sum(res, 0)

        if self.reduce:
            return res
        if self.size_average:
            n = res.size(0)
            return torch.sum(res) / n
        return torch.sum(res)
