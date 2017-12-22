import math
import numpy as np
import torch


class Metric(object):
    def __init__(self):
        pass

    def torchlize(func):
        # preprocess the args into torch tensor
        def convert(*args, **kwargs):
            new_args = []
            for idx, each in enumerate(args):
                if isinstance(each, np.ndarray):
                    each = torch.from_numpy(each)
                new_args.append(each)
            args = tuple(new_args)

            for key, value in kwargs:
                if isinstance(value, np.ndarray):
                    kwargs[key] = torch.from_numpy(value)
            return func(*args, **kwargs)

        return convert


class TopKAccuracy(Metric):
    def __init__(self, k):
        super(TopKAccuracy, self).__init__()
        self.k = k

    @Metric.torchlize
    def __call__(self, output, target):
        assert output.size() == target.size(), \
            '''To compute TopK, `output` must have same shape as `target` '''
        sorted, indices = torch.topk(output, self.k, output.dim() - 1)
        comp = indices - target
        correct = comp.eq(0).cpu().sum()
        print(indices.size(), target.size(), correct, output.size(0))
        return correct / output.size(0)


class Accuracy(TopKAccuracy):
    def __init__(self):
        super(Accuracy, self).__init__(k=1)


class MeanSquareError(Metric):
    @Metric.torchlize
    def __call__(self, output, target):
        assert output.size() == target.size(), "To compute MSE, output must have same shape as target"
        return torch.mean((output - target) ** 2)


if __name__ == "__main__":
    m = Accuracy()
    a = np.random.random_sample([100, 100, 5])
    b = np.random.randint(0, 5, [100, 100, 1])
    print(m(a, b))
