import torch

import torchvision.transforms


class colorspace(object):
    def __init__(self):
        self.trans = torch.ones(3, 3)

    def __call__(self, tensor):
        return torch.multi(self.trans, tensor)


class RGB2GRAY(object):
    def __init__(self):
        super(RGB2GRAY, self).__init__()
        self.trans = torch.ones(3, 3)
        raise NotImplementedError

class GRAY2RGB(colorspace):
    def __init__(self):
        super(GRAY2RGB, self).__init__()
        self.trans = torch.eigen(3, 3)
        raise NotImplementedError

class RGB2YUV(colorspace):
    def __init__(self):
        super(RGB2YUV, self).__init__()
        raise NotImplementedError


class YUV2RGB(colorspace):
    def __init__(self):
        super(YUV2RGB, self).__init__()
        raise NotImplementedError
