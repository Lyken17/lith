import os.path as osp
import logging
from collections import OrderedDict

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math
import numpy as np
import torch


class Metric(object):
    def __init__(self):
        pass

    @property
    def name(self):
        return self.__class__.__name__

    def torchlize(func):
        # preprocess the args from numpy into torch tensor
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


class Compose(Metric):
    def __init__(self, metrics):
        super(Compose, self).__init__()
        if not isinstance(metrics, list):
            self.metrics = metrics

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
        :param output: *xC
        :param target: *x1
        :return:
        """
        assert output.dim() == target.dim(), \
            '''To compute TopK, `output` must have same shape as `target` '''
        sorted, indices = torch.topk(output, self.k, output.dim() - 1)
        comp = indices - target
        correct = comp.eq(0).cpu().sum()
        # print(indices.size(), target.size(), correct, output.size(0))
        return correct / output.size(0)


class Accuracy(TopKAccuracy):
    def __init__(self):
        super(Accuracy, self).__init__(k=1)


class Trainer(object):
    def __init__(self, model, train_loader, valid_loader,
                 start_epoch=0, total_epoch=150,
                 criterion=nn.CrossEntropyLoss(), eval_criterion=Compose(Accuracy()),
                 scheduler=None, root=".", name="base"):
        # init variables
        self.epoch = start_epoch

        # environment
        self.use_cuda = 0
        self.root = root
        self.name = name
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion

        # init by assignment
        self.model = model
        self.optimizer = None
        self.epoch = self.start_epoch = start_epoch
        self.total_epochs = total_epoch

        self.criterion = criterion
        self.eval_criterion = 1

        # fb.resnet.torch scheduler for CIFAR
        self.scheduler = scheduler
        if scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer,
                milestones=[int(self.total_epochs * 0.5), int(self.total_epochs * 0.75)], gamma=0.1)

        # other variables
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(levelname)s: %(message)s',
                            datefmt='%b/%d[%H:%M:%S]',
                            filename=osp.join(self.root, '%s.log' % self.name),
                            filemode='w')

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
        self.logger = logging.info

    def run(self):
        while self.epoch <= self.total_epochs:
            self.scheduler.step(self.epoch)
            self.train(self.train_loader, self.optimizer, self.epoch)
            self.valid(self.valid_loader, self.epoch)
            self.snapshot()
            self.epoch += 1

    def log_info(self, type, epoch, progress, output, target):
        msg = "%s: %d [%d/%d]" % (type, epoch, progress[0], progress[1])
        evaluation = self.eval_criterion(output, target)
        length = len(msg)
        for key, value in evaluation.items():
            current = "{}: {}".format(key, value)
            length += len(current)
            if length > 80:
                msg += "\n"
                length = len(current)
            msg += current

    # train one epoch
    def train(self, train_loader, optimizer, epoch):
        self.model.train()
        n_samples = len(train_loader.dataset)
        n_batchs = len(train_loader)
        for batch_idx, (input, target) in train_loader:
            if self.use_cuda:
                input, target = input.cuda(), target.cuda()
            input, target = Variable(input), Variable(target)

            optimizer.zero_grad()
            output = self.model(input)
            loss = self.criterion(output, target)
            loss.backward()
            optimizer.step()
            # evaluation

            # log information

    def valid(self, valid_loader , epoch):
        self.model.eval()
        n_samples = len(valid_loader.dataset)
        for batch_idx, (input, target) in valid_loader:
            if self.use_cuda:
                input, target = input.cuda(), target.cuda()
            input, target = Variable(input), Variable(target)
            output = self.model(input)
            loss = self.criterion(output, target)
            # evaluation

            # log information
        return

    def evaluate(self, test_loader):
        self.model.eval()
        n_samples = len(test_loader.dataset)
        for batch_idx, (input, target) in test_loader:
            if self.use_cuda:
                input, target = input.cuda(), target.cuda()
            input, target = Variable(input), Variable(target)
            output = self.model(input)
            loss = self.criterion(output, target)

    def serialize(self):
        raise NotImplementedError

    def snapshot(self, is_best, error):
        if is_best:
            torch.save({
                "model": self.model,
                "optimizer": self.optimizer,
                "best": is_best,
                "error": error,
                "epoch": self.epoch
            }, osp.join(self.root, "model-best.t7"))

        torch.save({
            "model": self.model,
            "optimizer": self.optimizer,
            "best": is_best,
            "error": error,
            "epoch": self.epoch
        }, osp.join(self.root, "model-latest.t7"))

    def resume(self):
        pass

