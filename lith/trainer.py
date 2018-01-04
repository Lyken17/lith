import os.path as osp
import logging
import collections
from collections import OrderedDict

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math
import numpy as np
import torch

from .metric import Compose, Accuracy, Error


class Trainer(object):
    def __init__(self, model, train_loader, valid_loader, optimizer,
                 start_epoch=0, total_epoch=150,
                 criterion=nn.CrossEntropyLoss(), eval_criterion=Compose(Error()),
                 scheduler=None, root=".", name="base"):
        # init variables
        self.epoch = start_epoch

        # environment
        self.use_cuda = 0
        self.root = root
        self.name = name

        # dataset
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # training
        self.criterion = criterion
        self.eval_criterion = eval_criterion

        # init by assignment
        self.model = model
        self.optimizer = optimizer
        self.epoch = self.start_epoch = start_epoch
        self.total_epochs = total_epoch
        self.best_error = 100

        self.criterion = criterion

        # fb.resnet.torch scheduler for CIFAR
        self.scheduler = scheduler
        if scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=self.optimizer,
                milestones=[int(self.total_epochs * 0.5),
                            int(self.total_epochs * 0.75)],
                gamma=0.1
            )

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
            loss, error = self.valid(self.valid_loader, self.epoch)
            self.snapshot(error)
            self.epoch += 1

    def log_info(self, type, epoch, progress, output, target, loss=None):
        msg = "%s: %d [%d/%d](%.2f) " % (type, epoch, progress[0], progress[1], progress[0] / progress[1] * 100)
        if loss is not None:
            msg += "{}: {:.4f}\t".format("Loss", loss.data.cpu()[0])
        length = len(msg)

        evaluation = self.eval_criterion(output, target)
        for key, value in evaluation.items():
            current = "{}: {:.4f}\t".format(key, value)
            length += len(current)
            if length > 80:
                self.logger(msg)
                msg = ""
            msg += current
            length = len(msg)
        self.logger(msg)
        return evaluation

    # train one epoch
    def train(self, train_loader, optimizer, epoch):
        self.model.train()
        n_samples = len(train_loader.dataset)
        n_batchs = len(train_loader)
        for batch_idx, (input, target) in enumerate(train_loader):
            if self.use_cuda:
                input, target = input.cuda(), target.cuda()
            input, target = Variable(input), Variable(target)

            optimizer.zero_grad()
            output = self.model(input)
            loss = self.criterion(output, target)
            loss.backward()
            optimizer.step()
            # evaluation
            self.log_info("Train", epoch, [batch_idx, n_batchs], output, target, loss)
            # log information

    def valid(self, valid_loader, epoch):
        self.model.eval()
        n_samples = len(valid_loader.dataset)
        n_batchs = len(valid_loader)

        total_error = 0.0
        total_loss = 0.0

        for batch_idx, (input, target) in enumerate(valid_loader):
            if self.use_cuda:
                input, target = input.cuda(), target.cuda()
            input, target = Variable(input), Variable(target)
            output = self.model(input)
            loss = self.criterion(output, target)

            # evaluation
            evaluation = self.log_info("Valid", epoch, [batch_idx, n_batchs], output, target, loss)
            batch_size = input.size(0)
            total_error += evaluation["Error"] * batch_size
            total_loss += loss * batch_size

        return total_loss / n_samples, total_error / n_samples

    def evaluate(self, test_loader):
        raise NotImplementedError
        self.model.eval()
        n_samples = len(test_loader.dataset)
        for batch_idx, (input, target) in enumerate(test_loader):
            if self.use_cuda:
                input, target = input.cuda(), target.cuda()
            input, target = Variable(input), Variable(target)
            output = self.model(input)
            evaluation = self.eval_criterion(output, target)

    def serialize(self):
        raise NotImplementedError

    def snapshot(self,  error):
        is_best = error < self.best_error
        if is_best:
            self.best_error = error
            torch.save({
                "model": self.model,
                "optimizer": self.optimizer,
                "best": is_best,
                "error": error,
                "epoch": self.epoch
            }, osp.join(self.root, self.name, "model-best.t7"))

        torch.save({
            "model": self.model,
            "optimizer": self.optimizer,
            "best": is_best,
            "error": error,
            "epoch": self.epoch
        }, osp.join(self.root, self.name, "model-latest.t7"))

    def resume(self):
        pass
