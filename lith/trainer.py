import os, os.path as osp, shutil
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
                 scheduler=None, use_cuda=False, root=".", name="base", resume=False):
        # dataset
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.eval_criterion = eval_criterion

        # environment
        self.use_cuda = use_cuda
        self.resume = resume
        self.root = root
        self.name = name
        self.dir = osp.join(self.root, self.name)
        if osp.exists(self.dir) and not self.resume:
            shutil.rmtree(self.dir)
        os.makedirs(self.dir, exist_ok=True)

        # init variables
        self.epoch = start_epoch

        # init by assignment
        self.model = model
        self.optimizer = optimizer
        self.epoch = self.start_epoch = start_epoch
        self.total_epochs = total_epoch
        self.best_error = 100
        self.write_mode = "w+"

        if osp.exists(self.dir) and self.resume:
            archived = torch.load(osp.join(self.dir, "model-latest.pth"))
            self.model.load_state_dict(archived["model"])
            self.optimizer.load_state_dict(archived["optimizer"])
            self.best_error = archived["error"]
            self.epoch = self.start_epoch = archived["epoch"]
            self.write_mode = "a+"

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
                            filename=osp.join(self.dir, 'log.txt'),
                            filemode=self.write_mode)

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
        self.logger = logging.info

    def run(self, start_epoch=None, total_epochs=None):
        if start_epoch is not None:
            self.epoch = start_epoch
        if total_epochs is not None:
            self.total_epochs = total_epochs

        while self.epoch <= self.total_epochs:
            self.scheduler.step(self.epoch)
            self.train(self.train_loader, self.optimizer, self.epoch)
            loss, error = self.valid(self.valid_loader, self.epoch)
            self.snapshot(error)
            self.epoch += 1

    def log_info(self, type, epoch, progress, output, target, loss=None):
        msg = "%s: %d [%d/%d](%.2f%%)\t" % (type, epoch, progress[0], progress[1], progress[0] / progress[1] * 100)
        if loss is not None:
            msg += "{}: {:.4f}\t".format("Loss", loss.data.cpu()[0])
        length = len(msg)

        evaluation = self.eval_criterion(output, target)
        for key, value in evaluation.items():
            current = "{}: {:.4f}\t".format(key, value)
            length += len(current)
            if length > 120:
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
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best": is_best,
                "error": error,
                "epoch": self.epoch
            }, osp.join(self.root, self.name, "model-best.pth"))

        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best": is_best,
            "error": error,
            "epoch": self.epoch
        }, osp.join(self.root, self.name, "model-latest.pth"))

    def load_state_dict(self):
        pass
