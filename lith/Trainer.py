import torch
import torch.optim
import torch.nn.functional as F
from torch.autograd import Variable


class Trainer(object):
    def __init__(self):
        # init variables
        self.epoch = 0

        # init by assignment
        self.model = None
        self.optimizer = None
        self.total_epochs = None

        self.use_cuda = 0

        # fb.resnet.torch scheduler
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer,
            milestones=[int(self.total_epochs * 0.5), int(self.total_epochs * 0.75)], gamma=0.1)

    def run(self):
        while self.epoch <= self.total_epochs:
            self.scheduler.step()
            self.train()
            self.test()
            self.epoch += 1

    def snapshot(self, is_best, error):
        torch.save({"model": self.model, "optimizer": self.optimizer, "error": error, "epoch": self.epoch})

    # train one epoch
    def train(self, trainloader, epoch):
        self.model.train()
        n_samples = len(trainloader.dataset)
        for batch_idx, (input, target) in trainloader:
            if self.use_cuda:
                input, target = input.cuda(), target.cuda()
            input, target = Variable(input), Variable(target)

            self.optimizer.zero_grad()
            output = self.model(input)
            loss = F.cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()
            # evaluation

            # log information

    def test(self, testloader, epoch):
        self.model.train()
        n_samples = len(testloader.dataset)
        for batch_idx, (input, target) in testloader:
            if self.use_cuda:
                input, target = input.cuda(), target.cuda()
            input, target = Variable(input), Variable(target)
            output = self.model(input)
            loss = F.cross_entropy(output, target)
            # evaluation

            # log information
        return

    def evaluate(self):
        pass

    def resume(self):
        pass

    def adjust_lr(self):
        pass

    def serialize(self):
        # serialize whole model
        pass
