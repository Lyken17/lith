import torch
import numpy as np

from lith import metric

m = metric.IoU()

output = torch.randn([10, 3, 32, 32])
noise = torch.randn([10, 3, 32, 32]) / 5

target = torch.randn([10, 1, 32, 32])
pred = (output + noise).max(1)[1]
# pred = pred.view(pred.size(0), 1, pred.size(1), pred.size(2))

print(pred.size())
print(m(output, pred))