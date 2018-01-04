import torch
import numpy as np

from lith import metric

m = metric.Error()

output = torch.randn([64, 5])
noise = torch.randn([64, 5])
pred = (output + noise).max(1)[1]

res = m(output, pred)
print(m.name, res, type(res))