from lith import meter
import numpy as np

m = meter.MovingAverageMeter(momemtum=0.1)

for i in range(50):
    ran = i + np.random.randn() * 5
    m.add(ran)
    print("%.4f, %.4f" % (ran, m.value()))

