from __future__ import print_function
import torch
import time

x = torch.empty(5, 3)
print(x)



r1, c1 = (150, 50)
r2, c2 = (150, 50)

a = torch.rand(r1, c1)
b = torch.rand(r2, c2)

t1 = time.perf_counter()

c = a*b

t2 = time.perf_counter()
print('cpu time={} result={}'.format(t2-t1,c.sum() ))


a = a.cuda()
b = b.cuda()
torch.cuda.synchronize()
torch.cuda.synchronize()

t1 = time.perf_counter()

c = a*b
torch.cuda.synchronize()
t2 = time.perf_counter()
print('gpu time={} result={}'.format(t2-t1,c.sum()))


