from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


## affine maps
lin = nn.Linear(5, 3,bias=False)  # maps from R^5 to R^3, parameters A, b
# data is 2x5.  A maps from 5 to 3... can we map "data" under A?
data = torch.ones(2, 5)

print(lin)
print(lin.weight)
print(lin.weight.shape)

print(data)
print(data.shape)


Ax = lin(data)
print(Ax)  # yes
print(Ax.shape)

import numpy as np
A = np.array(lin.weight.detach().numpy())
x = np.ones((2,5))

print(A)
print(np.dot(x,A.T))


# relu
data = torch.randn(2, 2)
print(data)
print(F.relu(data))


# softmax
data = torch.randn(5)
print(data)
print(F.softmax(data, dim=0))
print(F.softmax(data, dim=0).sum())  # Sums to 1 because it is a distribution!
print(F.log_softmax(data, dim=0))  # theres also log_softmax

