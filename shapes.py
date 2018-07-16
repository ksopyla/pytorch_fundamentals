from __future__ import print_function
import torch
import time


torch.manual_seed(1)

x = torch.randn(6, 4)
print(x)
print(x.shape)


x = x.view(-1)
print(x)
print(x.shape)

print("3D")
x = torch.randn(6, 4,2)
print(x)
print(x.shape)


# first dim=1 6->1 rest compute automatically
x = x.view(1,-1) 
print(x)
print(x.shape)


# first dim=2 6->2 rest compute automatically
x = x.view(2,-1) 
print(x)
print(x.shape)