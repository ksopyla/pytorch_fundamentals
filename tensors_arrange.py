from __future__ import print_function
import torch
import time


# split tensors

a = torch.randn(50, 80) #tensor of size 50 x 80
b = torch.split(a, 20, dim=1) # it returns a tuple of 4 elements, each is a tensor 50x20
b = list(b) # convert to list if you want
print(f'shape of the first element {b[0].shape}')
print(f'shape of the last element {b[-1].shape}')


a = torch.randn(10, 6, 8) # 3d tensor
b = torch.split(a, 4, dim=0) # it returns a tuple of 3 elements, split by 4 elements [4,4,2], 4x6x8, last is shorther 2x6x8
#b = list(b) # convert to list if you want
print(f'shape of the first element {b[0].shape}')
print(f'shape of the last element {b[-1].shape}')

b = torch.split(a, 4, dim=1) # now we split through dim=1, creates tuple of 2 elements [4,2]
#b = list(b) # convert to list if you want
print(f'shape of the first element {b[0].shape}')
print(f'shape of the last element {b[-1].shape}')

# chunk works different then split, we have provide how many chunks we want
# 10 we want to divide to 4 chunks 10/4= 3 (1) [3,3,3,1]
b = torch.chunk(a, 4, dim=0) # now we split through dim=1
#b = list(b) # convert to list if you want
print(f'shape of the first element {b[0].shape}')
print(f'shape of the last element {b[-1].shape}')
