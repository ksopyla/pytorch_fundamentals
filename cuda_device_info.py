from __future__ import print_function
import torch
import time

x = torch.empty(5, 3)
print(x)

# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    print(device)

    print(torch.cuda.get_device_properties(0))
else:
    print('GPU not enabled')
    
