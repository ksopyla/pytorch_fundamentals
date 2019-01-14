import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

# classfication losses

## prepare the data

# we simulate 3 class classifiction problem, our classifier retunrs from last layer this output
np_output1 = np.array([[0.5, 0.1, 0.3 ]], dtype=np.float32)
np_output2 = np.array([[0.1, 0.5, 0.3 ]], dtype=np.float32)
output1 = torch.from_numpy(np_output1)
print(output1)
output2 = torch.from_numpy(np_output2)
print(output2)

# but we know our target class, encoded as one hot encoding
np_target = np_output2 = np.array([[1, 0, 0 ]], dtype=np.float32)
target = torch.from_numpy(np_target)

# L1 loss
loss = nn.L1Loss()
loss_value = loss(output1, target)
print(loss_value) # (|0.5-1| + |0.1-0| + |0.3-0|)/3 = (0.5+0.1+0.3)/3 = 0.9/3 = 0.3

# for second wrong output the loss should be higher
loss_value = loss(output2, target)
print(loss_value) # (|0.1-1| + |0.5-0| + |0.3-0|)/3 = (0.9+0.5+0.3)/3 = 1.7/3 = 0.5667


# L2 loss - MSE - mean square error
loss = nn.MSELoss()
loss_value = loss(output1, target)
print(loss_value) # ((0.5-1)^2 + (0.1-0)^2 + (0.3-0)^2)/3 = 0.1167

# for second wrong output the loss should be higher
loss_value = loss(output2, target)
print(loss_value) # ((0.1-1)^2 + (0.5-0)^2 + (0.3-0)^2)/3 = 0.3833


# CrossEntropy - This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
# It is useful when training a classification problem with C classes.
loss = nn.CrossEntropyLoss()

class_number =torch.tensor([0], dtype=torch.long)
loss_value = loss(output1, class_number)
print(loss_value) # =

class_number =torch.tensor([1], dtype=torch.long)
loss_value = loss(output1, class_number)
print(loss_value) # =


# CrossEntropy for mini batch of size 3
loss = nn.CrossEntropyLoss()

# now we have 4 output classes and 3 examples in mini-batch
batch_output = torch.randn(3, 4)
# target for each mini-batch exmaple, should be from [0,3] - 
batch_target = torch.empty(3, dtype=torch.long).random_(4)
loss_value = loss(batch_output, batch_target)
print(loss_value) # =

