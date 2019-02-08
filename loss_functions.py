import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

# Classfication losses


## prepare the data

# we simulate 3 class classification problem, our model retunrs from last layer this output
np_output1 = np.array([[0.5, 0.1, 0.4 ]], dtype=np.float32)
np_output2 = np.array([[0.1, 0.5, 0.3 ]], dtype=np.float32)
output1 = torch.from_numpy(np_output1)
output2 = torch.from_numpy(np_output2)
target1 = 0
target2 = 1 
print(f'output1={output1} target={target1}')
print(f'output1={output2} target={target2}')

# but we know our target class, encoded as one hot encoding
np_target = np.array([[1, 0, 0 ]], dtype=np.float32)
cls_target = torch.from_numpy(np_target)

# L1 loss
print("L1")
loss = nn.L1Loss()
loss_value = loss(output1, cls_target)
print(loss_value) # (|0.5-1| + |0.1-0| + |0.4-0|)/3 = (0.5+0.1+0.4)/3 = 1.0/3 = 0.333

# for second wrong output the loss should be higher
loss_value = loss(output2, cls_target)
print(loss_value) # (|0.1-1| + |0.5-0| + |0.3-0|)/3 = (0.9+0.5+0.3)/3 = 1.7/3 = 0.5667





# CrossEntropy - This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
# It is useful when training a classification problem with C classes.

print("CrossEntropyLoss")
loss = nn.CrossEntropyLoss()

class_number =torch.tensor([target1], dtype=torch.long)
loss_value = loss(output1, class_number)
print(loss_value) # =

class_number =torch.tensor([target2], dtype=torch.long)
loss_value = loss(output2, class_number)
print(loss_value) # =


# CrossEntropy for mini batch of size 2
print("CrossEntropyLoss - batch")
loss = nn.CrossEntropyLoss(reduction='none')
# input = torch.randn(3, 5)
# target = torch.empty(3, dtype=torch.long).random_(5)

# now we have 3 output classes and 2 examples in mini-batch
#batch_output = torch.stack([output1, output2])
batch_output = torch.cat([output1, output2])
# target for each mini-batch exmaple, should be from [0,2] - 
batch_target = torch.tensor( [target1, target2], dtype= torch.int64)
loss_value = loss(batch_output, batch_target)
print(loss_value) # =


# regression problem 


# we simulate 3 variable regression problem, our model returns from last layer this output
np_output1 = np.array([-0.9, 3.3, 4.5 ], dtype=np.float32)
np_output2 = np.array([5., -1, 3. ], dtype=np.float32)
# but we know our target is
np_target = np.array([-1., 3., 4. ], dtype=np.float32)

# make tensors
output1 = torch.from_numpy(np_output1)
output2 = torch.from_numpy(np_output2)
reg_target = torch.from_numpy(np_target)


# L2 loss - MSE - mean square error
print("MSE Loss")
print(f'output1={output1}')
print(f'output2={output2}')
print(f'target={reg_target}')


loss = nn.MSELoss()
loss_value = loss(output1, reg_target)
print(f'loss(output1, target)={loss_value}') # ((-0.9- -1)^2 + (3.3-3)^2 + (4.5-4)^2)/3 = 0.1167

# for second wrong output the loss should be higher
loss_value = loss(output2, reg_target)
print(f'loss(output2, target)={loss_value}')  # = 17.666

print("MSEs - without reduction")
loss = nn.MSELoss(reduction='none')
loss_value = loss(output1, reg_target)
print(f'loss(output1, target)={loss_value}') # [(-0.9- -1)^2 ,  (3.3-3)^2,  (4.5-4)^2 ] = [ 0.01, 0.09, 0.25]


# Multi-label

# BCE loss

# we have batch_size=2 (one example in a row), multi label problem 
# this should be returned in the last layer of our model
# data shouldbe numbers from [0,1]
np_output1 = np.array([[0.1, 0.9], [0.6, 0.8]], dtype=np.float32)
np_output2 = np.array([[0.4, 0.5], [0.1, 0.9]], dtype=np.float32)
# but we know our target is
np_target = np.array([[0.1, 0.9],[0.5, 0.7]], dtype=np.float32)

# make tensors
output1 = torch.from_numpy(np_output1)
output2 = torch.from_numpy(np_output2)
reg_target = torch.from_numpy(np_target)

# Binary Cross Entropy
print("BCELoss")
print(f'output1={output1}')
print(f'output2={output2}')
print(f'target={reg_target}')


loss = nn.BCELoss()

# L= sum_i { l_i = -1*[ y_i * log(x_i) + (1-y_i)*log(1-x_i) ]}
loss_value = loss(output1, reg_target)
print(f'loss(output1, target)={loss_value}') # 

# for second wrong output the loss should be higher
loss_value = loss(output2, reg_target)
print(f'loss(output2, target)={loss_value}')  # = 

print("BCELoss - without reduction")
loss = nn.BCELoss(reduction='none')
loss_value = loss(output1, reg_target)
print(f'loss(output1, target)={loss_value}') # =


print(f'output1={output1}')
print(f'output2={output2}')
print(f'target={reg_target}')

reg_target2 = torch.FloatTensor(2,2).random_(2)


lst = []
for i in range(len(output1)):
    lst.append(-np.log(x[i])*y[i] + -np.log(1-x[i])*(1-y[i]))
lst, np.mean(lst)