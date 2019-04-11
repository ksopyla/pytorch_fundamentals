import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

# Classfication losses


## prepare the data

# we simulate 2x3 class classification problem, our model returns from last layer this output
# first 3 are for first feature, last 3 are for second feature, each can have 3 values
np_output1 = np.array([0.7, 0.1, 0.2, 0.1, 0.1, 0.8  ], dtype=np.float32)
np_output2 = np.array([0.1, 0.7, 0.2, 0.9, 0.05, 0.05  ], dtype=np.float32)

output1 = torch.tensor([ np_output1], requires_grad=True)
output2 = torch.tensor([np_output2], requires_grad=True)

# but we know our target class, encoded as one hot encoding
np_target = np.array([[1, 0, 0, 0 ,0, 1 ]], dtype=np.float32)
cls_target = torch.from_numpy(np_target)

print(f'output1={output1}\noutput2={output2} \ntarget={cls_target}')


# CrossEntropy - This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
# It is useful when training a classification problem with C classes.

print("CrossEntropyLoss")
loss = nn.CrossEntropyLoss()

target1=0
target2 = 2

class_number1 =torch.tensor([target1], dtype=torch.long)
class_number2 =torch.tensor([target2], dtype=torch.long)

# output1
loss_value1 = loss(output1[:,0:3], class_number1)
print(loss_value1)
loss_value2 = loss(output1[:, 3:], class_number2)
print(loss_value2)
loss_sum = loss_value1 + loss_value2
print(loss_sum)

loss_sum.backward()
print(output1.grad)

# output2
loss_value1 = loss(output2[:,0:3], class_number1)
print(loss_value1)
loss_value2 = loss(output2[:, 3:], class_number2)
print(loss_value2)
loss_sum = loss_value1 + loss_value2
print(loss_sum)

loss_sum.backward()
print(output2.grad)



# CrossEntropy for mini batch of size 2
print("CrossEntropyLoss - batch")
loss = nn.CrossEntropyLoss(reduction='none')

batch_output = torch.cat([output1, output2])
print(batch_output)

# target for each mini-batch exmaple, should be from [0,2] - 
batch_target1 = torch.tensor([target1, target1], dtype=torch.int64)
batch_target2 = torch.tensor( [target2, target2], dtype= torch.int64)


loss_value1 = loss(batch_output[:, 0:3], batch_target1)
print(loss_value1) 

loss_value2 = loss(batch_output[:,3:], batch_target2)
print(loss_value2)

loss_sum = loss_value1 + loss_value2
print(loss_sum)

loss_sum.sum().backward()
print(output1.grad)
print(output2.grad)

# CrossEntropy for mini batch of size 2 and reshape
print("CrossEntropyLoss - batch combined k-dim")
loss = nn.CrossEntropyLoss(reduction='none')
#loss = nn.CrossEntropyLoss(reduction='mean')

batch_output = torch.cat([output1, output2])
print(batch_output)
batch_output = batch_output.view(-1, 2, 3)
print(batch_output)

batch_output = batch_output.transpose(1,2)
print(batch_output)


batch_target = torch.tensor([ [target1, target2], [target1, target2]], dtype=torch.int64)


loss_value = loss(batch_output, batch_target)
print(loss_value) 


loss_value.sum().backward()
print(output1.grad)
print(output2.grad)