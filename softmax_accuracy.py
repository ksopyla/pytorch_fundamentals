'''
Example shows how to compute combined softmax for multilabel multivalue setting.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np


# in one output (6 dim vector) we want to encode 2 features and level of its expression
# each feature is represented by 3 values (low, mid, high)
# eg. in a multilabel classification problem we want to encode information that each label
# could have 3 stages (low,mid, high)
np_output1 = np.array([0.99, 0.005, 0.005, 0.1, 0.1, 0.8  ], dtype=np.float32)

np_output2 = np.array([0.1, 0.7, 0.2, 0.8, 0.05, 0.15 ], dtype=np.float32)

output1 = torch.tensor([ np_output1], requires_grad=True)
output2 = torch.tensor([np_output2], requires_grad=True)

output = torch.cat([output1, output2])
print(output)

# our target is encoded for crossentropy, contains index number 
# index 0 coressponds to 0.99, idex 2 coresponds to 0.8, second row accordingly for output2
np_target = np.array([[0, 2], [1,0]], dtype=np.int64)
cls_target = torch.from_numpy(np_target)

# batch_output = output
# print(batch_output)
# softmaxed = F.softmax(batch_output)
# print(softmaxed)
# print(softmaxed.sum(dim=1))


batch_output = output.view(-1, 2, 3)
print(batch_output)
softmaxed = F.softmax(batch_output, dim=2)
print(softmaxed)

_, positions = softmaxed.max(dim=2)

print(positions,cls_target)

correct = (cls_target== positions)
print(correct)
acc = correct.sum()/len(correct)

####
# batch_output = output.view(-1, 2, 3)
# batch_output = batch_output.transpose(1,2)
# print(batch_output)
# softmaxed = F.softmax(batch_output)
# print(softmaxed)
