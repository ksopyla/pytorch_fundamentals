import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


def init_inputs(num=1, seq_len=4, shape=(1, 2)):
    inputs = list()
    for i in range(num):
        # i*seq_len+(k+1)
        inputs.append([(i+1)*torch.ones(shape) for k in range(seq_len)])
    return inputs


def init_hidden(batch_size=1, hidden_size=3):
    return (2*torch.ones(1, batch_size, hidden_size), 2*torch.ones(1, batch_size, hidden_size))




lstm = nn.LSTM(2, 3)  # Input dim is 2, output dim is 3
inputs = init_inputs(num=1, shape=(1,2))
# initialize the hidden state.
hidden = init_hidden(hidden_size=3)

for seq in inputs:
    for i in seq:
        # Step through the sequence one element at a time.
        # after each step, hidden contains the hidden state.
        out, hidden = lstm(i.view(1, 1, -1), hidden)
        print(out, hidden)

# alternatively, we can do the entire sequence all at once.
# the first value returned by LSTM is all of the hidden states throughout
# the sequence. the second is just the most recent hidden state
# (compare the last slice of "out" with "hidden" below, they are the same)
# The reason for this is that:
# "out" will give you access to all hidden states in the sequence
# "hidden" will allow you to continue the sequence and backpropagate,
# by passing it as an argument  to the lstm at a later time
# Add the extra 2nd dimension

print(inputs)
inputs = inputs[0]
print(inputs)
inputs2 = torch.cat(inputs)
print(inputs2)
inputs2 = inputs2.view(len(inputs2), 1, -1)
print(inputs2)

hidden = init_hidden(hidden_size=3)
out, hidden = lstm(inputs2, hidden)
print(out)
print(hidden)


# cretea 2D list of list of tensors of default shape
# each list symbolize a sequence(time series) of items,
# each item is tensor of shape (1,dim) eg it could be one word embeding
inputs_batch = init_inputs(2)
print(inputs_batch)

# create 3D tensor, this is placeholder for transformed inputs
# this tensor has shapes [batch, seq_len, seq_item_size]
# tensor([
#         # 1 batch
#         [
#  1 seq item  [ 0.,  0.],
#  2 seq item  [ 0.,  0.],
#  3 seq item  [ 0.,  0.],
#  4 seq item  [ 0.,  0.]
#         ],
#         # 2 batch
#         [
#  1 seq item  [ 0.,  0.],
#  2 seq item  [ 0.,  0.],
#  3 seq item  [ 0.,  0.],
#  4 seq item  [ 0.,  0.]
#         ],
#        ])
input_tensor = torch.zeros(2, 4, 2)

# go through the list of sequences and concat all sequence items into 2D tensor
for i, seq in enumerate(inputs_batch):
    print(seq, i)
    # put each 2D sequence of shape [seq_len, seq_item_size]
    # into 3D tensor, first dim is a batch
    input_tensor[i] = torch.cat(inputs_batch[i])
# we have 3D tensor

# permutate dimensions in order to have [seq_len, batch, seq_item_size]
input_tensor = input_tensor.permute([1, 0, 2])
print(input_tensor)

hidden = init_hidden(batch_size=2, hidden_size=3)
out, hidden = lstm(input_tensor, hidden)
print(out)
print(hidden)


# inputs3 = torch.cat(inputs_batch)
# print(inputs3)
# inputs3 = inputs3.view(len(inputs), 2, -1)
# print(inputs3)
# hidden = init_hidden()
# out, hidden = lstm(inputs3, hidden)
# print(out)
# print(hidden3)
