import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


def init_inputs(num=1, seq_len=4, shape=(1, 2)):
    inputs = list()
    for i in range(num):
        # i*seq_len+(k+1)
        inputs.append([torch.ones(shape) for k in range(seq_len)])
    return inputs


def init_hidden():
    return (2*torch.ones(1, 1, 3), 2*torch.ones(1, 1, 3))


lstm = nn.LSTM(2, 3)  # Input dim is 2, output dim is 3

# inputs = [torch.ones(1, 2) for _ in range(4)]  # make a sequence of length 5
inputs = init_inputs(1)

# initialize the hidden state.
hidden = init_hidden()
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

hidden = init_hidden()
out, hidden = lstm(inputs2, hidden)
print(out)
print(hidden)

inputs_batch = [[torch.ones(1, 2) for _ in range(4)],
                [2*torch.ones(1, 2) for _ in range(4)]]
print(inputs_batch)

input_tensor = torch.zeros(2, 4, 2)

for i, seq in enumerate(inputs_batch):
    print(seq, i)
    input_tensor[i] = torch.cat(inputs_batch[i])
# permutate dimensions not values!!
input_tensor.permute([1,0,2])

inputs3 = torch.cat(inputs_batch)
print(inputs3)
inputs3 = inputs3.view(len(inputs), 2, -1)
print(inputs3)
hidden = init_hidden()
out, hidden = lstm(inputs3, hidden)
print(out)
print(hidden3)
