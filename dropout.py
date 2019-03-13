

import torch.nn as nn
import torch


prob=0.5
# define dropout
drop = nn.Dropout(p=prob)

# generate tensor
rows=2
columns=10
input = torch.ones(rows, columns)
print(f'input={input}')


# with probability 'prob' change each tensor value to 0
output = drop(input)

# count how many zeros, should be about rows*columns*prob zeros
# eg. 2*10*0.5 = 10
zeros = torch.sum( output==0 )
print(f'zeros={zeros}')
# ohhh! Where my '1' go? Each value was scaled by prob 1/prob, 
print(f'output={output}')


for b, i in enumerate(range(0, 20, 5)):
    print(b,i)
