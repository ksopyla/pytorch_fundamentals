import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1,6, (5,5))
        self.conv2 = nn.Conv2d(6,16,5)

        # fully connected, input must be 32, because after polling and convolution operationr our image resize to 5x5 pixel
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,80)
        self.fc3 = nn.Linear(80,10)

    def forward(self, x):

        print("shape 1 x={}".format(x.shape))
        x = F.relu(self.conv1(x))
        print("shape 2 x={}".format(x.shape))
        x = F.max_pool2d(x, (2,2))
        print("shape 3 x={}".format(x.shape))
        x = F.relu(self.conv2(x))
        print("shape 4 x={}".format(x.shape))
        x = F.max_pool2d(x, 2)
        print("shape 5 x={}".format(x.shape))
        x = x.view(-1, self.num_flat_features(x))
        print("shape4 x={}".format(x.shape))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight    

for a in params:
    print(a.size())  # conv1's .weight    
    
        
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

net.zero_grad()
# dx = 0.01*torch.ones(1,10)
# out.backward(torch.randn(1, 10))


output = net(input)
target = torch.arange(1, 11)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)


# weight = weight - learning_rate * gradient
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)



import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update