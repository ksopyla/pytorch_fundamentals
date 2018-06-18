import torch

x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)
print(y.grad_fn)
print(y.grad)
print(x.grad)

z = y * y * 3

#out = z.mean()
out = 2*x[0,0]+3*x[0,1]*x[0,0]

print(z, out)

out.backward()

print(x.grad)
