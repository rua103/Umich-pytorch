import torch

x = torch.arange(24)
print('Here is x:')
print(x)
y = x.view(2, 3, 4).transpose(0, 1).reshape(3, 8)
print('Here is y:')
print(y)

