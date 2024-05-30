import torch.nn as nn
import torch


input = torch.randn(1, 3, 256, 256)# N=1,

c = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)

output = c(input)

print('size of input:', input.shape)

print('size of output:', output.shape)


