from __future__ import print_function
import torch

x = torch.empty(5, 3)
print("empty():", x)

x = torch.rand(5, 3)
print("rand():", x)

x = torch.zeros(5, 3, dtype=torch.long)
print("zeros(long):", x)

x = torch.zeros(5, 3, dtype=torch.float32)
print("zeros(float32):", x)

x = torch.tensor([5.5, 3])
print("List --> Tensor:", x)

x = x.new_ones(5, 3, dtype=torch.double)
print("new_ones(double):", x)

x = torch.randn_like(x, dtype=torch.float)
print("rand_like(float):", x)

print("Size of Tensor:", x.size())