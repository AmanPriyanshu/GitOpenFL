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

y = torch.rand(5, 3)
print("Arithmetic Operations similar to Numpy: x + y =", x + y)

print("OR")

print("torch.add(_, _):", torch.add(x, y))

print("You can also provide output tensor as an argument: torch.add(x, y, out=result) and the output will be stored in result.")

print("Shorthand Operators: y.add_(x):", y.add_(x))

print("Slicing of tensors allowed:", x[:, 1])

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print("x.view automatically reshapes x to flatten it first and then proceeds to calculate/infer the rest", x.size(), y.size(), z.size())

x_n = x.numpy()
print("tensor --> numpy: x.numpy()", type(x_n), "However, torch tensor tends to manipulate the numpy's values as well")

x = torch.from_numpy(x_n)
print("numpy --> tensor: torch.from_numpy(x_n)", type(x))

###

print('\n')

###

if torch.cuda.is_available():
	device = torch.device("cuda")
	print("Successful GPU Integeration")
	y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
	x = x.to(device)                       # or just use strings ``.to("cuda")``
	z = x + y
	print(z)
	print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!