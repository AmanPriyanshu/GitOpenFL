import torch

x = torch.ones(2, 2, requires_grad=True)
print("Setting Requires Grad to True:", x)

y = x + 2
print("y = x + 2:", y)

print("Grad_fn:", y.grad_fn)

z = y * y * 3
out = z.mean()

print("z = 3*y^2 and out = mean(z):", z, out)

print('\nTrying requires_grad:\n')

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

print('\nGoing back to out now:\n')

out.backward()

print("d(out)/dx", x.grad)

print('\nPaussing autograd:\n')

print("Outside with torch.no_grad wrapper",x.requires_grad)
print("Outside with torch.no_grad wrapper",(x ** 2).requires_grad)

with torch.no_grad():
	print("Within with torch.no_grad wrapper",(x ** 2).requires_grad)

print("\nDetaching\n")

print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())