import torch
import torch.nn as nn
import torch.nn.functional as F

### Understanding functions can be call stacked and always follows input, output. 
### To reshape, do not forget batch size, can cause issues, find out x.shape()[1:] to find out the current tensor shape
### whereas x.shape[0] is the batch size

class TutorialNet(nn.Module):

    def __init__(self):
        super(TutorialNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
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

print("Summary:")
tutorial_net = TutorialNet()
print(tutorial_net)

print("\nParameters:")
params = list(tutorial_net.parameters())
print(len(params))
print("Size of Layer 1's weights", params[0].size())
print("Size of Layer 1's bias", params[1].size())
print("Size of Layer 2's weights", params[2].size())
print("Size of Layer 2's bias", params[3].size())
print("Size of Layer 3's weights", params[4].size())
print("Size of Layer 3's bias", params[5].size())

print("\nImage dims calculation: 32 x 32 --CNN(3x3)--> 30 x 30 --max_pool2d(2x2)--> 15 x 15 --CNN(3x3)--> 13 x 13 --max_pool2d(2x2)--> 6 x 6\n")

print("Trying a random input")
input = torch.randn(1, 1, 32, 32)
out = tutorial_net(input)
print(out)

print("\nSetting Grads to zero first and then backpropogating.")
tutorial_net.zero_grad()
out.backward(torch.randn(1, 10))

print("\nLoss Function:")
output = tutorial_net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print("loss:", loss, "from expected output:", target)

print("\nTaking a look at the steps so far:\n")
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

print("\nBack Propogation:\n")
tutorial_net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print("After setting grads to zero:", tutorial_net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print("After backpropogating once:", tutorial_net.conv1.bias.grad)

print("\nUpdating weights once:\n")

output = tutorial_net(input)
loss = criterion(output, target)
print("Before Update Loss:", loss)

learning_rate = 0.01
for f in tutorial_net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

output = tutorial_net(input)
loss = criterion(output, target)
print("After Update Loss:", loss)

print("\nUsing an optimizer once:\n")

import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(tutorial_net.parameters(), lr=0.01)

output = tutorial_net(input)
loss = criterion(output, target)
print("Before SGD Update Loss:", loss)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = tutorial_net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update

output = tutorial_net(input)
loss = criterion(output, target)
print("After SGD Update Loss:", loss)

print("\nUsing an optimizer n-times(10):\n")

output = tutorial_net(input)
loss = criterion(output, target)
print("Before SGD Update Loss:", loss)

# in your training loop:
for i in range(10):
    optimizer.zero_grad()   # zero the gradient buffers
    output = tutorial_net(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()    # Does the update

    output = tutorial_net(input)
    loss = criterion(output, target)
    print("Epoch "+str(i+1)+": Loss=", loss)