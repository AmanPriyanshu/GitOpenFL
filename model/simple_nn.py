import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange
import numpy as np
from sklearn import metrics as sklearn_metrics

np.random.seed(1)
torch.manual_seed(1)

class SimpleNet(nn.Module):

	def __init__(self):
		super(SimpleNet, self).__init__()
		self.fc1 = nn.Linear(4, 3)  # 6*6 from image dimension
		self.fc2 = nn.Linear(3, 2)
		self.fc3 = nn.Linear(2, 1)
		self.criterion = nn.MSELoss()
		self.optimizer = None
		self.epochs = None
		self.batches = None
		self.x = None
		self.y = None
		self.x_test = None
		self.y_test = None
		self.history = []
		
	def forward(self, x):
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		x = torch.sigmoid(self.fc3(x))
		return x

	def _backward(self, loss):
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

	def compile(self, optimizer, metrics=[]):
		self.optimizer = optimizer
		self.metrics = metrics

	def _epoch_end(self, description, x, y, prefix=''):
		with torch.no_grad():
			output = self.forward(x)

		y = y.numpy()
		output = output.numpy()

		for m in self.metrics:
			try:
				description.update({prefix+'_'+str(m.__name__):round(m(y, output), 3)})
			except (ValueError):
				description.update({prefix+'_'+str(m.__name__):round(m(y, np.array([1 if i>0.5 else 0 for i in output])), 3)})

		self.batches.set_description(str(description))
		self.batches.refresh()
		return description

	def _on_epoch_end(self, running_loss, epoch):
		description = {'epoch':str((epoch+1))+'/'+str(self.epochs), 'running_loss':str(round(running_loss.item(), 3))}
		description = self._epoch_end(description, self.x, self.y, prefix='train')
		if self.x_test is not None:
			description = self._epoch_end(description, self.x_test, self.y_test, prefix='test')
		self.history.append(description)
		return description

	def _on_training_end(self):
		desc = {params:[] for params in self.history[0].keys()}
		for epoch in self.history:
			for key, item in epoch.items():
				desc[key].append(item)
		self.history = desc

	def fit(self, x, y, batch_size, epochs, x_test=None, y_test=None, drop_remaining=True, verbose=0):
		self.x, self.y, self.x_test, self.y_test = x, y, x_test, y_test
		self.epochs = epochs
		for epoch in range(epochs):
			running_loss = torch.tensor([0], dtype=torch.float32)
			batch_idx = 0
			description = {'epoch':str((epoch+1))+'/'+str(epochs), 'running_loss':str(round(running_loss.item()/(batch_idx+1), 3))}
			self.batches = trange(self.x.size(0)//batch_size, desc='running_loss: inf', disable = True if verbose>0 else False)
			description = {}
			for batch_idx in self.batches:
				output = self.forward(self.x[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)])
				target = self.y[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)]
				loss = self.criterion(output, target)
				self._backward(loss)
				running_loss += loss
				description = {'epoch':str((epoch+1))+'/'+str(epochs), 'running_loss':str(round(running_loss.item()/(batch_idx+1), 3))}
				self.batches.set_description(str(description))
				self.batches.refresh()
				if batch_idx==self.x.size(0)//batch_size - 1:
					description = self._on_epoch_end(running_loss/(batch_idx+1), epoch)
					if verbose==1:
						print(str(description))
		self._on_training_end()
		return self.history

def data_gen_example():
	x = np.array([np.random.random(4) for _ in range(100)] + [-np.random.random(4) for _ in range(100)], dtype=np.float32)
	x = torch.from_numpy(x)
	y = np.array([[0] for _ in range(100)] + [[1] for _ in range(100)], dtype=np.float32)
	y = torch.from_numpy(y)
	indexes = np.arange(200)
	np.random.shuffle(indexes)
	x = x[indexes]
	y = y[indexes]
	return x, y

if __name__ == '__main__':

	x,y = data_gen_example()

	net = SimpleNet()

	optimizer = optim.Adam(net.parameters(), lr=0.01)
	metrics = [sklearn_metrics.accuracy_score, sklearn_metrics.roc_auc_score]
	net.compile(optimizer, metrics)

	x_test, y_test = x[int(0.9*x.size(0)):], y[int(0.9*x.size(0)):]
	x, y = x[:int(0.9*x.size(0))], y[:int(0.9*x.size(0))]

	history = net.fit(x, y, 50, 10, x_test, y_test, verbose=0)

	print('\n\n',history)