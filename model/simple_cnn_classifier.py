import torch
from pytorch_lightining import metrics as torch_metrics
import numpy as np

class simple_cnn_classifier:
	def __init__(self):
		self.model = None

	def initialize_model(self):
		self.model = torch.nn.Sequential(
			torch.nn.Conv2d(1, 2, kernel_size=5),
			torch.nn.ReLU(),
			torch.nn.Conv2d(2, 4, kernel_size=7),
			torch.nn.ReLU(),
			torch.nn.Flatten(),
			torch.nn.Linear(1296, 512),
			torch.nn.ReLU(),
			torch.nn.Linear(512, 128),
			torch.nn.ReLU(),
			torch.nn.Linear(128, 32),
			torch.nn.ReLU(),
			torch.nn.Linear(32, 10),
			torch.nn.Softmax(dim=1),
		)
		return self.model