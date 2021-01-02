import torch
import numpy as np
import pandas as pd
import os
import importlib

class ClientSide:
	def __init__(self, current_epoch=0, cfg_path='fl.cfg'):
		self.current_epoch = current_epoch
		self.cfg_path = cfg_path
		self.details = self.extract_details()
		self.client_epochs = self.details['client-epochs']
		self.model = self.get_model()

	def extract_details(self):
		file = open(self.cfg_path, 'r')
		details = [line.replace('\n', '') for line in file.readlines()]
		file.close()
		details = {line[:line.index(':')].strip(): line[line.index(':')+2:].strip() for line in details}
		details['client-epochs'] = int(details['client-epochs'])
		details['minimum-number-of-clients-per-aggregation'] = int(details['minimum-number-of-clients-per-aggregation'])
		details['number-of-aggregations'] = int(details['number-of-aggregations'])
		return details

	def get_model(self):
		ModelClass = importlib.import_module('model.'+self.details['model'])
		return ModelClass.initialize_model()

	def get_weights(self, dtype=np.float32):
		weights = []
		for layer in self.model:
			try:
				weights.append([layer.weight.detach().numpy().astype(dtype), layer.bias.detach().numpy().astype(dtype)])
			except:
				continue
		return np.array(weights)

	def set_weights(self, weights):
		index = 0
		for layer_no, layer in enumerate(self.model):
			try:
				_ = self.model[layer_no].weight
				self.model[layer_no].weight = torch.nn.Parameter(weights[index][0])
				self.model[layer_no].bias = torch.nn.Parameter(weights[index][1])
				index += 1
			except:
				continue

	def train(self):



if __name__ == '__main__':
	cs = ClientSide()
	cs.train()