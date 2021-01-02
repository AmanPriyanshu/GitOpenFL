import numpy as np
import torch

class weight_manager:
	def __init__(self):
		self.something = None

	def average_weights(self, all_weights):
		all_weights = np.array(all_weights)
		all_weights = np.mean(all_weights, axis=0)
		all_weights = [[torch.from_numpy(i[0].astype(np.float32)), torch.from_numpy(i[1].astype(np.float32))] for i in all_weights]
		return all_weights