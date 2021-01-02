### Aim: To check connections with the different clients and their branches, etc.
### as well as to extract weights when ready.

import requests

class ClientConnector:

	def __init__(self, cfg_file='./client_usernames.cfg'):
		self.client_ids = []
		self.cfg_file = cfg_file
		self.derive_git_id()

	def derive_git_id(self):
		file = open(self.cfg_file)
		client_details = [i.replace('\n', '') for i in file.readlines()[1:]]
		file.close()
		for client in client_details:
			if '--' not in client:
				client = client + ' -- main'
			self.client_ids.append({'username': client[:client.index(' --')], 'branch': client[4+client.index(' -- '):]})

if __name__ == '__main__':
	c = ClientConnector()
	print(c.client_ids)
	#c.history_check()