### Aim: To check connections with the different clients and their branches, etc.
### as well as to extract weights when ready.

import requests

class ClientConnector:

	def __init__(self):
		self.client_ids = []
		self.cfg_file = './client_usernames.cfg'
		self.derive_git_id()
		self.check_existence()

	def derive_git_id(self):
		file = open(self.cfg_file)
		client_details = [i.replace('\n', '') for i in file.readlines()[1:]]
		file.close()
		for client in client_details:
			if '--' not in client:
				client = client + ' -- main'
			self.client_ids.append({'username': client[:client.index(' --')], 'branch': client[4+client.index(' -- '):]})

	def check_existence(self):
		links = ['https://github.com/'+client['username']+'/GitOpenFL/tree/'+client['branch'] for client in self.client_ids]
		responses = [requests.get(link).status_code for link in links]
		self.client_ids = [{'client_update':'https://raw.githubusercontent.com/'+client['username']+'/GitOpenFL/'+client['branch']+'/current_epoch/update.pth', 'history_link' : 'https://raw.githubusercontent.com/'+client['username']+'/GitOpenFL/'+client['branch']+'/history/history.csv', 'username' : client['username'], 'branch' : client['branch'], 'existence' : (response == 200)} for client, response in zip(self.client_ids, responses)]
		print("number of clients verified : "+str(sum([1 if client['existence'] else 0 for client in self.client_ids])))
		print("\n--- users which could not be verified ---")
		[print('{username: "'+str(client['username'])+'", branch: "'+str(client['branch'])+'"}') for client in self.client_ids if client['existence']==False]
	
	def history_check(self, current_epoch=0):
		pass

if __name__ == '__main__':
	c = ClientConnector()
	print(c.client_ids)
	#c.history_check()