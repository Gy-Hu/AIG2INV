import torch
from torch import nn

class MessageNorm(nn.Module):
	""" Customized Batch Normalization for Message Passing """
	def __init__(self, vhs, epsilon=1e-4):
		super().__init__()
		self.vhs = vhs
		self.epsilon = epsilon

		gamma = torch.Tensor(1)
		beta = torch.Tensor(1)
		self.gamma = nn.Parameter(gamma)
		self.beta = nn.Parameter(beta)

		# for stat
		self.mean = None
		self.varianceN = None

		# self.mean_acc = None
		# self.var_nsq_acc = None
		# self.num_sample = 0

		# initialize gamma & beta
		nn.init.ones_(self.gamma)
		nn.init.zeros_(self.beta)

	# def clear_mean_var(self):
	# 	self.mean = None
	# 	self.varianceN = None

	def set_initial_param(self, num_sample, device):
		self.num_sample = torch.tensor([num_sample]).to(device)
		self.varianceN = torch.tensor([0.0]).to(device)
		self.mean = torch.tensor([0.0]).to(device)


	def get_initial_param_from(self, set_of_x):
		assert self.mean is None
		self.num_sample = set_of_x.shape[0]
		assert self.vhs == set_of_x.shape[1]
		with torch.no_grad():
			self.varianceN = torch.var(set_of_x, dim=0, unbiased=False) * self.num_sample
			self.mean = torch.mean(set_of_x, dim=0)

	# def _update_param(self, new_x):
	# 	"""Online algorithm from : https://datagenetics.com/blog/november22017/index.html"""

	# 	n_sample = new_x.shape[0]
	# 	assert new_x.shape[1] == self.vhs

	# 	with torch.no_grad():
	# 		if self.mean_acc is None:
	# 			assert self.num_sample == 0
	# 			self.mean_acc = torch.mean(new_x, dim=0)
	# 			self.var_nsq_acc = torch.var(new_x, dim=0, unbiased=False) * n_sample
	# 			self.num_sample += n_sample
	# 		else:
	# 			for idx in range(n_sample):
	# 				newValue = new_x[idx]
	# 				self.num_sample += 1
	# 				delta = newValue - self.mean_acc
	# 				self.mean_acc += delta / self.num_sample
	# 				delta2 = newValue - self.mean_acc
	# 				self.var_nsq_acc += delta*delta2


	def update_param_sliding_window(self, new_x, pop_x):
		"""algorithm from : https://nestedsoftware.com/2019/09/26/incremental-average-and-standard-deviation-with-sliding-window-470k.176143.html"""

		n_sample = new_x.shape[0]
		assert n_sample == pop_x.shape[0]
		assert new_x.shape[1] == self.vhs
		assert pop_x.shape[1] == self.vhs
		#assert self.mean_acc is not None
		assert self.num_sample > 0

		with torch.no_grad():
			for idx in range(n_sample):
				newValue = new_x[idx]
				popValue = pop_x[idx]

				newMean = self.mean + (newValue - popValue)/self.num_sample
				newDSquare = self.varianceN + ( newValue - popValue ) * (newValue - newMean + popValue - self.mean)
				self.mean = newMean
				self.varianceN = newDSquare
		if not (torch.min(self.varianceN)>-self.epsilon).item():
			print ('min variance:', torch.min(self.varianceN))
			assert False

	# def use_update_param(self):
	# 	self.mean, self.variance = self.mean_acc, self.var_nsq_acc/self.num_sample

	# def reset_acc(self):
	# 	self.mean_acc, self.var_nsq_acc = None, None
	# 	self.num_sample = 0

	def forward(self, x):
		#print (self.varianceN)
		#print (self.mean)
		#exit(1)
		return (x-self.mean)/torch.sqrt( self.varianceN / self.num_sample + self.epsilon ) * self.gamma + self.beta



# def test():
# 	rand_inp = torch.randn((7,10))
# 	all_mean = torch.mean(rand_inp, dim=0)
# 	all_var = torch.var(rand_inp, dim=0, unbiased=False)

# 	msgnorm = MessageNorm(vhs=10)
# 	msgnorm._update_param(rand_inp[:4])
# 	msgnorm._update_param(rand_inp[4:])
# 	msgnorm.use_update_param()

# 	print (torch.sum(torch.abs(all_mean-msgnorm.mean)))
# 	print (torch.sum(torch.abs(all_var-msgnorm.variance)))




def test():
	rand_inp = torch.randn((7,10))
	all_mean = torch.mean(rand_inp, dim=0)
	all_var = torch.var(rand_inp, dim=0, unbiased=False)

	msgnorm = MessageNorm(vhs=10)
	msgnorm.get_initial_param_from(rand_inp[:4])
	for idx in range(4,7):
		msgnorm.update_param_sliding_window(rand_inp[idx:idx+1], rand_inp[idx-4:idx-3])

		vdata = rand_inp[idx-3:idx+1]
		expected_mean = torch.mean(vdata, dim=0)
		expected_var = torch.var(vdata, dim=0, unbiased=False) * 4

		print (torch.sum(torch.abs(expected_mean-msgnorm.mean)))
		print (torch.sum(torch.abs(expected_var-msgnorm.varianceN)))


if __name__ == '__main__':
	test()
