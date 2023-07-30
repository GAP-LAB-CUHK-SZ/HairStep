import torch
from torch import nn
from torch.autograd import Variable

class inception(nn.Module):
	def __init__(self, input_size, config, channel_scale_factor=1):
		self.config = config
		super(inception,self).__init__()
		self.convs = nn.ModuleList()

		# Base 1*1 conv layer
		self.convs.append(nn.Sequential(
			nn.Conv2d(input_size//channel_scale_factor, config[0][0]//channel_scale_factor,1),
			nn.BatchNorm2d(config[0][0]//channel_scale_factor,affine=False),
			nn.ReLU(True),
		))

		# Additional layers
		for i in range(1, len(config)):
			filt = config[i][0]
			pad = int((filt-1)/2)
			out_a = config[i][1]//channel_scale_factor
			out_b = config[i][2]//channel_scale_factor
			conv = nn.Sequential(
				nn.Conv2d(input_size//channel_scale_factor, out_a,1),
				nn.BatchNorm2d(out_a,affine=False),
				nn.ReLU(True),
				nn.Conv2d(out_a, out_b, filt,padding=pad),
				nn.BatchNorm2d(out_b,affine=False),
				nn.ReLU(True)
				)
			self.convs.append(conv)

	def __repr__(self):
		return "inception"+str(self.config)

	def forward(self, x):
		ret = []
		for conv in (self.convs):
			ret.append(conv(x))
		return torch.cat(ret,dim=1)

if __name__ == '__main__':
	# testing
	testModule = inception(256, [[64], [3,32,64], [5,32,64], [7,32,64]])
	print(testModule)
	x = Variable(torch.rand(2,256,125,125))
	# conv = nn.Conv2d(256, 32,1)
	# norm = nn.BatchNorm2d(32,affine=False)
	# relu = nn.ReLU(True)
	# conv2 = nn.Conv2d(32,64,3,padding=1)
	# norm2 = nn.BatchNorm2d(64,affine=False)
	# print(relu(norm2(conv2(relu(norm(conv(x)))))))
	print(testModule(x))