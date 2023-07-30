import torch
from torch import nn
from torch.autograd import Variable
from lib.model.img2hairstep.layers.inception import inception
# from device_choice import device

class Channels1(nn.Module):
	def __init__(self,channel_scale_factor):
		super(Channels1, self).__init__()
		self.list = nn.ModuleList()
		self.list.append(
			nn.Sequential(
				inception(256,[[64],[3,32,64],[5,32,64],[7,32,64]],channel_scale_factor),
				inception(256,[[64],[3,32,64],[5,32,64],[7,32,64]],channel_scale_factor)
				)
			) #EE
		self.list.append(
			nn.Sequential(
				nn.AvgPool2d(2),
				inception(256,[[64],[3,32,64],[5,32,64],[7,32,64]],channel_scale_factor),
				inception(256,[[64],[3,32,64],[5,32,64],[7,32,64]],channel_scale_factor),
				inception(256,[[64],[3,32,64],[5,32,64],[7,32,64]],channel_scale_factor),
				nn.UpsamplingNearest2d(scale_factor=2)
				)
			) #EEE

	def forward(self,x):
		return self.list[0](x)+self.list[1](x)

class Channels2(nn.Module):
	def __init__(self,channel_scale_factor):
		super(Channels2, self).__init__()
		self.list = nn.ModuleList()
		self.list.append(
			nn.Sequential(
				inception(256, [[64], [3,32,64], [5,32,64], [7,32,64]],channel_scale_factor),
				inception(256, [[64], [3,64,64], [7,64,64], [11,64,64]],channel_scale_factor)
				)
			)#EF
		self.list.append(
			nn.Sequential(
				nn.AvgPool2d(2),
				inception(256, [[64], [3,32,64], [5,32,64], [7,32,64]],channel_scale_factor),
				inception(256, [[64], [3,32,64], [5,32,64], [7,32,64]],channel_scale_factor),
				Channels1(channel_scale_factor),
				inception(256, [[64], [3,32,64], [5,32,64], [7,32,64]],channel_scale_factor),
				inception(256, [[64], [3,64,64], [7,64,64], [11,64,64]],channel_scale_factor),
				nn.UpsamplingNearest2d(scale_factor=2)
				)
			)#EE1EF

	def forward(self,x):
		return self.list[0](x)+self.list[1](x)

class Channels3(nn.Module):
	def __init__(self,channel_scale_factor):
		super(Channels3, self).__init__()
		self.list = nn.ModuleList()
		self.list.append(
			nn.Sequential(
				nn.AvgPool2d(2),
				inception(128, [[32], [3,32,32], [5,32,32], [7,32,32]],channel_scale_factor),
				inception(128, [[64], [3,32,64], [5,32,64], [7,32,64]],channel_scale_factor),
				Channels2(channel_scale_factor),
				inception(256, [[64], [3,32,64], [5,32,64], [7,32,64]],channel_scale_factor),
				inception(256, [[32], [3,32,32], [5,32,32], [7,32,32]],channel_scale_factor),
				nn.UpsamplingNearest2d(scale_factor=2)
				)
			)#BD2EG
		self.list.append(
			nn.Sequential(
				inception(128, [[32], [3,32,32], [5,32,32], [7,32,32]],channel_scale_factor),
				inception(128, [[32], [3,64,32], [7,64,32], [11,64,32]],channel_scale_factor)
				)
			)#BC

	def forward(self,x):
		return self.list[0](x)+self.list[1](x)

class Channels4(nn.Module):
	def __init__(self,channel_scale_factor):
		super(Channels4, self).__init__()
		self.list = nn.ModuleList()
		self.list.append(
			nn.Sequential(
				nn.AvgPool2d(2),
				inception(128, [[32], [3,32,32], [5,32,32], [7,32,32]],channel_scale_factor),
				inception(128, [[32], [3,32,32], [5,32,32], [7,32,32]],channel_scale_factor),
				Channels3(channel_scale_factor),
				inception(128, [[32], [3,64,32], [5,64,32], [7,64,32]],channel_scale_factor),
				inception(128, [[16], [3,32,16], [7,32,16], [11,32,16]],channel_scale_factor),
				nn.UpsamplingNearest2d(scale_factor=2)
				)
			)#BB3BA
		self.list.append(
			nn.Sequential(
				inception(128, [[16], [3,64,16], [7,64,16], [11,64,16]],channel_scale_factor)
				)
			)#A

	def forward(self,x):
		return self.list[0](x)+self.list[1](x)


class Model(nn.Module):
	def __init__(self,channel_scale_factor=1):
		super(Model, self).__init__()

		self.seq = nn.Sequential(
			nn.Conv2d(3, 128//channel_scale_factor, 7, padding=3),
			nn.BatchNorm2d(128//channel_scale_factor),
			nn.ReLU(True),
			Channels4(channel_scale_factor),
			nn.Conv2d(64//channel_scale_factor, 1, 3, padding=1)
		)

		self.sigmoid=nn.Sigmoid()

	def forward(self,x):
		return self.seq(torch.cat([x],dim=1))#+y


def get_model():
	return Model().cuda()

from lib.model.img2hairstep.criterion.relative_depth import relative_depth_crit
def get_criterion(margin):
	return relative_depth_crit(margin)

def f_depth_from_model_output():
	print(">>>>>>>>>>>>>>>>>>>>>>>>>    depth = model_output")
	return ____get_depth_from_model_output

def ____get_depth_from_model_output(model_output):
	return model_output


if __name__ == '__main__':
	from torchsummary import summary
	summary(Model(channel_scale_factor=2),(3,512,512),device='cpu')