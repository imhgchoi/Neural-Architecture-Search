import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv(nn.Module):
	def __init__(self, indim, outdim, kernel, bias):
		super(SeparableConv, self).__init__()
		self.depthwise = nn.Conv2d(indim, outdim, 
								   kernel_size=kernel,
								   padding=(kernel-1)//2,
								   groups=indim,
								   bias=bias)
		self.pointwise = nn.Conv2d(indim, outdim, 
								   kernel_size=1,
								   bias=bias)

	def forward(self, x):
		x = self.depthwise(x)
		x = self.pointwise(x)
		return x

