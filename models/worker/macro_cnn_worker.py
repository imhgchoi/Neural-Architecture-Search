
import torch
import torch.nn as nn
import torch.nn.functional as F

from util.neural_blocks import SeparableConv



def get_class_num(dataType):
	if dataType in ['mnist','cifar10'] :
		return 10
	else :
		raise ValueError('invalid dataset type {}'.format(dataType))

def get_in_channel_num(dataType):
	if dataType in ['mnist'] :
		return 1
	elif dataType in ['cifar10'] :
		return 3
	else :
		raise ValueError('invalid dataset type {}'.format(dataType))


class FactorizedReduction(nn.Module):
	'''
	reference
	https://github.com/melodyguan/enas/blob/master/src/cifar10/general_child.py#L129
	https://github.com/TDeVries/enas_pytorch/blob/master/models/shared_cnn.py
	'''
	def __init__(self, indim, outdim, stride=2) :
		super(FactorizedReduction, self).__init__()

		assert outdim % 2 == 0, ("outdim of factorized reduction should be even")

		self.stride = stride

		if stride == 1 :
			self.facRed = nn.Sequential(
				nn.Conv2d(indim, outdim, kernel_size=1, bias=False),
				nn.BatchNorm2d(outdim, track_running_stats=False)
			)
		else :
			self.path1 = nn.Sequential(
				nn.AvgPool2d(1, stride=stride),
				nn.Conv2d(indim, outdim//2, kernel_size=1, bias=False)
			)
			self.path2 = nn.Sequential(
				nn.AvgPool2d(1, stride=stride),
				nn.Conv2d(indim, outdim//2, kernel_size=1, bias=False)
			)
			self.batch_norm = nn.BatchNorm2d(outdim, track_running_stats=False)

	def forward(self, x):
		if self.stride == 1 :
			return self.facRed(x)
		else :
			x1 = self.path1(x)
			# pad the right and the bottom, then crop to include those pixels
			x2 = F.pad(x, pad=(0, 1, 0, 1), mode='constant', value=0.)
			x2 = x2[:, :, 1:, 1:]
			x2 = self.path2(x2)

			x = torch.cat([x1, x2], dim=1)
			x = self.batch_norm(x)
			return x

class FixedMacroLayer(nn.Module):
	def __init__(self, lid, indim, outdim, layer_info):
		super(FixedMacroLayer, self).__init__()
		self.id = lid

		if lid > 0 :
			self.skipIdx = layer_info[1]
		else :
			self.skipIdx = torch.zeros(1)

		if layer_info[0] == 0 :
			self.node = nn.Sequential(
				nn.ReLU(),
				nn.Conv2d(indim, outdim, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(outdim, track_running_stats=False)
			)
		elif layer_info[0] == 1 :
			self.node = nn.Sequential(
				nn.ReLU(),
				SeparableConv(indim, outdim, 3, False),
				nn.BatchNorm2d(outdim, track_running_stats=False),
			)
		elif layer_info[0] == 2 :
			self.node = nn.Sequential(
				nn.ReLU(),
				nn.Conv2d(indim, outdim, kernel_size=5, padding=2, bias=False),
				nn.BatchNorm2d(outdim, track_running_stats=False)
			)
		elif layer_info[0] == 3 :
			self.node = nn.Sequential(
				nn.ReLU(),
				SeparableConv(indim, outdim, 5, False),
				nn.BatchNorm2d(outdim, track_running_stats=False)
			)
		elif layer_info[0] == 4 :
			self.node = nn.Sequential(
				nn.AvgPool2d(kernel_size=3, stride=1, padding=1) 
			)
		elif layer_info[0] == 5 :
			self.node = nn.Sequential(
				nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
			)

		stab_in = int((torch.sum(self.skipIdx).item() + 1)*indim)
		self.stabilizer = nn.Sequential(
			nn.Conv2d(stab_in, outdim, kernel_size=1, bias=False),
			nn.BatchNorm2d(outdim, track_running_stats=False),
			nn.ReLU()
		)

		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

	def forward(self, x, prevLayers, arch):
		x = self.node(x)

		skip_out = []
		for i, skip in enumerate(self.skipIdx) :
			if skip == 1 :
				skip_out.append(prevLayers[i])
		x = torch.cat([x] + skip_out, dim=1).to(self.device)

		x = self.stabilizer(x)
		return x


class MacroLayer(nn.Module):
	def __init__(self, lid, indim, outdim):
		super(MacroLayer, self).__init__()
		self.id = lid

		self.node1 = nn.Sequential(
			nn.ReLU(),
			nn.Conv2d(indim, outdim, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(outdim, track_running_stats=False)
		)
		self.node2 = nn.Sequential(
			nn.ReLU(),
			SeparableConv(indim, outdim, 3, False),
			nn.BatchNorm2d(outdim, track_running_stats=False),
		)
		self.node3 = nn.Sequential(
			nn.ReLU(),
			nn.Conv2d(indim, outdim, kernel_size=5, padding=2, bias=False),
			nn.BatchNorm2d(outdim, track_running_stats=False)
		)
		self.node4 = nn.Sequential(
			nn.ReLU(),
			SeparableConv(indim, outdim, 5, False),
			nn.BatchNorm2d(outdim, track_running_stats=False)
		)
		self.node5 = nn.Sequential(
			nn.AvgPool2d(kernel_size=3, stride=1, padding=1) 
		)
		self.node6 = nn.Sequential(
			nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
		)

		self.stabilizer = nn.Sequential(
			nn.Conv2d(outdim, outdim, kernel_size=1, bias=False),
			nn.BatchNorm2d(outdim, track_running_stats=False),
			nn.ReLU(),
		)
		self.batch_norm = nn.BatchNorm2d(outdim, track_running_stats=False)

	def forward(self, x, prevLayers, arch):
		nodeIdx = arch[0]
		if self.id > 0 :
			skipIdx = arch[1]
		else :
			skipIdx = []

		if nodeIdx == 0 :
			x = self.node1(x)
		elif nodeIdx == 1 :
			x = self.node2(x)
		elif nodeIdx == 2 :
			x = self.node3(x)
		elif nodeIdx == 3 :
			x = self.node4(x)
		elif nodeIdx == 4 :
			x = self.node5(x)
		elif nodeIdx == 5 :
			x = self.node6(x)
		else :
			raise ValueError('invalid node index {}'.format(nodeIdx))

		for i, skip in enumerate(skipIdx):
			if skip == 1 :
				x = x + prevLayers[i]
		x = self.stabilizer(x)

		return self.batch_norm(x)


class MacroCNN(nn.Module):
	def __init__(self, args, fixed_arch=None):
		super(MacroCNN, self).__init__()
		self.args = args
		self.classNum = get_class_num(args.dataset)

		self.first_layer = nn.Sequential(
			nn.Conv2d(get_in_channel_num(args.dataset), args.cnn_first_layer_outdim, 
					  kernel_size=args.cnn_first_layer_kernel,
					  stride=1,
					  padding=args.cnn_first_layer_pad,
					  bias=False),
			nn.BatchNorm2d(args.cnn_first_layer_outdim, track_running_stats=False)
		)

		self.layers = nn.ModuleList([])
		self.pooled_layers = nn.ModuleList([])
		self.pool_indices = list(range(self.args.macro_num_layers-5, 0, -4))

		outdim = args.cnn_first_layer_outdim
		for lid in range(self.args.macro_num_layers) :
			if args.mode.upper() == 'FIX' :
				layer = FixedMacroLayer(lid, outdim, outdim, fixed_arch[str(lid)])
			else :
				layer = MacroLayer(lid, outdim, outdim)
			self.layers.append(layer)

			if lid in self.pool_indices :
				for _ in range(len(self.layers)):
					if args.mode.upper() == 'FIX' :
						self.pooled_layers.append(FactorizedReduction(outdim, outdim * 2))
					else :	
						self.pooled_layers.append(FactorizedReduction(outdim, outdim))
				if args.mode.upper() == 'FIX' :
					outdim = outdim * 2

		self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
		self.final_layer = nn.Linear(outdim, self.classNum)

		self.initialize()

		self.loss = nn.CrossEntropyLoss()
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.to(self.device)


	def initialize(self):
		for mod in self.modules():
			if isinstance(mod, nn.Conv2d):
				nn.init.kaiming_uniform_(mod.weight, nonlinearity='relu')

	def forward(self, x, arch):
		x = self.first_layer(x)

		prevLayers = []
		pool_count = 0
		for lid in range(self.args.macro_num_layers) :
			x = self.layers[lid](x, prevLayers, arch[str(lid)])
			prevLayers.append(x)
			if lid in self.pool_indices :
				for i, prev in enumerate(prevLayers) :
					prevLayers[i] = self.pooled_layers[pool_count](prev)
					pool_count+=1
				x = prevLayers[-1]

		x = self.global_avg_pool(x)
		x = x.view(x.shape[0], -1)
		x = F.dropout(x, p=self.args.dropout)

		return self.final_layer(x)





class MacroCNNlight(nn.Module):
	def __init__(self, args, fixed_arch=None):
		super(MacroCNNlight, self).__init__()
		self.args = args
		self.classNum = get_class_num(args.dataset)

		self.first_layer = nn.Sequential(
			nn.Conv2d(get_in_channel_num(args.dataset), args.cnn_first_layer_outdim, 
					  kernel_size=args.cnn_first_layer_kernel,
					  stride=1,
					  padding=args.cnn_first_layer_pad,
					  bias=False),
			nn.BatchNorm2d(args.cnn_first_layer_outdim, track_running_stats=False)
		)

		self.layers = nn.ModuleList([])

		outdim = args.cnn_first_layer_outdim
		for lid in range(self.args.macro_num_layers) :
			if args.mode == 'fix' :
				layer = FixedMacroLayer(lid, outdim, outdim, fixed_arch[str(lid)])
			else :
				layer = MacroLayer(lid, outdim, outdim)
			self.layers.append(layer)

		self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
		self.final_layer = nn.Linear(outdim, self.classNum)

		self.initialize()

		self.loss = nn.CrossEntropyLoss()
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.to(self.device)


	def initialize(self):
		for mod in self.modules():
			if isinstance(mod, nn.Conv2d):
				nn.init.kaiming_uniform_(mod.weight, nonlinearity='relu')

	def forward(self, x, arch):
		x = self.first_layer(x)

		prevLayers = []
		for lid in range(self.args.macro_num_layers) :
			x = self.layers[lid](x, prevLayers, arch[str(lid)])
			prevLayers.append(x)

		x = self.global_avg_pool(x)
		x = x.view(x.shape[0], -1)
		x = F.dropout(x, p=self.args.dropout)

		return self.final_layer(x)

