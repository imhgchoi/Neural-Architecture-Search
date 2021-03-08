import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import networkx as nx
from networkx.drawing.nx_pydot import to_pydot
import matplotlib.pyplot as plt
from IPython.core.display import Image
import pylab
import numpy as np
import copy
import pdb

from models.worker import *

#####################################################################################
# Code Reference																	#
# https://github.com/TDeVries/enas_pytorch/blob/master/models/controller.py 		#
# https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py 	#
#####################################################################################

class ENASagent(nn.Module) :
	def __init__(self, args):
		super(ENASagent, self).__init__()
		self.args = args

		self.w_lstm = nn.LSTM(input_size=args.lstm_size,
							  hidden_size=args.lstm_size,
							  num_layers=args.lstm_num_layers)
		self.g_emb = nn.Embedding(1, args.lstm_size)
		self.w_emb = nn.Embedding(6, args.lstm_size) # number of nodes = 6
		self.w_soft = nn.Linear(args.lstm_size, 6, bias=False)

		self.w_attn_1 = nn.Linear(args.lstm_size, args.lstm_size, bias=False)
		self.w_attn_2 = nn.Linear(args.lstm_size, args.lstm_size, bias=False)
		self.skip_attn = nn.Linear(args.lstm_size, 1, bias=False)

		self.initialize()

		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.to(self.device)

		self.forward()


	def initialize(self):
		nn.init.uniform_(self.w_lstm.weight_hh_l0, -0.1, 0.1)
		nn.init.uniform_(self.w_lstm.weight_ih_l0, -0.1, 0.1)

		for mod in self.modules():
			if isinstance(mod, nn.Linear) or isinstance(mod, nn.Embedding):
				nn.init.uniform_(mod.weight, -0.1, 0.1)

	def sample_architecture(self, data, worker, sample_pool_size) :
		self.eval()
		worker.eval()

		bestArch, bestAcc = None, 0
		for smpl in range(sample_pool_size):
			self.forward()

			accs = []
			for X, y in data :
				X = X.to(self.device)
				y = y.to(self.device)
				Xpred = worker(X, self.sampledArch)
				acc = torch.mean((torch.argmax(Xpred, dim=1) == y).float())
				accs.append(acc.item())
			accuracy = np.mean(accs)
			if accuracy > bestAcc :
				bestArch = copy.deepcopy(self.sampledArch)
				bestAcc = copy.deepcopy(accuracy)

		return bestArch

	def visualize_architecture(self, arch):
		# text-wise viz
		for key, value in arch.items():
			if len(value) == 1:
				node_type = value[0].cpu().numpy().tolist()
				print('[' + ' '.join(str(n) for n in node_type) + ']')
			else:
				node_type = value[0].cpu().numpy().tolist()
				skips = value[1].cpu().numpy().tolist()
				print('[' + ' '.join(str(n) for n in (node_type + skips)) + ']')

		# graph-wise viz
		G = nx.DiGraph()
		edges = []
		nodeSeq = ['Input']
		for i, (key, value) in enumerate(arch.items()):
			if i == 0 :
				prev_node_type = 'L1_Node_'+str(value[0].item()+1)
				edges.append(('Input', prev_node_type))
				nodeSeq.append(prev_node_type)
				prev_value = []
			else :
				node_type = 'L'+str(i+1)+'_Node_'+str(value[0].item()+1)
				nodeSeq.append(node_type)
				edges.append((prev_node_type, node_type))
				prev_node_type = node_type
				for k, n in enumerate(prev_value) :
					if n.item() == 1 :
						edges.append((nodeSeq[k+1], node_type))
				prev_value = value[1]
		nodeSeq.append('Output')
		for k, n in enumerate(prev_value) :
			if n.item() == 1 :
				edges.append((nodeSeq[k+1], 'Output'))
		edges.append((nodeSeq[-2], 'Output'))

		for node in nodeSeq : 
			G.add_node(node)
		for edge in edges :
			G.add_edge(*edge)

		dot = to_pydot(G)
		dot.set_dpi(300)
		dot.set_rankdir('LR')
		dot.set_size('"5,5!"')
		file_name = './save/dags/{}_{}'.format(self.args.controller, self.args.dataset)
		if self.args.light_mode :
			file_name += '_lgt'
		if self.args.mode == 'fixed' :
			file_name += '_fixed'
		dot.write_png(file_name+'_arch.png')



	def forward(self):
		#####################################################################################
		# Code Reference																	#
		# https://github.com/TDeVries/enas_pytorch/blob/master/models/controller.py 		#
		# https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py 	#
		#####################################################################################
		anchors, anchors_w_1 = [], []
		entropies, logProbs, skipCount, skipPenalties = [], [], [], []
		archSeq = {}

		# initial input is the graph embedding
		inp = self.g_emb.weight
		h0 = None
		skipTarget = torch.Tensor([1-self.args.skip_target,self.args.skip_target]).to(self.device)
		
		for lid in range(self.args.macro_num_layers):
			# compute hidden state for the layer's node type
			inp = inp.unsqueeze(0)
			out, hn = self.w_lstm(inp, h0)
			out = out.squeeze(0)
			h0 = hn

			logit = self.w_soft(out)
			logit = logit / self.args.temperature
			logit = self.args.tanh_constant * torch.tanh(logit)
			nodeIdDist = Categorical(logits=logit)

			# sample node from node selection policy
			nodeID = nodeIdDist.sample()
			archSeq[str(lid)] = [nodeID]

			logProb = nodeIdDist.log_prob(nodeID)
			logProbs.append(logProb.view(-1))
			entropy = nodeIdDist.entropy()
			entropies.append(entropy.view(-1))

			# compute hidden state for the next layer's skip connection
			inp = self.w_emb(nodeID).unsqueeze(0)
			out, hn = self.w_lstm(inp, h0)
			out = out.squeeze(0)

			if lid != 0 :
				# for skip connections
				query = torch.cat(anchors_w_1, dim=0)
				query = torch.tanh(query + self.w_attn_2(out))
				query = self.skip_attn(query)
				logit = torch.cat([-query, query], dim=1)
				logit = logit / self.args.temperature
				logit = self.args.tanh_constant * torch.tanh(logit)
				skipProb = torch.sigmoid(logit)

				kl = skipProb * torch.log(skipProb/skipTarget)
				kl = torch.sum(kl)
				skipPenalties.append(kl)

				# skipDist = Categorical(logits=logit)
				skipDist = Categorical(probs=skipProb)
				skipID = skipDist.sample().view(lid)
				archSeq[str(lid)].append(skipID)

				logProb = torch.sum(skipDist.log_prob(skipID))
				logProbs.append(logProb.view(-1))

				entropy = torch.sum(skipDist.entropy())
				entropies.append(entropy.view(-1))

				# compute next step input with previous skip connections' hidden states
				skipID = skipID.to(torch.float).view(1, lid)
				skipCount.append(torch.sum(skipID))
				inp = torch.matmul(skipID, torch.cat(anchors, dim=0))
				inp = inp / (1.0+torch.sum(skipID))
			else :
				# first layer does not have any skip connections
				# thus, next layer input is substituted with the graph embedding as well
				inp = self.g_emb.weight

			anchors.append(out)
			anchors_w_1.append(self.w_attn_1(out))

		self.sampledEntropy = torch.sum(torch.cat(entropies))
		self.sampledLogProb = torch.sum(torch.cat(logProbs))
		self.skipCount = torch.sum(torch.stack(skipCount))
		self.skipPenalties = torch.mean(torch.stack(skipPenalties))

		self.sampledArch = archSeq

