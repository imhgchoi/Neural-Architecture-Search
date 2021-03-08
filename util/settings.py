import os
import torch
import numpy as np
import random



def set_seeds(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)


def set_dirs():
	dirs = ['out/', 'save/states/', 'save/data/', 'save/dags/', 'save/plots/']
	for path in dirs :
		if not os.path.exists(path) :
			os.makedirs(path)

def set_debug(args):
	print('\nSTARTING DEBUG MODE...\n')
	args.batch_size = 10
	args.epochs = 5
	args.controller_max_epochs = 5
	args.worker_max_epochs = 5
	args.sample_pool_size = 2
	args.print_step = 1
	args.test_step = 1
	args.controller_batch_size = 1
	return args

def split_data(full, debug=False, light=False, ratio=0.9):
	if debug :
		full = torch.utils.data.Subset(full, range(400))
	elif light :
		full = torch.utils.data.Subset(full, range(len(full)//5))
	threshold = int(len(full) * ratio)
	set1 = torch.utils.data.Subset(full, range(0, threshold))
	set2 = torch.utils.data.Subset(full, range(threshold, len(full)))
	return set1, set2