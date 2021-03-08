import torch
import numpy as np

import util.settings as settings
import config

from data import set_data
from models.controller import get_controller
from models.worker import get_worker
from trainer.trainer import Trainer



def load(trainer, args) :
	if args.mode.upper()in ['TRAIN', 'TEST'] :
		if args.restart :
			return trainer

		load_file_name = './save/states/{}_{}'.format(
			args.controller, 
			args.dataset
		)
		if args.light_mode :
			load_file_name += '_lgt'
		load_file_name += '_best_state.tar'
		state_info = torch.load(load_file_name)

		trainer.startEpoch = state_info['epoch']+1
		trainer.worker.load_state_dict(state_info['worker'])
		trainer.controller.load_state_dict(state_info['controller'])
		trainer.worker_optimizer.load_state_dict(state_info['worker_optimizer'])
		trainer.controller_optimizer.load_state_dict(state_info['controller_optimizer'])
		trainer.workerAccStream = state_info['w_acc']
		trainer.workerLossStream = state_info['w_loss']
		trainer.controllerAccStream = state_info['c_acc']
		trainer.controllerLossStream = state_info['c_loss']
		trainer.controllerRewardStream = state_info['c_rwd']
		trainer.controllerAdvStream = state_info['c_adv']
		trainer.testAccStream = state_info['t_acc']
		trainer.lrStream = state_info['w_lr']
		trainer.bestAcc = state_info['best_acc']

	elif args.mode.upper() == 'FIX' :
		if args.restart_fix :
			return trainer

		load_file_name = './save/states/{}_{}'.format(
			args.controller, 
			args.dataset
		)
		if args.light_mode :
			load_file_name += '_lgt'
		load_file_name += '_best_state.fixed.tar'
		state_info = torch.load(load_file_name)

		trainer.startEpoch = state_info['epoch']+1
		trainer.worker.load_state_dict(state_info['worker'])
		trainer.worker_optimizer.load_state_dict(state_info['worker_optimizer'])
		trainer.workerAccStream = state_info['w_acc']
		trainer.workerLossStream = state_info['w_loss']
		trainer.controllerAccStream = state_info['c_acc']
		trainer.controllerLossStream = state_info['c_loss']
		trainer.testAccStream = state_info['t_acc']
		trainer.lrStream = state_info['w_lr']
		trainer.bestAcc = state_info['best_acc']
		trainer.fixedTestAcc = state_info['test_acc']

	return trainer


def main(args):

	settings.set_seeds(args)
	
	data = set_data(args)

	controller = get_controller(args)

	if args.mode.upper() == 'FIX' :
		fixedArch = torch.load(args.fixed_arch_dir)
		worker = get_worker(args, architecture=fixedArch)
		controller.sampledArch = fixedArch
	else :	
		worker = get_worker(args)

	trainer = Trainer(args, data, controller, worker)
	trainer = load(trainer, args)

	if args.mode.upper() == 'TRAIN' :
		trainer.train()
	elif args.mode.upper() == 'FIX' :
		trainer.train_fixed()
	elif args.mode.upper() == 'TEST' :
		trainer.test()




if __name__ == "__main__" :	

	settings.set_dirs()
	args = config.get_args()
	args = settings.set_debug(args) if args.debug else args
	print(args,'\n')

	main(args)
