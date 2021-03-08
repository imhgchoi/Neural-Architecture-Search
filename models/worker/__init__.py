
from models.worker.macro_cnn_worker import MacroCNN, MacroCNNlight

__all__ = ['MacroCNN', 'MacroCNNlight']


def get_worker(args, architecture=None) :
	if args.task_type == 'vision' :
		if args.worker_type == 'macro' :
			if args.controller == 'enas' :
				worker = MacroCNN(args, architecture)
			elif args.controller == 'enas_light' :
				worker = MacroCNNlight(args, architecture)
		else :
			raise NotImplementedError
	elif args.task_type == 'text' :
		raise NotImplementedError

	return worker