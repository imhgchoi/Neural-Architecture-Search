from models.controller.enas_controller import ENASagent


__all__ = ['ENASagent']

def get_controller(args) :
	if args.controller in ['enas', 'enas_light'] :
		return ENASagent(args)
	else :
		raise NotImplementedError("controller agent {} not implemented".format(args.controller))