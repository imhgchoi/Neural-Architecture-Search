from data.vision import *


__all__ = ['vision', 'text']

VISION = ["mnist","cifar10","imagenet"]
TEXT = []

def set_data(args):
	if args.dataset in VISION :
		return VisionData(args)
	elif args.dataset in TEXT :
		raise NotImplementedError("not yet implemented")
