import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import pdb

from util.settings import split_data

class VisionData :
	def __init__(self, args):
		self.args = args

		if args.dataset == 'mnist' :
			Data = datasets.MNIST
			normalize = transforms.Normalize((0.5,), (1.0,))
			train_transform = transforms.Compose([
				transforms.ToTensor(),
				normalize
			])
			test_transform = transforms.Compose([
				transforms.ToTensor(),
				normalize
			])

		elif args.dataset == 'cifar10' :
			Data = datasets.CIFAR10

			mean = [0.49139968, 0.48215827, 0.44653124]
			std = [0.24703233, 0.24348505, 0.26158768]
			normalize = transforms.Normalize(mean, std)

			train_transform = transforms.Compose([
 				transforms.RandomCrop(32, padding=4),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				normalize
			])
			test_transform = transforms.Compose([
				transforms.ToTensor(),
				normalize
			])

		else :
			raise ValueError("dataset {} not available".format(args.dataset))

		train_valid = Data(root='./save/data', train=True, transform=train_transform, download=True)
		test = Data(root='./save/data', train=False, transform=test_transform, download=True)
		train, valid = split_data(train_valid, debug=args.debug, light=args.light_mode)
		batch_size = 10 if args.debug else args.batch_size

		self.train = torch.utils.data.DataLoader(train, 
												 batch_size=batch_size, 
												 shuffle=True, 
												 pin_memory=True,
												 num_workers=4)
		self.valid = torch.utils.data.DataLoader(valid, 
												 batch_size=batch_size, 
												 shuffle=True, 
												 pin_memory=True,
												 num_workers=4)
		self.test = torch.utils.data.DataLoader(test,
            									batch_size=batch_size, 
            									shuffle=True, 
            									pin_memory=True,
												 num_workers=4)
