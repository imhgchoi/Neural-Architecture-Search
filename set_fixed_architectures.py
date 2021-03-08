import torch
from torch import tensor
import numpy as np
import pdb

from config import get_args

args = get_args()


# RANDOM ARCHITECTURE
RNDM_ARCH = {}
for i in range(args.macro_num_layers) :
	node = tensor([np.argmax(np.random.multinomial(1, [1/6]*6))], device='cuda:0')
	if i == 0 :
		RNDM_ARCH[str(i)] = [node]
	else :
		skip = tensor(np.random.binomial(1,[args.skip_target]*i), device='cuda:0')
		RNDM_ARCH[str(i)] = [node, skip]
torch.save(RNDM_ARCH, './save/states/architecture_rndm.tar')



# ENAS (H. Pham et al)
ENAS_ARCH = {
	'0':  [tensor([3], device='cuda:0')], 
	'1':  [tensor([2], device='cuda:0'), tensor([0], device='cuda:0')], 
	'2':  [tensor([2], device='cuda:0'), tensor([1, 0], device='cuda:0')], 
	'3':  [tensor([3], device='cuda:0'), tensor([1, 0, 0], device='cuda:0')], 
	'4':  [tensor([4], device='cuda:0'), tensor([0, 0, 0, 0], device='cuda:0')], 
	'5':  [tensor([1], device='cuda:0'), tensor([1, 0, 0, 1, 0], device='cuda:0')], 
	'6':  [tensor([2], device='cuda:0'), tensor([1, 0, 1, 1, 0, 1], device='cuda:0')], 
	'7':  [tensor([1], device='cuda:0'), tensor([1, 0, 1, 1, 0, 1, 1], device='cuda:0')], 
	'8':  [tensor([3], device='cuda:0'), tensor([1, 0, 0, 1, 0, 1, 0, 1], device='cuda:0')], 
	'9':  [tensor([4], device='cuda:0'), tensor([0, 0, 0, 0, 0, 0, 1, 0, 0], device='cuda:0')], 
	'10': [tensor([2], device='cuda:0'), tensor([1, 1, 0, 1, 0, 1, 0, 1, 0, 0], device='cuda:0')], 
	'11': [tensor([3], device='cuda:0'), tensor([0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1], device='cuda:0')],
	'12': [tensor([0], device='cuda:0'), tensor([1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1], device='cuda:0')],
	'13': [tensor([3], device='cuda:0'), tensor([0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0], device='cuda:0')]
}
torch.save(ENAS_ARCH, './save/states/architecture_enas.tar')



