import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import numpy as np
import matplotlib.pyplot as plt
import pdb

from models.worker.macro_cnn_worker import MacroCNN


class Trainer :
	def __init__(self, args, data, controller, worker):
		self.args = args
		self.trainData = data.train
		self.validData = data.valid
		self.testData = data.test
		self.controller = controller
		self.controller_baseline = None
		self.worker = worker
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

		self.startEpoch = 1
		self.bestAcc = 0
		self.workerAccStream = []   # in fact, train set accuracies
		self.workerLossStream = []
		self.controllerLossStream = []   # in fact, valid set accuracies
		self.controllerAccStream = []
		self.controllerRewardStream = []
		self.controllerAdvStream = []
		self.testAccStream = []
		self.lrStream = []

		self.controller_optimizer = torch.optim.Adam(params=self.controller.parameters(),
													 lr=args.controller_lr,
													 betas=(0.0, 0.999),
													 eps=0.001)
		self.worker_optimizer = torch.optim.SGD(params=self.worker.parameters(),
												lr=args.worker_max_lr,
												nesterov=True,
												momentum=0.9,
												weight_decay=args.worker_l2_weight_decay)

		if args.worker_lr_scheduler == 'cosine' :
			self.scheduler = CosineAnnealingLR(optimizer=self.worker_optimizer,
											   T_max=args.worker_lr_T,
											   eta_min=args.worker_min_lr)
		elif args.worker_lr_scheduler == 'lambda' :
			self.scheduler = LambdaLR(optimizer=self.worker_optimizer,
									  lr_lambda=lambda epoch : 0.95 ** epoch)

		# params for fixed training
		self.fixedTestAcc = 0


	def train_fixed(self):

		print('start training fixed architecture..')
		self.controller.visualize_architecture(self.controller.sampledArch)

		for epoch in range(self.startEpoch-1, self.args.fixed_epochs):

			msg = 'EPOCH {} ::'.format(epoch+1)

			self.train_worker(msg)

			validAcc = self.eval_worker(msg)

			self.lrStream.append(self.worker_optimizer.param_groups[0]['lr'])
			self.scheduler.step(epoch)
			
			testAcc, _ = self.test(msg=msg)

			self.plot(epoch)

			if validAcc > self.bestAcc and not self.args.debug:
				self.bestAcc = validAcc
				self.fixedTestAcc = testAcc
				best_info = {
					'epoch' : epoch+1,
					'args' : self.args,
					'worker' : self.worker.state_dict(),
					'worker_optimizer' : self.worker_optimizer.state_dict(),
					'best_acc' : self.bestAcc,
					'test_acc' : self.fixedTestAcc,
					'w_acc' : self.workerAccStream,  
					'w_loss' : self.workerLossStream,
					'c_acc' : self.controllerAccStream,
					'c_loss' : self.controllerLossStream,
					't_acc' : self.testAccStream,
					'w_lr' : self.lrStream
				}
				file_name = './save/states/{}_{}'.format(
					self.args.controller, 
					self.args.dataset
				)
				if self.args.light_mode :
					file_name += '_lgt'
				torch.save(best_info, file_name+'_best_state.fixed.tar')
				print('\nSaved Best State Info for Fixed Mode!\n')



	def train(self):

		for epoch in range(self.startEpoch-1, self.args.epochs):

			msg = 'EPOCH {} ::'.format(epoch+1)

			self.train_worker(msg)

			self.train_controller(msg)

			self.lrStream.append(self.worker_optimizer.param_groups[0]['lr'])
			self.scheduler.step(epoch)

			acc, arch = self.test(msg)

			self.plot(epoch)

			if acc > self.bestAcc and not self.args.debug:
				self.bestAcc = acc
				best_info = {
					'epoch' : epoch+1,
					'args' : self.args,
					'worker' : self.worker.state_dict(),
					'controller' : self.controller.state_dict(),
					'worker_optimizer' : self.worker_optimizer.state_dict(),
					'controller_optimizer' : self.controller_optimizer.state_dict(),
					'best_acc' : self.bestAcc,
					'w_acc' : self.workerAccStream,  
					'w_loss' : self.workerLossStream,
					'c_acc' : self.controllerAccStream,
					'c_loss' : self.controllerLossStream,
					'c_rwd': self.controllerRewardStream,
					'c_adv' : self.controllerAdvStream,
					't_acc' : self.testAccStream,
					'w_lr' : self.lrStream
				}
				self.controller.visualize_architecture(arch)
				file_name = './save/states/{}_{}'.format(
					self.args.controller, 
					self.args.dataset
				)
				if self.args.light_mode :
					file_name += '_lgt'
				torch.save(best_info, file_name+'_best_state.tar')
				torch.save(arch, './save/states/architecture.tar')
				print('\nSaved Best State Info!\n')


	def train_worker(self, msg=''):
		"""
		train worker CNN model
		"""
		self.controller.eval()
		self.worker.train()

		trainAccuracies, losses = [], []

		for i, (X, y) in enumerate(self.trainData):
			worker_msg = ''
			X = X.to(self.device)
			y = y.to(self.device)

			# generate architecture
			if self.args.mode.upper() == 'TRAIN' :
				self.controller()

			# forward pass on worker
			self.worker.zero_grad()
			Xpred = self.worker(X, self.controller.sampledArch)
			loss = self.worker.loss(Xpred, y)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.worker.parameters(), self.args.grad_clip)
			self.worker_optimizer.step()
			losses.append(loss.item())

			trainAcc = torch.mean((torch.argmax(Xpred, dim=1) == y).float())
			trainAccuracies.append(trainAcc.item())

			# display info
			worker_msg += msg
			worker_msg += ' Training Worker Batch {}'.format(i+1)
			worker_msg += ' | loss={0:.5f}  lr={1:.4f}  TrainAcc={2:.4f}'.format(
				np.mean(losses), 
				self.worker_optimizer.param_groups[0]['lr'], 
				np.mean(trainAccuracies)
			)

			if i % self.args.print_step == 0 :
				print(worker_msg)

			if self.args.debug :
				if i > 20 : 
					break

		self.workerAccStream.append(np.mean(trainAccuracies))
		self.workerLossStream.append(np.mean(losses))

	def eval_worker(self, msg=''):
		"""
		evaluate worker CNN model
		"""
		self.worker.eval()
		self.controller.eval()

		validAccuracies, losses = [], []

		for i, (X, y) in enumerate(self.validData):
			worker_msg = ''
			X = X.to(self.device)
			y = y.to(self.device)

			# forward pass on worker
			Xpred = self.worker(X, self.controller.sampledArch)
			loss = self.worker.loss(Xpred, y)
			losses.append(loss.item())

			validAcc = torch.mean((torch.argmax(Xpred, dim=1) == y).float())
			validAccuracies.append(validAcc.item())

			# display info
			worker_msg += msg
			worker_msg += ' Evaluating Worker Batch {}'.format(i+1)
			worker_msg += ' | loss={0:.5f}  lr={1:.4f}  ValidAcc={2:.4f}'.format(
				np.mean(losses), 
				self.worker_optimizer.param_groups[0]['lr'], 
				np.mean(validAccuracies)
			)

			if i % self.args.print_step == 0 :
				print(worker_msg)

			if self.args.debug :
				if i > 20 : 
					break

		self.controllerLossStream.append(np.mean(losses))
		self.controllerAccStream.append(np.mean(validAccuracies))

		return np.mean(validAccuracies)


	def train_controller(self, msg=''):
		"""
		train controller model
		"""
		self.controller.train()
		self.worker.eval()

		validAccuracies, losses, rewards, advantages = [], [], [], []
		kls, entropies = [], []
		BatchCount, stepCount, flag = 0, 0, False

		self.controller.zero_grad()
		while True :
			for i, (X, y) in enumerate(self.validData) :
				X = X.to(self.device)
				y = y.to(self.device)

				# used for PPO
				old_pi = self.controller.sampledLogProb.detach()

				# generate architecture
				self.controller()

				# compute reward
				Xpred = self.worker(X, self.controller.sampledArch)
				validAcc = torch.mean((torch.argmax(Xpred, dim=1) == y).float())


				if self.args.algo.upper() == 'PPO' :	
					reward = validAcc.detach()

					if self.controller_baseline is None :
						self.controller_baseline = validAcc
					else :
						self.controller_baseline += -(1-self.args.bl_decay) * (self.controller_baseline-reward)
					self.controller_baseline = self.controller_baseline.detach()
					advantage = reward - self.controller_baseline

					# compute policy loss
					ratio = torch.exp(self.controller.sampledLogProb - old_pi)
					surrogate1 = ratio * advantage
					surrogate2 = torch.clamp(ratio, 1-self.args.ppo_eps, 1+self.args.ppo_eps) * advantage
					ent = self.controller.sampledEntropy 
					kl = self.controller.skipPenalties
					loss = -torch.min(surrogate1, surrogate2) - self.args.entropy_weight*ent + self.args.skip_weight*kl
					loss = loss / self.args.controller_batch_size * self.args.ppo_K

					# Aggregate Loss for multiple steps, then update params
					stepCount += 1
					if stepCount%self.args.controller_batch_size == self.args.controller_batch_size-1 :
						BatchCount += 1   # Note that multiple architectures' aggregate loss is considered one batch
						torch.nn.utils.clip_grad_norm_(self.controller.parameters(), self.args.grad_clip)
						self.controller_optimizer.step()
						self.controller.zero_grad()

						controller_msg = msg
						controller_msg += ' Training Controller Batch {}'.format(BatchCount)
						controller_msg += ' | loss={0:.4f} lr={1:.4f} Entropy={2:.2f} KL={3:.3f} reward={4:.3f} Adv={5:.3f} validAcc={6:.4f}'.format(
							np.sum(losses[-self.args.controller_batch_size:]), 
							self.controller_optimizer.param_groups[0]['lr'], 
							np.mean(entropies[-self.args.controller_batch_size:]),
							np.mean(kls[-self.args.controller_batch_size:]),
							np.mean(rewards[-self.args.controller_batch_size:]),
							np.mean(advantages[-self.args.controller_batch_size:]),
							np.mean(validAccuracies[-self.args.controller_batch_size:])
						)
						print(controller_msg)

				elif self.args.algo.upper() == 'PG' :
					ent = self.controller.sampledEntropy	
					reward = validAcc + self.args.entropy_weight * ent 
					reward = reward.detach()

					if self.controller_baseline is None :
						self.controller_baseline = validAcc
					else :
						self.controller_baseline += -(1-self.args.bl_decay) * (self.controller_baseline-reward)
					self.controller_baseline = self.controller_baseline.detach()

					# compute policy loss
					kl = self.controller.skipPenalties
					advantage = reward - self.controller_baseline
					loss = -self.controller.sampledLogProb * advantage + self.args.skip_weight * kl
					loss = loss / self.args.controller_batch_size
					loss.backward(retain_graph=True)

					# Aggregate Loss for multiple steps, then update params
					stepCount += 1
					if stepCount%self.args.controller_batch_size == self.args.controller_batch_size-1 :
						BatchCount += 1   # Note that multiple architectures' aggregate loss is considered one batch
						torch.nn.utils.clip_grad_norm_(self.controller.parameters(), self.args.grad_clip)
						self.controller_optimizer.step()
						self.controller.zero_grad()

						controller_msg = msg
						controller_msg += ' Training Controller Batch {}'.format(BatchCount)
						controller_msg += ' | loss={0:.4f} lr={1:.4f} Entropy={2:.2f} KL={3:.3f} reward={4:.3f} Adv={5:.3f} validAcc={6:.4f}'.format(
							np.sum(losses[-self.args.controller_batch_size:]), 
							self.controller_optimizer.param_groups[0]['lr'], 
							np.mean(entropies[-self.args.controller_batch_size:]),
							np.mean(kls[-self.args.controller_batch_size:]),
							np.mean(rewards[-self.args.controller_batch_size:]),
							np.mean(advantages[-self.args.controller_batch_size:]),
							np.mean(validAccuracies[-self.args.controller_batch_size:])
						)

				validAccuracies.append(validAcc.item())
				rewards.append(reward.item())
				kls.append(kl.item())
				entropies.append(ent.item())
				advantages.append(advantage.item())
				losses.append(loss.item())
				if stepCount >= self.args.controller_max_steps :
					flag = True
					break
			if self.args.debug :
				if stepCount > 10 :
					break
			if flag :
				break

		self.controllerAccStream.append(np.mean(validAccuracies))
		self.controllerLossStream.append(np.mean(losses))
		self.controllerRewardStream.append(np.mean(rewards))
		self.controllerAdvStream.append(np.mean(advantages))


	def test(self, msg=''):
		self.controller.eval()
		self.worker.eval()

		if self.args.mode.upper() == 'TRAIN' :
			print('\n'+msg+' Sampling Architectures...')
			arch = self.controller.sample_architecture(data=self.validData, 
													   worker=self.worker, 
													   sample_pool_size=self.args.sample_pool_size)
		elif self.args.mode.upper() == 'FIX' :
			arch = self.controller.sampledArch
		elif self.args.mode.upper() == 'TEST' : 
			arch = torch.load('./save/states/architecture.tar')
			self.controller.visualize_architecture(arch)

		testAccuracies = []
		for i, (X, y) in enumerate(self.testData) :
			X = X.to(self.device)
			y = y.to(self.device)

			Xpred = self.worker(X, arch)
			testAcc = torch.mean((torch.argmax(Xpred, dim=1) == y).float())
			testAccuracies.append(testAcc.item())
		testAccuracy = np.mean(testAccuracies)
		self.testAccStream.append(testAccuracy)

		msg += ' Testing Controller'
		msg += ' | test Accuracy = {0:.4f}'.format(testAccuracy)
		if self.args.mode.upper() == 'FIX' :
			msg += ' ( BEST : {0:.4f} )'.format(self.fixedTestAcc)
		msg += '\n\n'
		print(msg)

		return testAccuracy, arch


	def plot(self, epoch):
		def _plot(data, datatype, legend, filename) :
			plt.figure(figsize=(12, 5))
			plt.plot(data)
			plt.title(filename)
			plt.xlabel('Epochs')
			plt.ylabel(datatype)
			plt.legend([legend])
			plt.xticks(list(range(0,epoch+1,20)), list(range(1,epoch+2,20)))
			plt.savefig('./save/plots/{}.png'.format(filename))
			plt.close()

		if self.args.mode == 'train' :
			_plot(self.workerAccStream, 'Accuracy', 'train', 'worker_accuracy')
			_plot(self.workerLossStream, 'Loss', 'train', 'worker_loss')
			_plot(self.controllerLossStream, 'Loss', 'valid', 'controller_loss')
			_plot(self.controllerAccStream, 'Accuracy', 'valid', 'controller_accuracy')
			_plot(self.controllerRewardStream, 'Reward', 'valid', 'rewards')
			_plot(self.controllerAdvStream, 'Advantage', 'valid', 'advantages')
			_plot(self.testAccStream, 'Accuracy', 'test', 'test_accuracy')
			_plot(self.lrStream, 'Learning Rate', 'train', 'learning_rate')
		elif self.args.mode == 'fix' :
			_plot(self.workerAccStream, 'Accuracy', 'train', 'fix_train_accuracy')
			_plot(self.workerLossStream, 'Loss', 'train', 'fix_train_loss')
			_plot(self.controllerAccStream, 'Accuracy', 'valid', 'fix_valid_accuracy')
			_plot(self.controllerLossStream, 'Loss', 'valid', 'fix_valid_loss')
			_plot(self.testAccStream, 'Accuracy', 'test', 'fix_test_accuracy')
			_plot(self.lrStream, 'Learning Rate', 'train', 'fix_learning_rate')
