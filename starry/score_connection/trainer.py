
import os
import torch
from tensorboardX import SummaryWriter
import time
from tqdm import tqdm

from ..transformer.optim import ScheduledOptim
from .models import TransformJointer



'''
	options:
		output_dir:			str
		save_mode:			str		'all' or 'best'
		d_model:			int
		epoch:				int
		lr_mul:				float
		n_warmup_steps:		int
'''

class Trainer:
	def __init__ (self, options):
		self.options = options

		self.model = TransformJointer(d_word_vec=options.d_model, d_model=options.d_model)

		self.optimizer = ScheduledOptim(
			torch.optim.Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09),
			options.lr_mul,
			options.d_model,
			options.n_warmup_steps,
		)

		self.tb_writer = SummaryWriter(log_dir=os.path.join(options.output_dir, 'logs'))


	def train (self, training_data, validation_data):
		def print_performances(header, ppl, accu, start_time, lr):
			print('  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, lr: {lr:8.5f}, elapse: {elapse:3.3f} min'
				.format(header=f"({header})", ppl=ppl, accu=100*accu, elapse=(time.time()-start_time)/60, lr=lr))

		valid_losses = []
		for epoch_i in range(self.options.epoch):
			print('[ Epoch', epoch_i, ']')

			start = time.time()
			train_loss, train_accu = self.train_epoch(training_data)
			#train_ppl = math.exp(min(train_loss, 100))

			# Current learning rate
			lr = self.optimizer._optimizer.param_groups[0]['lr']
			print_performances('Training', train_loss, train_accu, start, lr)

			start = time.time()
			valid_loss, valid_accu = self.eval_epoch(validation_data)
			#valid_ppl = math.exp(min(valid_loss, 100))
			print_performances('Validation', valid_loss, valid_accu, start, lr)

			valid_losses += [valid_loss]

			checkpoint = {'epoch': epoch_i, 'settings': self.options, 'model': self.model.state_dict()}

			if self.options.save_mode == 'all':
				model_name = 'model_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
				torch.save(checkpoint, model_name)
			elif self.options.save_mode == 'best':
				model_name = f'model_{epoch_i:02}.chkpt'
				if valid_loss <= min(valid_losses):
					torch.save(checkpoint, os.path.join(self.options.output_dir, model_name))
					print('	- [Info] The checkpoint file has been updated.')

			self.tb_writer.add_scalars('ppl', {'train': train_loss, 'val': valid_loss}, epoch_i)
			self.tb_writer.add_scalars('accuracy', {'train': train_accu, 'val': valid_accu}, epoch_i)
			self.tb_writer.add_scalar('learning_rate', lr, epoch_i)


	def train_epoch (self, dataset):
		self.model.train()
		total_loss, total_acc, n_batch = 0, 0, 0 

		for batch in tqdm(dataset, mininterval=2, desc='  - (Training)   ', leave=False):
			# forward
			self.optimizer.zero_grad()
			loss = self.model.forwardLoss(batch)
			acc = 1 - loss # TODO

			# backward and update parameters
			loss.backward()
			self.optimizer.step_and_update_lr()

			# note keeping
			n_batch += 1
			total_loss += loss.item()
			total_acc += acc

		return total_loss / n_batch, total_acc / n_batch


	def eval_epoch (self, dataset):
		self.model.eval()
		total_loss, total_acc, n_batch = 0, 0, 0 

		with torch.no_grad():
			for batch in tqdm(dataset, mininterval=2, desc='  - (Validation) ', leave=False):
				# forward
				loss = self.model.forwardLoss(batch)
				acc = 1 - loss # TODO

				# note keeping
				n_batch += 1
				total_loss += loss.item()
				total_acc += acc

		return total_loss / n_batch, total_acc / n_batch
