
import os
import torch
from tensorboardX import SummaryWriter
import time
from tqdm import tqdm
import logging

from ..transformer.optim import ScheduledOptim
from .models import TransformJointerLoss



LOG_DIR = os.environ.get('LOG_DIR', './logs')


class Trainer:
	def __init__ (self, config):
		self.config = config
		self.options = config['trainer']

		self.start_epoch = 0

		self.model = TransformJointerLoss(**config['model.args'])
		if self.config['best']:
			self.loadCheckpoint(self.config['best'])
		self.model.to(self.options['device'])

		self.optimizer = ScheduledOptim(
			torch.optim.Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09),
			d_model=config['model.args.d_model'],
			init_step=self.options.get('steps', 0),
			**config['optim'],
		)

		self.tb_writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, config.id))


	def train (self, training_data, validation_data):
		def print_performances(header, loss, accu, start_time, lr):
			print('  - {header:12} loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, lr: {lr:8.5f}, elapse: {elapse:3.3f} min'
				.format(header=f"({header})", loss=loss, accu=100*accu, elapse=(time.time()-start_time)/60, lr=lr))

		valid_losses = []
		for epoch_i in range(self.start_epoch, self.options['epoch']):
			logging.info(f'[ Epoch{epoch_i}]')

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

			checkpoint = {'epoch': epoch_i, 'model': self.model.state_dict()}

			model_name = f'model_{epoch_i:02}_acc_{100*valid_accu:3.3f}.chkpt'
			if self.options['save_mode'] == 'all':
				torch.save(checkpoint, self.config.localPath(model_name))
			elif self.options['save_mode'] == 'best':
				if valid_loss <= min(valid_losses):
					torch.save(checkpoint, self.config.localPath(model_name))
					logging.info('	- [Info] The checkpoint file has been updated.')

			if valid_loss <= min(valid_losses):
				self.config['best'] = model_name

			#self.tb_writer.add_scalars('loss', {'train': train_loss, 'val': valid_loss}, epoch_i)
			#self.tb_writer.add_scalars('accuracy', {'train': train_accu, 'val': valid_accu}, epoch_i)
			self.tb_writer.add_scalar('loss', train_loss, epoch_i)
			self.tb_writer.add_scalar('val_loss', train_loss, epoch_i)
			self.tb_writer.add_scalar('accuracy', train_accu, epoch_i)
			self.tb_writer.add_scalar('val_accuracy', valid_accu, epoch_i)
			self.tb_writer.add_scalar('learning_rate', lr, epoch_i)

			self.options['steps'] = self.optimizer.n_steps

			self.config.save()


	def train_epoch (self, dataset):
		self.model.train()
		total_loss, total_acc, n_batch = 0, 0, 0 

		for batch in tqdm(dataset, mininterval=2, desc='  - (Training)   ', leave=False):
			# forward
			self.optimizer.zero_grad()
			loss, acc = self.model(batch)

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
				loss, acc = self.model(batch)

				# note keeping
				n_batch += 1
				total_loss += loss.item()
				total_acc += acc

		return total_loss / n_batch, total_acc / n_batch


	def loadCheckpoint (self, filename):
		checkpoint = torch.load(self.config.localPath(filename), map_location=self.options['device'])
		self.model.load_state_dict(checkpoint['model'])
		self.start_epoch = checkpoint['epoch'] + 1
