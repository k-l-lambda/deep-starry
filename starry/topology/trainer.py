
import os
import torch
from tensorboardX import SummaryWriter
import time
from tqdm import tqdm
import logging

from ..transformer.optim import ScheduledOptim
from ..utils.model_factory import loadModel



LOG_DIR = os.environ.get('LOG_DIR', './logs')


class Trainer:
	def __init__ (self, config):
		self.config = config
		self.options = config['trainer']

		self.start_epoch = 0

		self.model = loadModel(config['model'], postfix='Loss')

		self.optimizer = ScheduledOptim(
			torch.optim.Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09),
			d_model=config['model.args.d_model'],
			init_step=self.options.get('steps', 0),
			**config['optim'],
		)

		if self.config['best']:
			self.loadCheckpoint(self.config['best'])
		self.model.to(self.options['device'])

		self.tb_writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, config.id))


	def train (self, training_data, validation_data):
		def print_performances(header, loss, accu, start_time, lr):
			print('  - {header:12} loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, lr: {lr:8.5f}, elapse: {elapse:3.3f} min'
				.format(header=f"({header})", loss=loss, accu=100*accu, elapse=(time.time()-start_time)/60, lr=lr))

		#val_losses = []
		best_acc = 0
		for epoch_i in range(self.start_epoch, self.options['epoch']):
			logging.info(f'[Epoch {epoch_i}]')

			start = time.time()
			train_loss, train_accu = self.train_epoch(training_data)
			#train_ppl = math.exp(min(train_loss, 100))

			# Current learning rate
			lr = self.optimizer._optimizer.param_groups[0]['lr']
			print_performances('Training', train_loss, train_accu, start, lr)

			start = time.time()
			val_loss, val_acc = self.eval_epoch(validation_data)
			print_performances('Validation', val_loss, val_acc, start, lr)

			checkpoint = {
				'epoch': epoch_i,
				'model': self.model.deducer.state_dict(),
				'optim': self.optimizer._optimizer.state_dict(),
			}

			model_name = f'model_{epoch_i:02}_acc_{100*val_acc:3.3f}.chkpt'
			if self.options['save_mode'] == 'all':
				torch.save(checkpoint, self.config.localPath(model_name))
			elif self.options['save_mode'] == 'best':
				#if val_loss <= min(val_losses):
				if val_acc > best_acc or epoch_i == 0:
					torch.save(checkpoint, self.config.localPath(model_name))
					logging.info('	- [Info] The checkpoint file has been updated.')

			if val_acc > best_acc or self.config['best'] is None:
				self.config['best'] = model_name

			#val_losses.append(val_loss)
			best_acc = max(best_acc, val_acc)

			#self.tb_writer.add_scalars('loss', {'train': train_loss, 'val': val_loss}, epoch_i)
			#self.tb_writer.add_scalars('accuracy', {'train': train_accu, 'val': val_acc}, epoch_i)
			self.tb_writer.add_scalar('loss', train_loss, epoch_i)
			self.tb_writer.add_scalar('val_loss', val_loss, epoch_i)
			self.tb_writer.add_scalar('accuracy', train_accu, epoch_i)
			self.tb_writer.add_scalar('val_accuracy', val_acc, epoch_i)
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
		self.model.deducer.load_state_dict(checkpoint['model'])
		self.start_epoch = checkpoint['epoch'] + 1

		if 'optim' in checkpoint:
			self.optimizer._optimizer.load_state_dict(checkpoint['optim'])
