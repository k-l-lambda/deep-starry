
import os
import torch
from tensorboardX import SummaryWriter
import time
from tqdm import tqdm
import logging

from .optim import optim
from .model_factory import loadModel



LOG_DIR = os.environ.get('LOG_DIR', './logs')


def print_acc (acc):
	if type(acc) == float or type(acc) == torch.Tensor:
		return f'accuracy: {acc*100:3.3f}%'
	elif type(acc) == dict:
		items = map(lambda item: f'{item[0]}: {item[1]:3.3f}',
			filter(lambda item: type(item) in [int, float],
				acc.items()))
		return ', '.join(items)
	else:
		return str(acc)


def stat_average (data, n_batch):
	return dict([(k, v / n_batch) for k, v in data.items()])


class Trainer:
	def __init__ (self, config):
		self.config = config
		self.options = config['trainer']

		self.start_epoch = 0

		self.model = loadModel(config['model'], postfix='Loss')
		self.model.to(self.options['device'])

		self.optimizer = optim(config['optim'], self.model.parameters(), init_step=self.options.get('steps', 0))

		weights = self.config['best'] or self.config['trainer.pretrained_weights']
		if weights:
			self.loadCheckpoint(weights)

		self.tb_writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, config.id))


	def train (self, training_data, validation_data):
		def print_performances(header, loss, accu, start_time, lr):
			print('  - {header:12} loss: {loss: .4e}, {accu}, lr: {lr:.4e}, elapse: {elapse:3.2f} min'
				.format(header=f"({header})", loss=loss, accu=print_acc(accu), elapse=(time.time()-start_time)/60, lr=lr))

		#val_losses = []
		best_acc = 0
		for epoch_i in range(self.start_epoch, self.options['epoch']):
			logging.info(f'[Epoch {epoch_i}]')

			start = time.time()
			train_loss, train_acc = self.train_epoch(training_data)
			#train_ppl = math.exp(min(train_loss, 100))

			# Current learning rate
			lr = self.optimizer._optimizer.param_groups[0]['lr']
			print_performances('Training', train_loss, train_acc, start, lr)

			start = time.time()
			val_loss, val_acc = self.eval_epoch(validation_data)
			print_performances('Validation', val_loss, val_acc, start, lr)

			checkpoint = {
				'epoch': epoch_i,
				'model': self.model.deducer.state_dict(),
				'optim': self.optimizer._optimizer.state_dict(),
			}

			if hasattr(self.model, 'need_states'):
				self.model.updateStates()
				checkpoint['extra'] = self.model.state_dict()

			val_acc_value = next(iter(val_acc.values()))

			model_name = f'model_{epoch_i:02}_acc_{100*val_acc_value:3.3f}.chkpt'
			if self.options['save_mode'] == 'all':
				torch.save(checkpoint, self.config.localPath(model_name))
			elif self.options['save_mode'] == 'best':
				#if val_loss <= min(val_losses):
				if val_acc_value > best_acc or epoch_i == 0:
					torch.save(checkpoint, self.config.localPath(model_name))
					logging.info('	- [Info] The checkpoint file has been updated.')

			if val_acc_value > best_acc or self.config['best'] is None:
				self.config['best'] = model_name

			#val_losses.append(val_loss)
			best_acc = max(best_acc, val_acc_value)

			self.tb_writer.add_scalar('loss', train_loss, epoch_i)
			self.tb_writer.add_scalar('val_loss', val_loss, epoch_i)
			for k, v in train_acc.items():
				if type(v) == dict:
					self.tb_writer.add_scalars(k, v, epoch_i)
				else:
					self.tb_writer.add_scalar(k, v, epoch_i)
			for k, v in val_acc.items():
				if type(v) == dict:
					self.tb_writer.add_scalars('val_' + k, v, epoch_i)
				else:
					self.tb_writer.add_scalar('val_' + k, v, epoch_i)
			self.tb_writer.add_scalar('learning_rate', lr, epoch_i)

			self.options['steps'] = self.optimizer.n_steps

			self.config.save()


	def train_epoch (self, dataset):
		self.model.train()
		total_loss, n_batch = 0, 0
		metric_data = {}

		for batch in tqdm(dataset, mininterval=2, desc='  - (Training)   ', leave=False):
			# forward
			self.optimizer.zero_grad()
			loss, metric = self.model(batch)

			# backward and update parameters
			loss.backward()
			self.optimizer.step()

			# note keeping
			n_batch += 1
			total_loss += loss.item()

			metric = metric if type(metric) == dict else {'acc': metric}
			for k, v in metric.items():
				metric_data[k] = metric_data[k] + v if k in metric_data else v

		stat = self.model.stat if hasattr(self.model, 'stat') else stat_average
		metrics = stat(metric_data, n_batch)

		return total_loss / n_batch, metrics


	def eval_epoch (self, dataset):
		self.model.eval()
		total_loss, n_batch = 0, 0
		metric_data = {}

		with torch.no_grad():
			for batch in tqdm(dataset, mininterval=2, desc='  - (Validation) ', leave=False):
				# forward
				loss, metric = self.model(batch)

				# note keeping
				n_batch += 1
				total_loss += loss.item()

				metric = metric if type(metric) == dict else {'acc': metric}
				for k, v in metric.items():
					metric_data[k] = metric_data[k] + v if k in metric_data else v

		stat = self.model.stat if hasattr(self.model, 'stat') else stat_average
		metrics = stat(metric_data, n_batch)

		return total_loss / n_batch, metrics


	def loadCheckpoint (self, filename):
		checkpoint = torch.load(self.config.localPath(filename), map_location=self.options['device'])
		self.model.deducer.load_state_dict(checkpoint['model'])
		self.start_epoch = checkpoint['epoch'] + 1

		if hasattr(self.model, 'need_states') and checkpoint.get('extra') is not None:
			self.model.load_state_dict(checkpoint['extra'])

		if 'optim' in checkpoint:
			self.optimizer._optimizer.load_state_dict(checkpoint['optim'])

		logging.info('Checkpoint loaded: %s', self.config.localPath(filename))
