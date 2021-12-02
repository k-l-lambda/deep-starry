
import torch
from tensorboardX import SummaryWriter
import time
from tqdm import tqdm
import logging

from .optim import optim
from .model_factory import loadModel
from .trainer import Moniter, print_metric, stat_average



class Trainer:
	def __init__ (self, config, rank=0):
		self.config = config
		self.options = config['trainer']
		self.moniter = Moniter(**self.options.get('moniter', {}))
		self.rank = rank

		self.start_epoch = 0

		self.model = loadModel(config['model'], postfix='Loss')
		self.model.to(self.options['device'])

		if self.rank == 0:
			self.optimizer = optim(config['optim'], self.model.parameters(), init_step=self.options.get('steps', 0))

			weights = self.config['best'] or self.config['trainer.pretrained_weights']
			if weights:
				self.loadCheckpoint(weights)

			# TODO: broadcast parameters

		#self.tb_writer = SummaryWriter(log_dir=config.dir)


	def print_performances(self, header, loss, metric, start_time, lr):
		logging.info('  - {header:12} loss: {loss: .4e}, {metric}, lr: {lr:.4e}, elapse: {elapse:3.2f} min'
			.format(header=f"({header})", loss=loss, metric=print_metric(metric), elapse=(time.time()-start_time)/60, lr=lr))


	def train (self, data):
		data_it = self.infiniteTraverse(data)

		for epoch_i in range(self.start_epoch, self.options['epoch']):
			logging.info(f'[Epoch {epoch_i}]')

			start = time.time()

			self.model.train()
			total_loss, n_batch = 0, 0
			metric_data = {}

			for batch in tqdm(self.finiteTraverse(data_it, self.options['epoch_size']), mininterval=1,
				desc='  - (Training)   ', leave=False, position=self.rank):
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
			train_loss = total_loss / n_batch

			lr = self.optimizer._optimizer.param_groups[0]['lr']
			self.print_performances('Training', train_loss, metrics, start, lr)

			checkpoint = {
				'epoch': epoch_i,
				'model': self.model.deducer.state_dict(),
				'optim': self.optimizer._optimizer.state_dict(),
			}
			torch.save(checkpoint, self.config.localPath('latest.chkpt'))

			# TODO: receive parameters from validator process

			#if hasattr(self.model, 'need_states'):
			#	self.model.updateStates()
			#	checkpoint['extra'] = self.model.state_dict()

			self.options['steps'] = self.optimizer.n_steps

			self.config.save()


	def validate (self, data):
		for epoch_i in range(self.start_epoch, self.options['epoch']):
			checkpoint = {
				'epoch': epoch_i,
				'model': self.model.deducer.state_dict(),
				'optim': self.optimizer._optimizer.state_dict(),
			}

			start = time.time()
			#val_loss, val_acc = self.eval_epoch(data)

			self.model.eval()
			total_loss, n_batch = 0, 0
			metric_data = {}

			with torch.no_grad():
				for batch in tqdm(data, mininterval=2, desc='  - (Validation) ', leave=False, position=self.rank):
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

			val_loss = total_loss / n_batch

			self.print_performances('Validation', val_loss, metrics, start, 0)

			if hasattr(self.model, 'need_states'):
				self.model.updateStates()
				checkpoint['extra'] = self.model.state_dict()

			moniter_value, new_record = self.moniter.update({
				**metrics,
				'loss': val_loss,
			})

			model_name = f'model_{epoch_i:02}_{self.moniter.field}_{moniter_value:.3f}.chkpt'
			if self.options['save_mode'] == 'all':
				torch.save(checkpoint, self.config.localPath(model_name))
			elif self.options['save_mode'] == 'best':
				if new_record or epoch_i == 0:
					torch.save(checkpoint, self.config.localPath(model_name))
					logging.info('	- [Info] The checkpoint file has been updated.')

			if new_record or self.config['best'] is None:
				self.config['best'] = model_name
				self.config['trainer.moniter.best_value'] = self.moniter.best_value

			'''# write tensorboard scalars
			scalars = {
				'val_loss': val_loss,
				**metrics,
			}
			for k, v in metrics.items():
				scalars['val_' + k] = v

			for k, v in scalars.items():
				if type(v) == dict:
					self.tb_writer.add_scalars(k, v, epoch_i)
				else:
					self.tb_writer.add_scalar(k, v, epoch_i)'''


	def infiniteTraverse (self, dataset):
		while True:
			for batch in dataset:
				yield batch


	def finiteTraverse (self, iter, count):
		i = 0
		while i < count:
			batch = next(iter)
			i += len(batch)

			yield batch


	def loadCheckpoint (self, filename):
		checkpoint = torch.load(self.config.localPath(filename), map_location=self.options['device'])
		self.model.deducer.load_state_dict(checkpoint['model'])
		self.start_epoch = checkpoint['epoch'] + 1

		if hasattr(self.model, 'need_states') and checkpoint.get('extra') is not None:
			self.model.load_state_dict(checkpoint['extra'])

		if 'optim' in checkpoint:
			self.optimizer._optimizer.load_state_dict(checkpoint['optim'])

		logging.info('Checkpoint loaded: %s', self.config.localPath(filename))
