
import os
import torch
from tensorboardX import SummaryWriter
import time
from tqdm import tqdm
import logging

from .optim import optim
from .model_factory import loadModel



def print_metric (metric):
	if type(metric) == float or type(metric) == torch.Tensor:
		return f'accuracy: {metric*100:3.3f}%'
	elif type(metric) == dict:
		items = map(lambda item: f'{item[0]}: {item[1]:.4f}',
			filter(lambda item: type(item[1]) in [int, float],
				metric.items()))
		return ', '.join(items)
	else:
		return str(metric)


def stat_average (data, n_batch):
	return dict([(k, v / n_batch) for k, v in data.items()])


class Moniter:
	def __init__ (self, field='loss', mode='min', best_value=None):
		self.field = field
		self.mode = mode

		self.best_value = best_value


	def update (self, metrics):
		value = metrics[self.field]
		new_record = False

		if self.best_value is None:
			new_record = True
		elif self.mode == 'min':
			new_record = value < self.best_value
		elif self.mode == 'max':
			new_record = value > self.best_value
		else:
			assert False, f'unexpected moniter mode: {self.mode}'

		if new_record:
			self.best_value = value

		return value, new_record


class Trainer:
	def __init__ (self, config):
		self.config = config
		self.options = config['trainer']
		self.moniter = Moniter(**self.options.get('moniter', {}))

		if self.options.get('env'):
			config.setEnv(self.options['env'])

		self.start_epoch = 0

		self.model = loadModel(config['model'], postfix='Loss')
		self.model.to(self.options['device'])

		self.optimizer = optim(config['optim'], self.model.parameters(), init_step=self.options.get('steps', 0))

		latest_path = 'latest.chkpt' if os.path.exists(self.config.localPath('latest.chkpt')) else self.config['best']
		weights = latest_path or self.config['trainer.pretrained_weights']
		if weights:
			self.loadCheckpoint(weights)

		self.tb_writer = SummaryWriter(log_dir=config.dir)


	def reportScalars (self, scalars, step):
		for k, v in scalars.items():
			if type(v) == dict:
				for kk, vv in v.items():
					self.tb_writer.add_scalar(f'{k}/{kk}', vv, step)
			else:
				self.tb_writer.add_scalar(k, v, step)


	def train (self, training_data, validation_data):
		def print_performances(header, loss, metric, start_time, lr):
			print('  - {header:12} loss: {loss: .4e}, {metric}, lr: {lr:.4e}, elapse: {elapse:3.2f} min'
				.format(header=f"({header})", loss=loss, metric=print_metric(metric), elapse=(time.time()-start_time)/60, lr=lr))

		report_step = 0
		checkpoint = None
		for epoch_i in range(self.start_epoch, self.options['epoch']):
			logging.info(f'[Epoch {epoch_i}]')

			# validation
			start = time.time()
			val_loss, val_metric = self.eval_epoch(validation_data)
			lr = self.optimizer._optimizer.param_groups[0]['lr']
			print_performances('Validation', val_loss, val_metric, start, lr)

			if hasattr(self.model, 'need_states'):
				self.model.updateStates()
				if checkpoint is not None:
					checkpoint['extra'] = self.model.state_dict()

			moniter_value, new_record = self.moniter.update({
				**val_metric,
				'loss': val_loss,
			})

			if checkpoint is not None:
				model_name = f'model_{epoch_i:02}_{self.moniter.field}_{moniter_value:.3e}.chkpt'
				if self.options['save_mode'] == 'all':
					torch.save(checkpoint, self.config.localPath(model_name))
				elif self.options['save_mode'] == 'best':
					if new_record or epoch_i == 0:
						torch.save(checkpoint, self.config.localPath(model_name))
						logging.info('	- [Info] The checkpoint file has been updated.')

				if new_record or self.config['best'] is None:
					self.config['best'] = model_name
					self.config['trainer.moniter.best_value'] = self.moniter.best_value

			# write tensorboard scalars
			scalars = {
				'val_loss': val_loss,
			}
			for k, v in val_metric.items():
				scalars['val_' + k] = v
			self.reportScalars(scalars, report_step)

			self.options['steps'] = self.optimizer.n_steps
			self.config.save()

			# training
			start = time.time()
			train_loss, train_metric = self.train_epoch(training_data)
			#train_ppl = math.exp(min(train_loss, 100))

			# Current learning rate
			lr = self.optimizer._optimizer.param_groups[0]['lr']
			print_performances('Training', train_loss, train_metric, start, lr)

			checkpoint = {
				'epoch': epoch_i,
				'model': self.model.deducer.state_dict(),
				'optim': self.optimizer._optimizer.state_dict(),
			}
			if hasattr(self.model, 'need_states'):
				checkpoint['extra'] = self.model.state_dict()
			torch.save(checkpoint, self.config.localPath('latest.chkpt'))

			scalars = {
				'loss': train_loss,
				'learning_rate': lr,
				**train_metric,
			}
			report_step_unit = self.options.get('report_step_unit')
			report_step = self.optimizer.n_steps * self.config['data.batch_size'] if report_step_unit == 'examples' else epoch_i
			self.reportScalars(scalars, report_step)


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
			self.model.load_state_dict(checkpoint['extra'], strict=False)

		if 'optim' in checkpoint:
			self.optimizer._optimizer.load_state_dict(checkpoint['optim'])

		logging.info('Checkpoint loaded: %s', self.config.localPath(filename))
