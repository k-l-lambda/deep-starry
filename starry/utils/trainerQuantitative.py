
import os
import sys
import torch
from tensorboardX import SummaryWriter
import time
from tqdm import tqdm
import logging

from .optim import optim
from .model_factory import loadModel
from .trainer import Moniter, print_metric, stat_average
from .dataset_factory import loadDataset



class Trainer:
	TRAINER_RANK = 0
	VALIDATOR_RANK = 1
	PROC_COUNT = 2


	@staticmethod
	def run (rank, config, data_dir, backend='nccl'):
		logging.basicConfig(filename=config.localPath('trainer.log'),
			format='%(asctime)s	%(levelname)s	%(message)s', datefmt='%H:%M:%S', level=logging.INFO)
		logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

		init_file = os.path.abspath(config.localPath('.torch_distributed_init'))
		torch.distributed.init_process_group(backend=backend, init_method=f'file://{init_file}', rank=rank, world_size=Trainer.PROC_COUNT)

		device = torch.device(config['trainer.device'], rank)
		trainer = Trainer(config, device=device, rank=rank)

		trainer.log('*	Loading data.')

		splits = config['data.splits'].split(':')
		data, = loadDataset(config, data_dir=data_dir, device=device,
			splits=splits[rank], batch_size=config['trainer.val_batch_size'] if rank == Trainer.VALIDATOR_RANK else None)

		if rank == Trainer.TRAINER_RANK:
			trainer.train(data)
		elif rank == Trainer.VALIDATOR_RANK:
			trainer.validate(data)

		torch.distributed.destroy_process_group()


	def __init__ (self, config, device, rank=0):
		self.config = config
		self.options = config['trainer']
		self.device = device
		self.rank = rank
		self.role = 'TR' if rank == Trainer.TRAINER_RANK else 'VA'

		self.start_epoch = 0

		self.model = loadModel(config['model'], postfix='Loss')
		self.model.to(self.options['device'])


	def log (self, message, *args):
		logging.info(f'[{self.role}]	' + message, *args)


	def print_performances(self, loss, metric, start_time, lr):
		self.log('loss: {loss: .4e}, {metric}, lr: {lr:.4e}, elapse: {elapse:3.2f} min'
			.format(loss=loss, metric=print_metric(metric), elapse=(time.time()-start_time)/60, lr=lr))


	def broadcastModule (self, module, src):
		for param in module.parameters():
			torch.distributed.broadcast(param.data.to(self.device), src=src)


	def broadcastParam (self, parameters, src):
		for param in parameters:
			torch.distributed.broadcast(param.data.to(self.device), src=src)


	def train (self, data):
		self.log('*	Initializing trainer.')

		self.optimizer = optim(self.config['optim'], self.model.parameters(), init_step=self.options.get('steps', 0))

		weights = 'latest.chkpt' if self.config['trainer.latest'] else self.config['trainer.pretrained_weights']
		if weights:
			self.loadCheckpoint(weights)

			self.log('Syncing training model parameters...')
			self.broadcastParam(self.model.training_parameters(), src=Trainer.TRAINER_RANK)

		#self.tb_writer = SummaryWriter(log_dir=config.dir)

		data_it = self.infiniteTraverse(data)

		need_states = hasattr(self.model, 'need_states')

		self.log('*	Training.')

		for epoch_i in range(self.start_epoch, self.options['epochs']):
			self.log(f'[Epoch {epoch_i}]')

			start = time.time()

			self.model.train()
			total_loss, n_batch = 0, 0
			metric_data = {}

			for batch in tqdm(self.finiteTraverse(data_it, self.options['epoch_size']), mininterval=1,
				total=self.options['epoch_size'] // self.config['data.batch_size'], desc='  - (Training)   ', position=self.rank):
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
			self.print_performances(train_loss, metrics, start, lr)

			if self.config['trainer.latest'] and need_states:
				self.broadcastParam(self.model.validation_parameters(), src=Trainer.VALIDATOR_RANK)
				self.log('Model validation parameters synchronized.')

			checkpoint = {
				'epoch': epoch_i,
				'model': self.model.deducer.state_dict(),
				'optim': self.optimizer._optimizer.state_dict(),
				'extra': self.model.state_dict() if need_states else None,
			}
			torch.save(checkpoint, self.config.localPath('latest.chkpt'))

			#self.log('Syncing training model parameters...')
			self.broadcastParam(self.model.training_parameters(), src=Trainer.TRAINER_RANK)

			self.config.load()
			self.config['trainer.steps'] = self.optimizer.n_steps
			self.config['trainer.latest'] = True
			self.config.save()


	def validate (self, data):
		self.moniter = Moniter(**self.options.get('moniter', {}))
		need_states = hasattr(self.model, 'need_states')

		if not self.config['trainer.latest']:
			self.start_epoch += 1

		for epoch_i in range(self.start_epoch, self.options['epochs'] + 1):
			self.broadcastParam(self.model.training_parameters(), src=Trainer.TRAINER_RANK)
			self.log('Model training parameters synchronized.')

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

			self.print_performances(val_loss, metrics, start, 0)

			checkpoint = {
				'epoch': epoch_i,
				'model': self.model.deducer.state_dict(),
			}

			if need_states and epoch_i < self.options['epochs']:
				self.model.updateStates()
				#checkpoint['extra'] = self.model.state_dict()
				#self.log(f'epoch_i: {epoch_i}, {self.options["epochs"]}')

				self.broadcastParam(self.model.validation_parameters(), src=Trainer.VALIDATOR_RANK)

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
					self.log('The checkpoint file has been updated.')

			if new_record or self.config['best'] is None:
				self.config.load()
				self.config['best'] = model_name
				self.config['trainer.moniter.best_value'] = self.moniter.best_value
				self.config.save()

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

		self.log('Checkpoint loaded: %s', self.config.localPath(filename))
