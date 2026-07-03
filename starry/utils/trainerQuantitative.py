
import os
import sys
import torch
import contextlib
from tensorboardX import SummaryWriter
import time
from tqdm import tqdm
import logging
import shutil
import time
import math
from datetime import timedelta

from .optim import optim
from .model_factory import loadModel
from .trainer import Moniter, print_metric, stat_average, infiniteTraverse, finiteTraverse
from .dataset_factory import loadDataset



INTERRUPTION_MARKER = '__SHUT'


class Trainer:
	TRAINER_RANK = 0
	VALIDATOR_RANK = 1
	PROC_COUNT = 2


	@staticmethod
	def run (rank, config, data_dir, init_file, backend='nccl'):
		logging.basicConfig(format='%(asctime)s	%(levelname)s	%(message)s', datefmt='%Y%m%d %H:%M:%S', level=logging.INFO,
			force=True, handlers=[
				logging.StreamHandler(sys.stdout),
				logging.FileHandler(config.localPath('trainer.log')),
			])

		init_method = f'file:///{init_file}' if os.name == 'nt' else f'file://{init_file}'
		torch.distributed.init_process_group(backend=backend, init_method=init_method, rank=rank, world_size=Trainer.PROC_COUNT, timeout=timedelta(seconds=7200))

		gpus = config['trainer.gpus'] or Trainer.PROC_COUNT
		device = torch.device(config['trainer.device'], rank % gpus)
		trainer = Trainer(config, device=device, rank=rank)

		trainer.log('*	Loading data.')

		splits = config['data.splits'].split(':')
		data, = loadDataset(config, data_dir=data_dir, device=device,
			splits=splits[rank], batch_size=config['trainer.val_batch_size'] if rank == Trainer.VALIDATOR_RANK else None)

		if rank == Trainer.TRAINER_RANK:
			trainer.train(data)
		elif rank == Trainer.VALIDATOR_RANK:
			trainer.validate(data)

		trainer.tb_writer.close()
		torch.distributed.destroy_process_group()


	def __init__ (self, config, device, rank=0):
		self.config = config
		self.options = config['trainer']
		self.device = device
		self.rank = rank
		self.role = 'TR' if rank == Trainer.TRAINER_RANK else 'VA'

		if self.options.get('env'):
			config.setEnv(self.options['env'])

		self.start_epoch = 0

		# autocast dtype: 'fp32' (default) or 'bf16'
		dtype_name = (self.options.get('dtype') or 'fp32').lower()
		self.autocast_dtype = {
			'fp32': None,
			'float32': None,
			'bf16': torch.bfloat16,
			'bfloat16': torch.bfloat16,
		}.get(dtype_name, None)
		if dtype_name not in ('fp32', 'float32') and self.autocast_dtype is None:
			logging.warning('Unknown trainer.dtype "%s", falling back to fp32.', dtype_name)

		self.model = loadModel(config['model'], postfix='Loss', imports=config['imports'])
		self.model.deducer.to(self.device)
		self.model.to(self.device)

		self.optimizer = optim(self.config['optim'], self.model.parameters(),
			init_step=self.options.get('steps', 0)) if self.rank == Trainer.TRAINER_RANK else None

		latest_path = 'latest.chkpt' if os.path.exists(self.config.localPath('latest.chkpt')) else self.config['best']
		weights_path = latest_path if self.config['trainer.latest'] else self.config['trainer.pretrained_weights']
		if weights_path:
			self.loadCheckpoint(weights_path)

		self.tb_writer = SummaryWriter(log_dir=config.localPath(self.role))

		# remove interruption marker
		if os.path.exists(config.localPath(INTERRUPTION_MARKER)):
			os.rename(config.localPath(INTERRUPTION_MARKER), config.localPath(INTERRUPTION_MARKER + '-'))


	def log (self, message, *args):
		logging.info(f'[{self.role}]	' + message, *args)


	def autocast (self):
		# autocast context for the configured dtype; no-op when fp32 or non-CUDA.
		if self.autocast_dtype is None or self.device.type != 'cuda':
			return contextlib.nullcontext()
		return torch.autocast(device_type='cuda', dtype=self.autocast_dtype)

	def _trainable_params (self):
		# params that actually receive grads (excludes a frozen encoder); matches what the
		# optimizer steps, so grad-clip norm is computed over the same set.
		return [p for p in self.model.parameters() if p.requires_grad]


	def cleanupSnapshots (self, keep=None):
		# Remove all model_*.chkpt snapshots except `keep`. Used in save_mode='best'
		# so the optim-bearing per-epoch snapshots don't accumulate and fill the disk;
		# best.chkpt retains the model weights independently.
		import glob
		keep_path = self.config.localPath(keep) if keep else None
		for path in glob.glob(self.config.localPath('model_*.chkpt')):
			if keep_path and os.path.abspath(path) == os.path.abspath(keep_path):
				continue
			try:
				os.remove(path)
				self.log('Removed old snapshot: %s', os.path.basename(path))
			except OSError as e:
				self.log('Failed to remove snapshot %s: %s', os.path.basename(path), e)


	def print_performances(self, loss, metric, start_time, lr=math.nan):
		self.log('loss: {loss: .4e}, {metric}, lr: {lr:.4e}, elapse: {elapse:3.2f} min'
			.format(loss=loss, metric=print_metric(metric), elapse=(time.time()-start_time)/60, lr=lr))


	def broadcastModule (self, module, src):
		for param in module.parameters():
			torch.distributed.broadcast(param, src=src)

	def broadcastParam (self, parameters, src):
		for param in parameters:
			torch.distributed.broadcast(param.detach() if src == self.rank else param, src=src)


	def broadcastScalar (self, scalar=None, src=0):
		t = torch.tensor(scalar or 0, device=self.device)
		torch.distributed.broadcast(t, src=src)

		return t.cpu().item()


	def reportScalars (self, scalars, step):
		for k, v in scalars.items():
			if type(v) == dict:
				for kk, vv in v.items():
					self.tb_writer.add_scalar(f'{k}/{kk}', vv, step)
			else:
				self.tb_writer.add_scalar(k, v, step)


	@property
	def exampleN (self):
		return (self.config['trainer.steps'] or 0) * self.config['data.batch_size']


	def train (self, data):
		self.log('*	Initializing trainer.')

		#self.broadcastScalar(self.start_epoch, src=self.rank)

		if self.config['trainer.latest']:
			self.log('Syncing training model parameters...')
			#self.model.requires_grad_(False)
			self.broadcastParam(self.model.training_parameters(), src=Trainer.TRAINER_RANK)

		data_it = infiniteTraverse(data)

		need_states = hasattr(self.model, 'need_states')

		self.log('*	Training.')

		for epoch_i in range(self.start_epoch, self.options['epochs']):
			if os.path.exists(self.config.localPath(INTERRUPTION_MARKER)):
				logging.warn('Trainer interrupted by marker!')
				break

			self.log(f'[Epoch {epoch_i}]')

			start = time.time()

			self.model.train()
			total_loss, n_batch = 0, 0
			metric_data = {}
			n_steps = self.options['epoch_size'] // self.config['data.batch_size']

			grad_clip = self.options.get('grad_clip')

			for batch in tqdm(finiteTraverse(data_it, n_steps), mininterval=1, leave=False,
				total=n_steps, desc='  - (Training)   ', position=self.rank):
				# forward
				self.optimizer.zero_grad()
				with self.autocast():
					loss, metric = self.model(batch)

				# Skip a pathological batch: a non-finite loss would poison every weight through
				# backward+step. Drop it (no grad applied) rather than corrupt the model.
				if not torch.isfinite(loss):
					logging.warning('non-finite loss (%s) at epoch %d; skipping batch', loss.item(), epoch_i)
					continue

				# backward and update parameters
				loss.backward()
				# Gradient clipping (trainer.grad_clip): bounds the update norm so a single spiky
				# batch can't knock the model into a bad basin (the InvSqrt-decayed LR can't climb
				# back out). Clip the SAME trainable params the optimizer sees. clip_grad_norm_
				# returns the PRE-clip total norm even when grad_clip is None (max_norm=inf) — a
				# system-level metric (like loss) so we can see the norm distribution / spikes.
				grad_norm = torch.nn.utils.clip_grad_norm_(self._trainable_params(),
					grad_clip if grad_clip else float('inf'))
				self.optimizer.step()

				# note keeping
				n_batch += 1
				total_loss += loss.item()

				metric = metric if type(metric) == dict else {'acc': metric}
				metric = {**metric, 'grad_norm': float(grad_norm)}
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
				'steps': self.optimizer.n_steps,		# LR-scheduler step count, so resume restores the exact LR
				'model': self.model.deducer.state_dict(),
				'optim': self.optimizer._optimizer.state_dict(),
				'extra': self.model.state_dict() if need_states else None,
			}
			torch.save(checkpoint, self.config.localPath('latest.chkpt'))	# NOTE: nccl backend will stuck here

			self.log('Syncing training model parameters...')
			self.broadcastParam(self.model.training_parameters(), src=Trainer.TRAINER_RANK)

			self.config.load()
			self.config['trainer.steps'] = self.optimizer.n_steps
			self.config['trainer.latest'] = True
			self.config.save()

			# write tensorboard scalars
			scalars = {
				'loss': train_loss,
				'learning_rate': lr,
				**metrics,
			}
			report_step_unit = self.options.get('report_step_unit')
			report_step = self.exampleN if report_step_unit == 'examples' else epoch_i
			self.reportScalars(scalars, report_step)


	def validate (self, data):
		self.moniter = Moniter(**self.options.get('moniter', {}))
		need_states = hasattr(self.model, 'need_states')

		#self.start_epoch = self.broadcastScalar(src=Trainer.TRAINER_RANK)

		if self.config['trainer.latest']:
			self.start_epoch -= 1
		else:
			# sleep for the first training epoch
			epoch_duration = self.options.get('epoch_duration', 1800)
			self.log('Sleep %s seconds for the first training epoch...', epoch_duration)
			time.sleep(epoch_duration)
		#self.log('start_epoch: %d', self.start_epoch)

		with torch.no_grad():
			self.model.eval().requires_grad_(False)
			for epoch_i in range(self.start_epoch, self.options['epochs']):
				if os.path.exists(self.config.localPath(INTERRUPTION_MARKER)):
					logging.warn('Trainer interrupted by marker!')
					break

				self.log('Waiting for training parameters...')
				self.broadcastParam(self.model.training_parameters(), src=Trainer.TRAINER_RANK)
				self.log('Model training parameters synchronized.')

				start = time.time()
				#val_loss, val_acc = self.eval_epoch(data)

				total_loss, n_batch = 0, 0
				metric_data = {}

				for batch in tqdm(data, mininterval=1, desc='  - (Validation) ', leave=False, position=self.rank):
					# forward
					with self.autocast():
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

				self.print_performances(val_loss, metrics, start)

				moniter_value, new_record = self.moniter.update({
					**metrics,
					'loss': val_loss,
				})

				model_name = f'model_{epoch_i:02}_{self.moniter.field}_{moniter_value:.3e}.chkpt'
				if self.options['save_mode'] == 'all':
					if os.path.isfile(self.config.localPath('latest.chkpt')):
						shutil.move(self.config.localPath('latest.chkpt'), self.config.localPath(model_name))
					time.sleep(1)
				elif self.options['save_mode'] == 'best':
					if new_record or epoch_i == 0:
						if os.path.isfile(self.config.localPath('latest.chkpt')):
							shutil.move(self.config.localPath('latest.chkpt'), self.config.localPath(model_name))
						time.sleep(1)

						checkpoint = {
							'epoch': epoch_i,
							'model': self.model.deducer.state_dict(),
						}
						torch.save(checkpoint, self.config.localPath('best.chkpt'))

						# dynamic cleanup: keep only the newest model_*.chkpt snapshot
						# (best.chkpt already holds the model weights), so the optim-bearing
						# 6GB per-epoch snapshots don't accumulate and fill the disk.
						self.cleanupSnapshots(keep=model_name)

						self.log('The checkpoint file has been updated.')

				self.config.load()
				if new_record or self.config['best'] is None:
					self.config['best'] = model_name
					self.config['trainer.moniter.best_value'] = self.moniter.best_value
					self.config.save()

				if need_states and epoch_i < self.options['epochs'] - 1:
					self.model.updateStates()
					#checkpoint['extra'] = self.model.state_dict()
					#self.log(f'epoch_i: {epoch_i}, {self.options["epochs"]}')

					self.broadcastParam(self.model.validation_parameters(), src=Trainer.VALIDATOR_RANK)

				# write tensorboard scalars
				scalars = {
					'val_loss': val_loss,
					**metrics,
				}
				report_step_unit = self.options.get('report_step_unit')
				report_step = self.exampleN if report_step_unit == 'examples' else epoch_i
				self.reportScalars(scalars, report_step)


	def loadCheckpoint (self, filename):
		checkpoint = torch.load(self.config.localPath(filename), map_location=self.options['device'])
		self.model.deducer.load_state_dict(checkpoint['model'])
		self.start_epoch = checkpoint['epoch'] + 1

		if hasattr(self.model, 'need_states') and checkpoint.get('extra') is not None:
			self.model.load_state_dict(checkpoint['extra'], strict=False)

		if 'optim' in checkpoint and self.optimizer is not None:
			self.optimizer._optimizer.load_state_dict(checkpoint['optim'])

		# Restore the LR-scheduler step count. Prefer the value saved IN the checkpoint (it always
		# matches these exact weights) over config['trainer.steps'] (a separate .state.yaml field
		# that can drift out of sync — e.g. after a checkpoint swap). Fall back to config for old
		# checkpoints saved before `steps` was persisted.
		if self.optimizer is not None:
			steps = checkpoint.get('steps')
			if steps is None:
				steps = self.config['trainer.steps'] or 0
			self.optimizer.n_steps = steps

		self.log('Checkpoint loaded: %s (steps=%s)', self.config.localPath(filename),
			self.optimizer.n_steps if self.optimizer is not None else 'n/a')
