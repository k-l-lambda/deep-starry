
import torch



class BaseScheduler:
	def __init__ (self, optimizer, init_step=0):
		self._optimizer = optimizer
		self.n_steps = init_step


	def step (self):
		# Step with the inner optimizer
		self._update_learning_rate()
		self._optimizer.step()


	def zero_grad (self):
		# Zero out the gradients with the inner optimizer
		self._optimizer.zero_grad()


	# abstractmethod
	def _get_lr (self) -> float:
		pass


	def _update_learning_rate (self):
		# Learning rate scheduling per step

		self.n_steps += 1
		lr = self._get_lr()

		for param_group in self._optimizer.param_groups:
			param_group['lr'] = lr


class InvSqrtScheduler (BaseScheduler):
	name = 'InvSqrt'

	def __init__ (self, optimizer, lr_mul, d_model, n_warmup_steps, init_step=0):
		super().__init__(optimizer, init_step)

		self.lr_mul = lr_mul
		self.d_model = d_model
		self.n_warmup_steps = n_warmup_steps

	# overload
	def _get_lr (self):
		d_model = self.d_model
		n_steps, n_warmup_steps, lr_mul = self.n_steps, self.n_warmup_steps, self.lr_mul

		return lr_mul * (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))


scheduler_dict = dict(map(lambda cls: (cls.name, cls), [
	InvSqrtScheduler,
]))


def createOptimizer (config, parameters):
	if config['type'] == 'Adam':
		return torch.optim.Adam(parameters, **config['args'])
	else:
		raise RuntimeError("Unexpected optimizer type '%s'" % config['type'])


def optim (config, parameters, init_step):
	scheduler = config['scheduler']
	scheduler_type = scheduler['type']
	if scheduler_type not in scheduler_dict:
		raise RuntimeError("Scheduler type %s not found" % scheduler_type)

	scheduler_class = scheduler_dict[scheduler_type]

	optimizer = createOptimizer(config, parameters)

	return scheduler_class(optimizer, init_step=init_step, **scheduler['args'])
