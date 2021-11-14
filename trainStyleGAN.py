
from starry.stylegan.training_loop import training_loop



if __name__ == '__main__':
	training_loop(
		run_dir='/safe/training/stylegan/test',
		training_set_kwargs={
			'class_name': 'training.dataset.ImageFolderDataset',
			'path': '/fast/data/metfaces/metfaces-256x256.zip',
			'use_labels': False,
			'max_size': 1336,
			'xflip': False,
			'resolution': 256,
			'random_seed': 0,
		},
		data_loader_kwargs={
			'pin_memory': True,
			'prefetch_factor': 2,
			'num_workers': 3,
		},
		G_kwargs={
			'class_name': 'training.networks_stylegan3.Generator',
			'z_dim': 512,
			'w_dim': 512,
			'mapping_kwargs': {
				'num_layers': 2
			},
			'channel_base': 65536,
			'channel_max': 1024,
			'magnitude_ema_beta': 0.9994456359721023,
			'conv_kernel': 1,
			'use_radial_filters': True,
		},
		D_kwargs={
			'class_name': 'training.networks_stylegan2.Discriminator',
			'block_kwargs': {
				'freeze_layers': 0
			},
			'mapping_kwargs': {},
			'epilogue_kwargs': {
				'mbstd_group_size': 4
			},
			'channel_base': 32768,
			'channel_max': 512,
		},
		G_opt_kwargs={
			'class_name': 'torch.optim.Adam',
			'betas': [0, 0.99],
			'eps': 1e-08,
			'lr': 0.0025,
		},
		D_opt_kwargs={
			'class_name': 'torch.optim.Adam',
			'betas': [0, 0.99],
			'eps': 1e-08,
			'lr': 0.002,
		},
		augment_kwargs={
			'class_name': 'training.augment.AugmentPipe',
			'xflip': 1,
			'rotate90': 1,
			'xint': 1,
			'scale': 1,
			'rotate': 1,
			'aniso': 1,
			'xfrac': 1,
			'brightness': 1,
			'contrast': 1,
			'lumaflip': 1,
			'hue': 1,
			'saturation': 1,
		},
		loss_kwargs             = {
			'class_name': 'training.loss.StyleGAN2Loss',
			'r1_gamma': 6.6,
			'blur_init_sigma': 10,
			'blur_fade_kimg': 100.0,
		},       # Options for loss function.
		#metrics                 = [],       # Metrics to evaluate during training.
		random_seed             = 0,        # Global random seed.
		num_gpus                = 1,        # Number of GPUs participating in the training.
		batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
		batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
		ema_kimg                = 5,       # Half-life of the exponential moving average (EMA) of generator weights.
		#ema_rampup              = 0.05,     # EMA ramp-up coefficient. None = no rampup.
		#G_reg_interval          = None,     # How often to perform regularization for G? None = disable lazy regularization.
		#D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
		#augment_p               = 0,        # Initial value of augmentation probability.
		ada_target              = 0.6,     # ADA target value. None = fixed p.
		#ada_interval            = 4,        # How often to perform ADA adjustment?
		ada_kimg                = 500,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
		total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
		kimg_per_tick           = 4,        # Progress snapshot interval.
		image_snapshot_ticks    = 20,       # How often to save image snapshots? None = disable.
		network_snapshot_ticks  = 20,       # How often to save network snapshots? None = disable.
		#resume_pkl              = None,     # Network pickle to resume training from.
		#resume_kimg             = 0,        # First kimg to report when resuming training.
		#cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
	)
