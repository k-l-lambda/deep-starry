
import argparse
import itertools
import math
import os
import sys
import logging
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import CLIPTextModel, CLIPTokenizer, CLIPFeatureExtractor
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

from starry.utils.config import Configuration
from starry.utils.iterators import FixedLengthIterator
from starry.vision.data import PerisCaption



logging.basicConfig(stream=sys.stdout, level=logging.INFO)

logger = get_logger(__name__)


DATA_DIR = os.environ.get('DATA_DIR')


def freeze_params(params):
	for param in params:
		param.requires_grad = False


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str)

	args = parser.parse_args()

	config = Configuration.createOrLoad(args.config)

	logging.basicConfig(format='%(asctime)s	%(levelname)s	%(message)s', datefmt='%Y%m%d %H:%M:%S', level=logging.WARN,
		force=True, handlers=[
			logging.StreamHandler(sys.stdout),
			logging.FileHandler(config.localPath('trainer.log')),
		])

	gradient_accumulation_steps = config['trainer.gradient_accumulation_steps']
	train_batch_size = config['trainer.train_batch_size']

	accelerator = Accelerator(
		gradient_accumulation_steps=gradient_accumulation_steps,
		mixed_precision=config['trainer.mixed_precision'],
		log_with='tensorboard',
		logging_dir=config.dir,
	)

	# If passed along, set the training seed now.
	if config['trainer.seed'] is not None:
		set_seed(config['trainer.seed'])

	pretrained_path = config['trainer.pretrained_model_name_or_path']
	tokenizer = CLIPTokenizer.from_pretrained(pretrained_path, subfolder='tokenizer')
	text_encoder = CLIPTextModel.from_pretrained(pretrained_path, subfolder='text_encoder')
	vae = AutoencoderKL.from_pretrained(pretrained_path, subfolder='vae')
	unet = UNet2DConditionModel.from_pretrained(pretrained_path, subfolder='unet')

	safety_checker = StableDiffusionSafetyChecker.from_pretrained(os.path.join(pretrained_path, 'safety_checker'))
	feature_extractor = CLIPFeatureExtractor.from_pretrained(os.path.join(pretrained_path, 'feature_extractor'))

	placeholder_token_ids = tokenizer.convert_tokens_to_ids([f'{token}</w>' for token in config['trainer.tokens']])
	freezed_ids = torch.tensor([id for id in range(len(tokenizer)) if not id in placeholder_token_ids])

	# Freeze vae and unet
	freeze_params(vae.parameters())
	freeze_params(unet.parameters())

	# Freeze all parameters except for the token embeddings in text encoder
	params_to_freeze = itertools.chain(
		text_encoder.text_model.encoder.parameters(),
		text_encoder.text_model.final_layer_norm.parameters(),
		text_encoder.text_model.embeddings.position_embedding.parameters(),
	)
	freeze_params(params_to_freeze)

	learning_rate = config['trainer.learning_rate']
	if config['trainer.scale_lr']:
		learning_rate *= gradient_accumulation_steps * train_batch_size * accelerator.num_processes

	# Initialize the optimizer
	optimizer = torch.optim.AdamW(
		text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
		lr=learning_rate,
		betas=(config['optim.adam_beta1'], config['optim.adam_beta2']),
		weight_decay=config['optim.adam_weight_decay'],
		eps=config['optim.adam_epsilon'],
	)

	noise_scheduler = DDPMScheduler(
		beta_start=0.00085,
		beta_end=0.012,
		beta_schedule='scaled_linear',
		num_train_timesteps=config['trainer.num_train_timesteps'],
		tensor_format='pt',
	)

	root = os.path.join(DATA_DIR, config['data.root'])
	labels = os.path.join(DATA_DIR, config['data.labels'])
	train_dataset = PerisCaption(root, labels, tokenizer, shuffle=True, **config['data.args'])
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size)

	# Scheduler and math around the number of training steps.
	overrode_max_train_steps = False
	num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
	max_train_steps = config['trainer.max_train_steps']
	if max_train_steps is None:
		max_train_steps = config['trainer.num_train_epochs'] * num_update_steps_per_epoch
		overrode_max_train_steps = True

	lr_scheduler = get_scheduler(
		config['trainer.lr_scheduler'],
		optimizer=optimizer,
		num_warmup_steps=config['trainer.lr_warmup_steps'] * gradient_accumulation_steps,
		num_training_steps=max_train_steps * gradient_accumulation_steps,
	)

	text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
		text_encoder, optimizer, train_dataloader, lr_scheduler
	)

	epoch_dataloader = FixedLengthIterator(train_dataloader, length=config['data.epoch_size']) if config['data.epoch_size'] else train_dataloader

	# Move vae and unet to device
	vae.to(accelerator.device)
	unet.to(accelerator.device)

	# Keep vae and unet in eval model as we don't train these
	vae.eval()
	unet.eval()

	# We need to recalculate our total training steps as the size of the training dataloader may have changed.
	num_update_steps_per_epoch = math.ceil(len(epoch_dataloader) / gradient_accumulation_steps)
	if overrode_max_train_steps:
		max_train_steps = config['trainer.num_train_epochs'] * num_update_steps_per_epoch
	# Afterwards we recalculate our number of training epochs
	num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

	# We need to initialize the trackers we use, and also store our configuration.
	# The trackers initializes automatically on the main process.
	if accelerator.is_main_process:
		tracker_config = {key: config['trainer'][key] for key in ['pretrained_model_name_or_path', 'num_train_timesteps', 'gradient_accumulation_steps',
			'max_train_steps', 'learning_rate', 'scale_lr', 'lr_scheduler', 'lr_warmup_steps', 'num_train_epochs']}
		accelerator.init_trackers('log', config=tracker_config)

	# Train!
	total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

	logger.info('***** Running training *****')
	logger.info(f'  Num examples = {len(train_dataset)}')
	logger.info(f'  Num Epochs = {num_train_epochs}')
	logger.info(f'  Instantaneous batch size per device = {train_batch_size}')
	logger.info(f'  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}')
	logger.info(f'  Gradient Accumulation steps = {gradient_accumulation_steps}')
	logger.info(f'  Total optimization steps = {max_train_steps}')

	# Only show the progress bar once on each machine.
	progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
	progress_bar.set_description('Steps')
	global_step = 0

	for epoch in range(num_train_epochs):
		logger.info(f'Epoch {epoch}')

		text_encoder.train()
		for step, batch in enumerate(epoch_dataloader):
			with accelerator.accumulate(text_encoder):
				# Convert images to latent space
				latents = vae.encode(batch['pixel_values']).latent_dist.sample().detach()
				latents = latents * 0.18215

				# Sample noise that we'll add to the latents
				noise = torch.randn(latents.shape).to(latents.device)
				bsz = latents.shape[0]
				# Sample a random timestep for each image
				timesteps = torch.randint(
					0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
				).long()

				# Add noise to the latents according to the noise magnitude at each timestep
				# (this is the forward diffusion process)
				noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

				# Get the text embedding for conditioning
				encoder_hidden_states = text_encoder(batch['input_ids'])[0]

				# Predict the noise residual
				noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

				loss = F.mse_loss(noise_pred, noise, reduction='none').mean([1, 2, 3]).mean()
				accelerator.backward(loss)

				# Zero out the gradients for all token embeddings except the newly added
				# embeddings for the concept, as we only want to optimize the concept embeddings
				if accelerator.num_processes > 1:
					grads = text_encoder.module.get_input_embeddings().weight.grad
				else:
					grads = text_encoder.get_input_embeddings().weight.grad

				# Get the index for tokens that we want to zero the grads for
				grads.data[freezed_ids, :] = grads.data[freezed_ids, :].fill_(0)

				optimizer.step()
				lr_scheduler.step()
				optimizer.zero_grad()

			# Checks if the accelerator has performed an optimization step behind the scenes
			if accelerator.sync_gradients:
				progress_bar.update(1)
				global_step += 1

			logs = {'loss': loss.detach().item(), 'lr': lr_scheduler.get_last_lr()[0]}
			progress_bar.set_postfix(**logs)
			accelerator.log(logs, step=global_step)

			if global_step >= max_train_steps:
				break

		accelerator.wait_for_everyone()

		# Create the pipeline using using the trained modules and save it.
		if accelerator.is_main_process:
			pipeline = StableDiffusionPipeline(
				text_encoder=accelerator.unwrap_model(text_encoder),
				vae=vae,
				unet=unet,
				tokenizer=tokenizer,
				scheduler=PNDMScheduler(
					beta_start=0.00085,
					beta_end=0.012,
					beta_schedule='scaled_linear',
					skip_prk_steps=True,
				),
				safety_checker=safety_checker,
				feature_extractor=feature_extractor,
			)
			pipeline.save_pretrained(config.localPath('stable-diffusion'))

			# Also save the newly trained embeddings
			learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight
			learned_embeds_dict = {id: learned_embeds[id].detach().cpu() for id in placeholder_token_ids}
			torch.save(learned_embeds_dict, config.localPath('learned_embeds.bin'))

	accelerator.end_training()


if __name__ == '__main__':
	main()
