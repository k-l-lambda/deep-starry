
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...conformer.modules import Linear
from .conformerU import ConformerEncoderU, ConformerEncoderDecoderU
from .vocalModules import VocalEncoder, MidiEncoder1, MidiEncoder2, MidiEncoder3, MidiEncoder4, Jointer2
from .modules import Encoder, Decoder
from ..vocal import TICK_ROUND_UNIT



class VocalAnalyzer (nn.Module):
	def __init__ (self, n_class=1, encoder_dim=128, **args):
		super().__init__()

		self.vocalEncoder = VocalEncoder(encoder_dim)
		self.encoder = ConformerEncoderU(encoder_dim=encoder_dim, **args)
		self.fc = Linear(encoder_dim, n_class, bias=False)


	def forward (self, pitch, gain):
		x = self.vocalEncoder(pitch, gain)
		x = self.encoder(x)
		x = self.fc(x)
		x = torch.sigmoid(x)

		return x


class VocalAnalyzerLoss (nn.Module):
	def __init__ (self, output_field='head', **args):
		super().__init__()

		self.deducer = VocalAnalyzer(**args)
		self.output_field = output_field


	def forward (self, batch):
		target = batch[self.output_field].unsqueeze(-1)
		pred = self.deducer(batch['pitch'], batch['gain'])

		loss = F.binary_cross_entropy(pred, target)

		acc = ((pred > 0.5).float() == target).float().mean()

		true_part = target == 1
		false_part = target == 0

		true_error = (pred[true_part] <= 0.5).float().mean()
		false_error = (pred[false_part] > 0.5).float().mean()

		return loss, {
			'acc': acc.item(),
			'true_error': true_error.item(),
			'false_error': false_error.item(),
		}


class VocalAnalyzerNotation (nn.Module):
	def __init__ (self, n_class=1, midiEncoder=MidiEncoder1, encoder_dim=128, d_time=128, notationArgs={}, num_down_layers=2, n_notation_enc_layers=4, n_notation_dec_layers=3, **args):
		super().__init__()

		d_inner = encoder_dim << num_down_layers

		self.vocalEncoder = VocalEncoder(encoder_dim)
		self.midiEncoder = midiEncoder(d_inner, d_time=d_time)
		self.notationEncoder = Encoder(n_notation_enc_layers, d_model=d_inner, **notationArgs)
		self.notationDecoder = Decoder(n_notation_dec_layers, d_model=d_inner, **notationArgs)
		self.encoder = ConformerEncoderDecoderU(encoder_dim=encoder_dim, decoder=self.notationDecoder, num_down_layers=num_down_layers, **args)
		self.fc = Linear(encoder_dim, n_class, bias=False)


	def forward (self, pitch, gain, midi_pitch, midi_tick):
		vocal = self.vocalEncoder(pitch, gain)
		midi = self.midiEncoder(midi_pitch, midi_tick)
		notation = self.notationEncoder(midi)

		x = self.encoder(vocal, notation)
		x = self.fc(x)

		return x


class VocalAnalyzerNotationBinary (nn.Module):
	def __init__ (self, **args):
		super().__init__()

		self.backbone = VocalAnalyzerNotation(**args)


	def forward (self, pitch, gain, midi_pitch, midi_tick):
		x = self.backbone(pitch, gain, midi_pitch, midi_tick)
		x = torch.sigmoid(x)

		return x


class VocalAnalyzerNotationBinaryLoss (nn.Module):
	def __init__ (self, output_field='head', tick_filed='midi_tick', **args):
		super().__init__()

		self.deducer = VocalAnalyzerNotationBinary(**args)
		self.output_field = output_field
		self.tick_filed = tick_filed


	def forward (self, batch):
		target = batch[self.output_field].unsqueeze(-1)
		pred = self.deducer(batch['pitch'], batch['gain'], batch['midi_pitch'], batch[self.tick_filed])

		loss = F.binary_cross_entropy(pred, target)

		acc = ((pred > 0.5).float() == target).float().mean()

		true_part = target == 1
		false_part = target == 0

		true_error = (pred[true_part] <= 0.5).float().mean()
		false_error = (pred[false_part] > 0.5).float().mean()

		return loss, {
			'acc': acc.item(),
			'true_error': true_error.item(),
			'false_error': false_error.item(),
		}


class VocalAnalyzerNotationRegress (nn.Module):
	def __init__ (self, **args):
		super().__init__()

		self.backbone = VocalAnalyzerNotation(**args)


	def forward (self, pitch, gain, midi_pitch, midi_tick):
		x = self.backbone(pitch, gain, midi_pitch, midi_tick)

		return x


class VocalAnalyzerNotationRegressLoss (nn.Module):
	def __init__ (self, output_field='tonf', tick_filed='midi_rtick', **args):
		super().__init__()

		self.deducer = VocalAnalyzerNotationRegress(**args)
		self.output_field = output_field
		self.tick_filed = tick_filed


	def forward (self, batch):
		target = batch[self.output_field].unsqueeze(-1)
		pred = self.deducer(batch['pitch'], batch['gain'], batch['midi_pitch'], batch[self.tick_filed])

		# mask tail
		nmask = torch.logical_not(batch['mask'])
		pred[nmask] = target[nmask]

		loss = F.mse_loss(pred, target)
		error = torch.sqrt(loss)

		return loss, {
			'error': error.item(),
		}


class VocalAnalyzerNotationClassification (nn.Module):
	def __init__ (self, **args):
		super().__init__()

		self.backbone = VocalAnalyzerNotation(midiEncoder=MidiEncoder2, **args)


	def forward (self, pitch, gain, midi_pitch, midi_tick):
		x = self.backbone(pitch, gain, midi_pitch, midi_tick)
		x = x.permute(0, 2, 1)
		#x = F.softmax(x, dim=1)

		return x


class VocalAnalyzerNotationClassificationLoss (nn.Module):
	def __init__ (self, n_class=100, output_field='nonf', tick_filed='midi_rtick', **args):
		super().__init__()

		self.deducer = VocalAnalyzerNotationClassification(n_class=n_class, **args)
		self.output_field = output_field
		self.tick_filed = tick_filed
		self.n_class = n_class


	def forward (self, batch):
		target = batch[self.output_field]
		pred = self.deducer(batch['pitch'], batch['gain'], batch['midi_pitch'], batch[self.tick_filed])

		target = (target / TICK_ROUND_UNIT).long()

		mask = batch['mask'][:, 0, :]

		loss = F.cross_entropy(pred, target, ignore_index=-1)

		pred_tick = torch.argmax(pred, dim=1)
		correct = pred_tick == target
		acc = correct[mask].float().mean()

		return loss, {
			'acc': acc.item(),
		}


class VocalAnalyzerNotationJointer (nn.Module):
	def __init__ (self, encoder_dim=128, d_time=128, n_tick=100, midi_encoder='MidiEncoder3', notationArgs={}, num_down_layers=2, n_notation_enc_layers=4, n_notation_dec_layers=3, **args):
		super().__init__()

		d_inner = encoder_dim << num_down_layers

		self.vocalEncoder = VocalEncoder(encoder_dim)
		if midi_encoder == 'MidiEncoder3':
			self.midiEncoder = MidiEncoder3(d_inner, d_time=d_time, n_tick=n_tick)
		else:
			self.midiEncoder = MidiEncoder4(d_inner, d_time=d_time)
		self.notationEncoder = Encoder(n_notation_enc_layers, d_model=d_inner, **notationArgs)
		self.notationDecoder = Decoder(n_notation_dec_layers, d_model=d_inner, **notationArgs)
		self.notationEmbed = Linear(d_inner, encoder_dim, bias=False)
		self.encoder = ConformerEncoderDecoderU(encoder_dim=encoder_dim, decoder=self.notationDecoder, num_down_layers=num_down_layers, **args)
		self.jointer = Jointer2()


	def forward (self, pitch, gain, midi_pitch, midi_tick):
		midi = self.midiEncoder(midi_pitch, midi_tick)
		notation = self.notationEncoder(midi)

		vocal = self.vocalEncoder(pitch, gain)
		vocal = self.encoder(vocal, notation)

		notation = self.notationEmbed(notation)
		matching = self.jointer(vocal, notation)
		matching = matching.permute(0, 2, 1)	# (n, n_note, n_frame)

		return matching


class VocalAnalyzerNotationJointerLoss (nn.Module):
	def __init__ (self, n_class=100, output_field='nionf', tick_filed='midi_rtick', **args):
		super().__init__()

		self.deducer = VocalAnalyzerNotationJointer(n_class=n_class, **args)
		self.output_field = output_field
		self.tick_filed = tick_filed
		self.n_class = n_class


	def forward (self, batch):
		midi_tick = torch.div(batch[self.tick_filed], TICK_ROUND_UNIT, rounding_mode='floor')

		target = batch[self.output_field]
		pred = self.deducer(batch['pitch'], batch['gain'], batch['midi_pitch'], midi_tick)

		mask = batch['mask']
		pred = pred * mask.float()

		loss = F.cross_entropy(pred, target, ignore_index=-1)

		pred_index = torch.argmax(pred, dim=1)
		correct = pred_index == target
		acc = correct[mask[:, 0, :]].float().mean()

		return loss, {
			'acc': acc.item(),
		}
