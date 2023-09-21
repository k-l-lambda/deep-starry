
import os
import struct
import torch
from tqdm import tqdm
from transformers import EncodecModel, AutoProcessor



ENCODEC_PRETRAINED = os.getenv('ENCODEC_PRETRAINED')
ENCODEC_BANDWIDTH = int(os.getenv('ENCODEC_BANDWIDTH'))
ENCODEC_BATCHSIZE = int(os.getenv('ENCODEC_BATCHSIZE'))

model = EncodecModel.from_pretrained(ENCODEC_PRETRAINED)
processor = AutoProcessor.from_pretrained(ENCODEC_PRETRAINED)

sampling_rate = processor.sampling_rate


def modelEncode (audio, bandwidth, sr):
	with torch.no_grad():
		inputs = processor(raw_audio=audio, sampling_rate=sr, return_tensors='pt')
		encoder_outputs = model.encode(inputs['input_values'], inputs['padding_mask'], bandwidth=bandwidth)

		return encoder_outputs.audio_codes, encoder_outputs.audio_scales, inputs['padding_mask']


def modelDecode (codes, scales, mask=None):
	with torch.no_grad():
		return model.decode(codes, scales, mask)[0]


def packInt10 (source: torch.Tensor) -> bytes:
	buf64 = (source.view(-1, 4) << torch.tensor([[0, 10, 20, 30]], dtype=torch.int64)).sum(dim=-1, keepdim=True)
	buf8 = (buf64 >> torch.tensor([0, 8, 16, 24, 32]) % 0x100).type(torch.uint8).flatten()

	return struct.pack('B' * buf8.shape[0], * buf8)


def unpackInt10 (buf: bytes, n_codebook) -> torch.Tensor:
	assert len(buf) % 5 == 0, f'invalid buffer length, must be multiples of 5: {len(buf)}'

	buf8 = torch.tensor(struct.unpack('B' * len(buf), buf), dtype=torch.int64)
	buf64 = (buf8.view(-1, 5) << torch.tensor([[0, 8, 16, 24, 32]], dtype=torch.int64)).sum(dim=-1, keepdim=True)
	buf10 = (buf64 >> torch.tensor([0, 10, 20, 30], dtype=torch.int64)).flatten() % (2 ** 10)

	return buf10.view(1, 1, n_codebook, -1)


def encode (audio, bandwidth=ENCODEC_BANDWIDTH, sr=sampling_rate):
	codes = []
	samples = ENCODEC_BATCHSIZE * 320
	for t in tqdm(range(0, len(audio), samples), leave=False):
		seg = audio[t:t + samples]
		c, scales, mask = modelEncode(seg, bandwidth, sr=sr)
		codes.append(c)

	codes = torch.cat(codes, dim=-1)
	buf = packInt10(codes)
	n_codebook = codes.shape[-2]

	return buf, n_codebook


def decode (buffer: bytes, n_codebook):
	codes = unpackInt10(buffer, n_codebook)

	audio = []
	for f in tqdm(range(0, codes.shape[-1], ENCODEC_BATCHSIZE), leave=False):
		c = codes[:, :, :, f:f + ENCODEC_BATCHSIZE]
		audio.append(modelDecode(c, [None]))

	return torch.cat(audio, dim=-1)[0, 0]
