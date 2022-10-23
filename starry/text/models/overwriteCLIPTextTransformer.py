
from typing import Any, Optional, Tuple, Union
import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling
#from transformers.models.clip.modeling_clip import CLIPTextTransformer



def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
	if len(mask.shape) == 2:
		# Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
		bsz, src_len = mask.size()
		tgt_len = tgt_len if tgt_len is not None else src_len

		expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
	elif len(mask.shape) == 3:
		# Expands attention_mask from `[bsz, tgt_seq_len, src_seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
		expanded_mask = mask[:, None, :, :].to(dtype)

	inverted_mask = 1.0 - expanded_mask

	return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def CLIPTextTransformer_forward(
	self,
	input_ids: Optional[torch.Tensor] = None,
	attention_mask: Optional[torch.Tensor] = None,
	position_ids: Optional[torch.Tensor] = None,
	output_attentions: Optional[bool] = None,
	output_hidden_states: Optional[bool] = None,
	return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPooling]:
	output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
	output_hidden_states = (
		output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
	)
	return_dict = return_dict if return_dict is not None else self.config.use_return_dict

	if input_ids is None:
		raise ValueError("You have to specify either input_ids")

	input_shape = input_ids.size()
	input_ids = input_ids.view(-1, input_shape[-1])

	hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

	bsz, seq_len = input_shape
	# CLIP's text model uses causal mask, prepare it here.
	# https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
	causal_attention_mask = self._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
		hidden_states.device
	)
	# expand attention_mask
	if attention_mask is not None:
		# [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
		attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

	encoder_outputs = self.encoder(
		inputs_embeds=hidden_states,
		attention_mask=attention_mask,
		causal_attention_mask=causal_attention_mask,
		output_attentions=output_attentions,
		output_hidden_states=output_hidden_states,
		return_dict=return_dict,
	)

	last_hidden_state = encoder_outputs[0]
	last_hidden_state = self.final_layer_norm(last_hidden_state)

	# text_embeds.shape = [batch_size, sequence_length, transformer.width]
	# take features from the eot embedding (eot_token is the highest number in each sequence)
	# casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
	pooled_output = last_hidden_state[
		torch.arange(last_hidden_state.shape[0]), input_ids.to(torch.int).argmax(dim=-1)
	]

	if not return_dict:
		return (last_hidden_state, pooled_output) + encoder_outputs[1:]

	return BaseModelOutputWithPooling(
		last_hidden_state=last_hidden_state,
		pooler_output=pooled_output,
		hidden_states=encoder_outputs.hidden_states,
		attentions=encoder_outputs.attentions,
	)
