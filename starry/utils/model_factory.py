
import torch
import logging



model_dict = None


def registerModels ():
	global model_dict

	from ..topology.models import jointers as tj
	from ..topology.models import rectifyJointer as tr
	from ..topology.models import rectifyJointer2 as tr2
	from ..topology.models import beadPicker as tb
	from ..vision import models as vm
	from ..paraff import models as pm

	classes = [
		tj.TransformJointer, tj.TransformJointerLoss,
		tj.TransformJointerH, tj.TransformJointerHLoss,
		tj.TransformJointerHV, tj.TransformJointerHVLoss,
		tj.TransformJointerH_ED, tj.TransformJointerH_EDLoss,
		tj.TransformJointerHV_EDD, tj.TransformJointerHV_EDDLoss,
		tj.TransformSieveJointerH, tj.TransformSieveJointerHLoss,
		tj.TransformSieveJointerHV, tj.TransformSieveJointerHVLoss,
		tr.RectifySieveJointer, tr.RectifySieveJointerLoss,
		tr2.RectifySieveJointer2, tr2.RectifySieveJointer2Loss,
		tb.BeadPicker, tb.BeadPickerLoss, tb.BeadPickerOnnx,
		vm.ScoreWidgets, vm.ScoreWidgetsInspection, vm.ScoreWidgetsLoss,
		vm.ScoreWidgetsMask, vm.ScoreWidgetsMaskLoss,
		vm.ScoreRegression, vm.ScoreRegressionLoss,
		vm.ScoreResidue, vm.ScoreResidueInspection,
		vm.ScoreResidueU, vm.ScoreResidueUInspection, vm.ScoreResidueULoss,
		vm.ScoreSemanticValue, vm.ScoreSemanticValueLoss,
		vm.GlyphRecognizer, vm.GlyphRecognizerLoss,
		pm.TokenGen, pm.TokenGenLoss,
		pm.SeqvaeLoss, pm.SeqvaeEncoderJit,
		pm.SparseAE, pm.SparseAELoss,
		pm.SeqShareVAE, pm.SeqShareVAELoss, pm.SeqShareVAEJitEnc, pm.SeqShareVAEJitDec,
		pm.PhaseGen, pm.PhaseGenLoss, pm.PhaseGenDecoder, pm.PhaseGenDecoderLora,
		pm.SeqDecoderBase, pm.SeqDecoderBaseLoss,
		pm.GraphParaffEncoder, pm.GraphParaffEncoderLoss, pm.GraphParaffEncoderTail, pm.GraphParaffEncoderDecoder,
		pm.GraphParaffTransformer, pm.GraphParaffTransformerLoss,
	]

	model_dict = dict([(c.__name__, c) for c in classes])



def loadModel (config, postfix=''):
	global model_dict
	if model_dict is None:
		registerModels()

	model_type = config['type'] + postfix

	if model_type not in model_dict:
		raise RuntimeError("Model type %s not found" % model_type)

	model_class = model_dict[model_type]

	return model_class(**config['args'])



def loadModelAndWeights (config, checkpoint_name=None, device='cpu', postfix=''):
	model = loadModel(config['model'], postfix=postfix)

	checkpoint = {}
	if checkpoint_name is not None:
		checkpoint = torch.load(config.localPath(checkpoint_name), map_location=device)
		model.load_state_dict(checkpoint['model'])
		logging.info('Weights file loaded: %s', config.localPath(checkpoint_name))

	return model, checkpoint
