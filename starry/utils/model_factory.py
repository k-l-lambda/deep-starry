
import torch



model_dict = None


def registerModels ():
	global model_dict

	#from ..topology.models import jointers as tj
	#from ..topology.models import rectifyJointer as tr
	from ..melody.models import TestEncoder, MatchJointerRaw, MatchJointer1, MatchJointer1Loss, MatchJointer2, MatchJointer2Loss, MatchJointer3, MatchJointer3Loss

	classes = [
		TestEncoder, MatchJointerRaw, MatchJointer1, MatchJointer1Loss, MatchJointer2, MatchJointer2Loss, MatchJointer3, MatchJointer3Loss
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



def loadModelAndWeights (config, checkpoint_name=None, device='cpu'):
	model = loadModel(config['model'])

	checkpoint = {}
	if checkpoint_name is not None:
		checkpoint = torch.load(config.localPath(checkpoint_name), map_location=device)
		model.load_state_dict(checkpoint['model'])

	return model, checkpoint
