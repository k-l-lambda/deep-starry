
model_dict = None


def registerModels ():
	global model_dict

	from ..topology import models as tpm
	from ..vision import models as vm

	classes = [
		tpm.TransformJointer, tpm.TransformJointerLoss,
		vm.ScoreWidgets, vm.ScoreWidgetsMask, vm.ScoreWidgetsInspection,
		vm.ScoreRegression,
	]

	model_dict = dict([(c.__name__, c) for c in classes])



def loadModel (config):
	global model_dict
	if model_dict is None:
		registerModels()

	if config['type'] not in model_dict:
		raise RuntimeError("Model type %s not found" % config['type'])

	model_class = model_dict[config['type']]

	return model_class(**config['args'])
