
model_dict = None


def registerModels ():
	global model_dict

	from ..topology import models as tpm
	from ..vision import models as vm

	classes = [
		tpm.TransformJointer, tpm.TransformJointerLoss,
		tpm.TransformJointerH, tpm.TransformJointerHLoss,
		vm.ScoreWidgets, vm.ScoreWidgetsMask, vm.ScoreWidgetsInspection,
		vm.ScoreRegression,
		vm.ScoreResidue, vm.ScoreResidueInspection,
	]

	model_dict = dict([(c.__name__, c) for c in classes])



def loadModel (config, postfix = ''):
	global model_dict
	if model_dict is None:
		registerModels()

	model_type = config['type'] + postfix

	if model_type not in model_dict:
		raise RuntimeError("Model type %s not found" % model_type)

	model_class = model_dict[model_type]

	return model_class(**config['args'])