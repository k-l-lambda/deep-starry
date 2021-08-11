
model_dict = None


def registerModels ():
	global model_dict
	model_dict = {}

	from ..topology import models as tpm
	from ..vision import score_widgets as sw

	classes = [
		tpm.TransformJointer, tpm.TransformJointerLoss,
		sw.ScoreWidgets, sw.ScoreWidgetsMask, sw.ScoreWidgetsInspection,
	]

	for c in classes:
		model_dict[c.__name__] = c



def loadModel (config):
	global model_dict
	if model_dict is None:
		registerModels()

	if config['type'] not in model_dict:
		raise RuntimeError("Model type %s not found" % config['type'])

	model_class = model_dict[config['type']]

	return model_class(**config['args'])
