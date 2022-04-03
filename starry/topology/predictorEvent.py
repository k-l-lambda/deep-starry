
import os
import torch
import logging

from ..utils.predictor import Predictor
from .data.events import clusterFeatureToTensors
from .event_element import EventElementType, BeamType_values, StemDirection_values



BATCH_SIZE = int(os.environ.get('TOPOLOGY_PREDICTOR_BATCH_SIZE', '1'))


class EvtopoPredictor (Predictor):
	def __init__(self, config, batch_size=BATCH_SIZE, device='cpu', **_):
		super().__init__(batch_size=batch_size, device=device)

		self.loadModel(config)


	def predict(self, clusters):
		for i in range(0, len(clusters), self.batch_size):
			batch_clusters = clusters[i:i+self.batch_size]
			inputs = clusterFeatureToTensors(batch_clusters)

			# to device
			for key in inputs:
				inputs[key] = inputs[key].to(self.device)

			with torch.no_grad():
				rec, mat = self.model(inputs)

			for ii, cluster in enumerate(batch_clusters):
				result = {
					'index': cluster.get('index'),
					'matrixH': mat[ii].reshape((len(cluster['elements']) - 1, -1)).cpu().tolist(),
					'elements': [{'index': elem['index']} for elem in cluster['elements']],
				}

				# rectification
				divisions = torch.argmax(rec['division'][ii], dim=1).cpu().tolist()
				dots = torch.argmax(rec['dots'][ii], dim=1).cpu().tolist()
				beams = [BeamType_values[x] for x in torch.argmax(rec['beam'][ii], dim=1).cpu().tolist()]
				stemDirections = [StemDirection_values[x] for x in torch.argmax(rec['stemDirection'][ii], dim=1).cpu().tolist()]
				graces = (rec['grace'][ii] > 0.5).cpu().tolist()

				for ei, (elem, resultElem) in enumerate(zip(cluster['elements'], result['elements'])):
					if elem['type'] in [EventElementType.CHORD, EventElementType.REST]:
						if divisions[ei] != elem['division']:
							resultElem['division'] = divisions[ei]
						if dots[ei] != elem['dots']:
							resultElem['dots'] = dots[ei]
						if beams[ei] != elem['beam']:
							resultElem['beam'] = beams[ei]
						if stemDirections[ei] != elem['stemDirection']:
							resultElem['stemDirection'] = stemDirections[ei]
						if graces[ei] != elem['grace']:
							resultElem['grace'] = graces[ei]

						divisionVec = rec['division'][ii, ei]
						dotsVec = rec['dots'][ii, ei]
						if divisions[ei] != elem['division'] or divisionVec[divisions[ei]] < 0.9:
							resultElem['divisionVector'] = divisionVec.cpu().tolist()
						if dots[ei] != elem['dots'] or dotsVec[dots[ei]] < 0.9:
							resultElem['dotsVector'] = dotsVec.cpu().tolist()

						timeWarped = rec['timeWarped'][ii, ei].item()
						if timeWarped > 1e-3 or elem.get('timeWarped'):
							resultElem['timeWarped'] = timeWarped

						fullMeasure = rec['fullMeasure'][ii, ei].item()
						if fullMeasure > 1e-3 or elem.get('fullMeasure'):
							resultElem['fullMeasure'] = fullMeasure

						fake = rec['fake'][ii, ei].item()
						if fake > 1e-3 or elem.get('fake'):
							resultElem['fake'] = fake

						resultElem['fakeP'] = 1 - (1 - fake) * (1 - graces[ei]) * (1 - fullMeasure)

					resultElem['tick'] = rec['tick'][ii, ei].item()
					if elem['type'] == EventElementType.EOS:
						result['duration'] = rec['tick'][ii, ei].item()

				yield result
