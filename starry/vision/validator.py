
import torch
import numpy as np
import yaml

from ..utils.predictor import Predictor
from .images import MARGIN_DIVIDER, splicePieces
from . import contours
from .datasetViewer import scoreAnno, scoreAnnoChromatic, scoreAnnoGauge



class Validator (Predictor):
	def __init__(self, config, skip_perfect=False, chromatic=True, gauge_mode=False, splice=True):
		super().__init__()

		self.config = config
		self.loadModel(config)

		self.skip_perfect = skip_perfect
		self.chromatic = chromatic
		self.gauge_mode = gauge_mode
		self.splice = splice

		self.compounder = contours.Compounder(config)

	def run(self, dataset):
		unit_size = self.config['data.args.unit_size'] or 1

		with torch.no_grad():
			for batch, (feature, target) in enumerate(dataset):
				#print(f'batch:{batch} feature:{feature.shape} target:{target.shape}')
				pred = self.model(feature)

				pred_compound = self.compounder.compound(pred)
				target_compound = self.compounder.compound(target)

				hms = pred_compound.cpu().numpy()
				img = feature.cpu().numpy()
				tar = target_compound.cpu().numpy()

				if self.gauge_mode:
					for i in range(img.shape[0]):
						image = img[i].reshape(img.shape[2:])
						scoreAnnoGauge(np.uint8(image * 255), tar[i], hms[i])

					continue

				#print(f'out shape:{hms.shape} img shape:{img.shape} target:{tar.shape}')
				if self.splice:
					heatmap = np.uint8(splicePieces(hms, MARGIN_DIVIDER) * 255)
					image = splicePieces(img, MARGIN_DIVIDER)[0]
					target = splicePieces(tar, MARGIN_DIVIDER)
				else:
					heatmap = np.uint8(hms[0] * 255)
					image = img[0][0]
					target = tar[0]

				image = np.clip(image, 0., 1.)
				image = np.uint8(image * 255)
				#print(f'heatmap shape:{heatmap.shape} image shape:{image.shape} target:{target.shape}')

				target = np.uint8(target * 255)

				if self.skip_perfect:
					perfect = True

					fake_negative, fake_positive, true_negative, true_positive = 0, 0, 0, 0

					for c, label in enumerate(self.compounder.labels):
						tar = target[c]
						pred = heatmap[c]

						points = contours.countHeatmaps(tar, pred, label, unit_size = unit_size)
						true_count = len([p for p in points if p['value'] > 0])
						if true_count > 0:
							confidence, error, precision, feasibility, fake_neg, fake_pos, true_neg, true_pos = contours.statPoints(points, true_count, 1, 1)

							fake_negative += fake_neg
							fake_positive += fake_pos
							true_negative += true_neg
							true_positive += true_pos

							#print('label:', label, error, precision, feasibility)
							if error > 0:
								perfect = False
								print('\nerror:', label, error, true_count)

								issue_points = list(filter(
									lambda p: (p['value'] < 0 and p['confidence'] >= confidence) or (p['value'] > 0 and p['confidence'] <= 0),
									points))
								for p in issue_points:
									p['x'] *= unit_size
									p['y'] = p['y'] * unit_size + pred.shape[0] // 2
								print(yaml.dump(issue_points))

								break

					if perfect:
						print('.', end='', flush=True)	# . stand for skipping a perfect sample
						continue
					else:
						# confusion matrix
						print(f'confusion:\n  {true_positive}\t{fake_positive}\n  {fake_negative}\t{true_negative}')

				if self.chromatic:
					scoreAnnoChromatic(image, target, heatmap / 255.)
				else:
					target = np.moveaxis(target, 0, 2)
					scoreAnno(heatmap, image, target=target, labels=self.compounder.labels)
