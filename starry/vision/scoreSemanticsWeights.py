
import math



def normalWeights (N, P, magnitude=1):
	factor = math.sqrt(N * P) * magnitude

	return (N * factor, P * factor)


LOSS_WEIGHTS = {	# (negative_weight, positive_weight)
	'vline_Stem':                normalWeights(2.5, 1),
	'Flag3':                     normalWeights(1.4, 1),
	#'rect_Text':                 normalWeights(1, 1, 0.1),
	'GraceNotehead':             normalWeights(1, 2.5),
	'SignLine':                  normalWeights(1, 2.5),
	'SignInterval':              normalWeights(1, 2.5),
}
