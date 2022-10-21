
import re
from copy import deepcopy



def parseFilterStr (filterStr):
	filterStr = filterStr[1:] if filterStr.startswith('*') else filterStr
	phases, cycle = filterStr.split('/')
	captures = re.match(r'(\d+)\.\.(\d+)', phases)
	if captures:
		phases = list(range(int(captures[1]), int(captures[2]) + 1))
	else:
		phases = list(map(int, phases.split(',')))
	cycle = int(cycle)

	return phases, cycle


def deep_update (d, u):
	for k, v in u.items():
		if isinstance(v, dict):
			d[k] = deep_update(d.get(k, {}), v)
		else:
			d[k] = v
	return d


def mergeArgs (args, argv):
	return deep_update(deepcopy(args), argv) if argv else args
