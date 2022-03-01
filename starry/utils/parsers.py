
import re



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
