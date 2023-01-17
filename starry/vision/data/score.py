
import os
import re

from ...utils.parsers import parseFilterStr
from .imageReader import ImageReader



STAFF = '_Staff'
GRAPH = '_Graph'
MASK = '_Mask'
PAGE = '_Page'
PAGE_LAYOUT = '_PageLayout'
FAULT = '_Fault'
FAULT_TARGET = '_FaultTarget'


def makeReader (root):
	name, ext = os.path.splitext(root)
	is_zip = ext == '.zip'
	nomalized_root = name if is_zip else root
	reader_url = ('zip://' + root) if is_zip else root

	return ImageReader(reader_url), nomalized_root


''' filterStr examples:
		0/1				ordered all
		*0/1			random all
		1/10			the second part of tenth
		1,3,5,7,9/10	the odd parts of tenth
		0..9/10			the first 9 parts of tenth
'''
def listAllScoreNames (reader, filterStr, dir=STAFF):
	all_names = list(map(lambda name: os.path.splitext(name)[0], reader.listFiles(dir)))

	titles = list(set(map(lambda name: name.split('-')[0], all_names)))
	titles.sort()
	titleIndices = dict(map(lambda pair: pair[::-1], enumerate(titles)))
	#print('titleIndices:', titleIndices)

	if filterStr is None:
		return all_names

	phases, cycle = parseFilterStr(filterStr)

	return [n for n in all_names if (titleIndices[n.split('-')[0]] % cycle) in phases]
