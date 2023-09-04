
import yaml



SEMANTIC_TABLE = yaml.safe_load(open('./assets/timewiseSemantics.yaml', 'r'))

SEMANTIC_MAX = len(SEMANTIC_TABLE)

TG_PAD = SEMANTIC_TABLE.index('_PAD')
TG_EOS = SEMANTIC_TABLE.index('_EOS')

STAFF_MAX = 3
