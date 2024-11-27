
import yaml
import os



PROJECT_DIR = os.path.join(os.path.dirname(__file__), '../..')

SEMANTIC_TABLE = yaml.safe_load(open(os.path.join(PROJECT_DIR, './assets/timewiseSemantics.yaml'), 'r'))

SEMANTIC_MAX = len(SEMANTIC_TABLE)

TG_PAD = SEMANTIC_TABLE.index('_PAD')
TG_EOS = SEMANTIC_TABLE.index('_EOS')

STAFF_MAX = 3
