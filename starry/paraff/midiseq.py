
import yaml



TOKENS = yaml.safe_load(open('./assets/midiseqVocab.yaml', 'r'))
T2I = {t: i for i, t in enumerate(TOKENS)}

N_VOCAB = len(TOKENS)
