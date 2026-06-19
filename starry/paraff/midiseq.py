
import os
import yaml



TOKENS = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), '../../assets/midiseqVocab.yaml'), 'r'))
T2I = {t: i for i, t in enumerate(TOKENS)}

N_VOCAB = len(TOKENS)

ID_PEDAL0 = T2I['Sustain0']
