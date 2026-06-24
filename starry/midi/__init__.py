'''MIDI-text modality for deep-starry.

Tokenizes the line-per-event MIDI text produced by music-widgets' MidiText codec
(SkyTNT-style event patches; see tokenizer.py) and packs it into NotaGen/bGPT
patch artifacts for the shared two-level decoder (starry.bgpt).

  - tokenizer.py : MidiTokenizer (one event = one fixed-size patch)
  - data/        : MidiPatchifier (pack .txt -> .pt) + MidiPatchy (Dataset feeder)
'''

from .tokenizer import MidiTokenizer
