'''Shared base for run-local VOCABULARY assets.

A vocabulary is the common case of a run-local asset (see `registry.ASSETS` and
`Configuration._preprocess_assets`): a file published into the run directory once, reloaded verbatim
on resume, and pointed at by `data.args.vocab_path` + `model.args.vocab_path`. The parts that differ
per modality are only *what* gets published and *what* the file implies about the model, so those are
the two hooks; everything else — reference validation, the relative/absolute invariant, the derived-arg
injection — lives here so the two implementations cannot drift apart.

Concrete builders: `starry.midi.data.seq2CondPachifier.Midiseq2Vocab` (copies the authoritative asset)
and `starry.midi.data.unifiedSeq2Tokenizer.UnifiedSeq2Vocab` (synthesizes a merged mapping).
'''

import os
import tempfile


def publish_atomically (path, write):
	'''Run `write(file)` into a temporary sibling, then rename it over `path`.

	A half-written vocabulary that a later resume would happily load is the failure this exists to
	prevent: the rename is atomic, so the run directory only ever holds a complete file.
	'''
	directory = os.path.dirname(os.path.abspath(path))
	fd, temporary = tempfile.mkstemp(prefix='.vocab-asset-', suffix='.tmp', dir=directory)
	try:
		with os.fdopen(fd, 'w', encoding='utf-8') as f:
			write(f)
			f.flush()
			os.fsync(f.fileno())
		os.replace(temporary, path)
	finally:
		if os.path.exists(temporary):
			os.remove(temporary)


class VocabAsset:
	'''Base builder implementing the ASSETS create/resume/resolve/relativize contract.

	Subclasses set `FILENAME` and implement:

		publish(path, args)   write the file into the run directory (use `publish_atomically`)
		describe(path)        load it and return the model args it implies, e.g.
		                      {'vocab_size': 838, 'eos_id': 2}. MUST validate, since this is the
		                      only thing standing between a corrupted pin and a silently
		                      mis-numbered model.
	'''

	FILENAME = None

	@classmethod
	def create (cls, config, args):
		path = config.localPath(cls.FILENAME)
		cls.publish(path, args or {})
		cls._inject(config, cls.describe(path), cls.FILENAME)
		return cls.FILENAME

	@classmethod
	def resume (cls, config, name):
		'''Load the pinned file named by the run state. Never republishes or regenerates.'''
		if not isinstance(name, str) or not name or os.path.isabs(name) or os.path.dirname(name):
			raise ValueError(f'{cls.__name__}: run state has no valid run-local reference: {name!r}')
		cls._inject(config, cls.describe(config.localPath(name)), name)

	@staticmethod
	def resolve (config, name):
		'''Stored relative reference -> absolute in-memory path. Called after every load and save.'''
		path = config.localPath(name)
		config['data.args.vocab_path'] = path
		config['model.args.vocab_path'] = path

	@staticmethod
	def relativize (data, name):
		'''The inverse of `resolve`, applied to a COPY of the state on its way to disk, so a run
		directory stays relocatable.'''
		data['data']['args']['vocab_path'] = name
		data['model']['args']['vocab_path'] = name

	@classmethod
	def _inject (cls, config, derived, name):
		'''Write the derived model args and the relocatable references into the config.

		A value the config states explicitly is VALIDATED rather than overwritten, so a hand-written
		number that no longer matches the vocabulary fails loudly instead of building a model whose
		embedding rows mean something else.
		'''
		if config['model'] is None:
			raise ValueError(f'{cls.__name__} requires a model section to inject {sorted(derived)}')
		model_args = config.data['model'].setdefault('args', {})
		data_args = config.data.setdefault('data', {}).setdefault('args', {})
		for key, value in derived.items():
			configured = model_args.get(key)
			if configured is not None and int(configured) != value:
				raise ValueError(f'{cls.__name__}: {key} must be {value}, got {configured}')
			model_args[key] = value
		# Persist relocatable references. They are resolved only after save, never written back.
		data_args['vocab_path'] = name
		model_args['vocab_path'] = name
