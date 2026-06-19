
import os
import typer
from typing_extensions import Annotated
import yaml
import re
from tqdm import tqdm



word_pattern = re.compile(r'\S+')

def main(paraff_path: Annotated[str, typer.Argument()], images_dir: Annotated[str, typer.Argument()]):
	print('Loading paraff...')
	original_lib = yaml.safe_load(open(paraff_path, 'r'))
	lib = {}
	for name, score in original_lib.items():
		normalized_name = name.replace('-', '_')
		lib[normalized_name] = score

	rows = []
	index = 0

	for root, dirs, files in os.walk(images_dir):
		for file in tqdm(files):
			segs = file.split('.')
			names = '.'.join(segs[1:-2]).split('_')
			mm = names[-1]
			name = '_'.join(names[:-1])
			#print(f'{file=}, {name=}, {mm=}')
			#print(f'{lib[name].keys()=}')
			sentence = lib[name][mm]

			words = word_pattern.findall(sentence)
			words = [word for word in words if not word.startswith('#')]
			sentence = ' '.join(words)

			#print(name, mm, sentence)
			rows.append([str(index), file, sentence])
			index += 1

	# dump rows as a csv file
	output_path = paraff_path.replace('.yaml', '-vl.csv')
	with open(output_path, 'w') as f:
		f.write('index,image,sentence\n')
		for row in rows:
			f.write(','.join(row) + '\n')

	print('Output:', output_path)
	print('Done.')


if __name__ == "__main__":
	typer.run(main)
