
import dill as pickle
import argparse

from starry.score_connection.data import Dataset
from starry.score_connection.trainer import Trainer



def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--data', type=str)
	parser.add_argument('-b', '--batch_size', type=int, default=16)
	parser.add_argument('-dv', '--device', type=str, default='cuda')
	parser.add_argument('-sv', '--save_mode', type=str, default='best')
	parser.add_argument('-e', '--epoch', type=int, default=400)
	parser.add_argument('-lr', '--lr_mul', type=float, default=2.0)
	parser.add_argument('-warm', '--n_warmup_steps', type=int, default=4000)

	args = parser.parse_args()
	args.output_dir = './output'

	data = pickle.load(open(args.data, 'rb'))
	args.d_model = data['d_word']

	train_data, val_data = Dataset.loadPackage(data, batch_size=args.batch_size, device=args.device)

	trainer = Trainer(args)
	trainer.train(train_data, val_data)



if __name__ == '__main__':
	main()
