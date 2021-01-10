#modified by : Sayantan Basu

import csv
import os
import argparse
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_cosine_with_hard_restarts_schedule_with_warmup
import warnings
import numpy as np
warnings.filterwarnings('ignore')
# from apex import amp

class MyDataset(Dataset):
	def __init__(self, data_file_name, data_dir='.data/'):
		super().__init__()

		data_path = os.path.join(data_file_name)

		self.data_list = []
		self.end_of_text_token = " <|endoftext|> "

		with open(data_path) as csv_file:
			csv_reader = csv.reader(csv_file, delimiter='\t')

			for row in csv_reader:
				data_str = f"{row[0]}: {row[1]}{self.end_of_text_token}"
				self.data_list.append(data_str)

	def __len__(self):
		return len(self.data_list)

	def __getitem__(self, item):
		return self.data_list[item]

def get_data_loader(data_file_name):
	dataset = MyDataset(data_file_name)
	data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
	return data_loader

def train(epochs, data_loader, batch_size, tokenizer, model, device, loader_test,model_name):
	batch_counter = 0
	sum_loss = 0.0
	early_stop = 5
	start_flag = 0
	best_p = 99999999999999
	count = 0

	for epoch in range(epochs):
		print (f'Running {epoch+1} epoch')

		for idx, txt in enumerate(data_loader):
			# print(txt)
			# print(txt[0])
			# print(len(txt))

			txt = torch.tensor(tokenizer.encode(txt[0]))
			# print(tokenizer.encode('<e10>'))
			# print(txt)
			# print(len(txt))
			# exit()
			txt = txt.unsqueeze(0).to(device)
			outputs = model(txt, labels=txt)
			loss, _ = outputs[:2]
			# with amp.scale_loss(loss, optimizer) as scaled_loss:
			# 	scaled_loss.backward()
			loss.backward()
			sum_loss += loss.data

			if idx%batch_size==0:
				batch_counter += 1
				optimizer.step()
				scheduler.step()
				optimizer.zero_grad()
				model.zero_grad()

			if batch_counter == 10:
				loss_test_sum = 0
				with torch.no_grad():
					for idx, txt in enumerate(loader_test):
						txt = torch.tensor(tokenizer.encode(txt[0]))
						txt = txt.unsqueeze(0).to(device)
						outputs = model(txt, labels=txt)
						loss_test, _ = outputs[:2]
						loss_test_sum += loss_test.data
				perplexity = np.exp(loss_test_sum)
				print("epoch:{}, test loss:{}".format(epoch, loss_test_sum))
				if loss_test_sum<best_p:
					best_p = loss_test_sum
					if count ==5:
						print(r'Get the new best model of Perplexity: {}, and save it'.format(
							perplexity))
						save_model(model,model_name+str(int(loss_test_sum)))
						count = 0
					count += 1

				if start_flag == 0:
					pri_loss = sum_loss
				if pri_loss > sum_loss:
					early_stop -= 1
				if pri_loss < sum_loss:
					early_stop = 5
					pri_loss = sum_loss
				if early_stop == 0:
					break
				print("sum_loss:{}".format(sum_loss))
				sum_loss = 0
				batch_counter = 0

	return model

def save_model(model, name):
	"""
	Summary:
		Saving model to the Disk
	Parameters:
		model: Trained model object
		name: Name of the model to be saved
	"""
	print ("Saving {} model to Disk".format(name))
	torch.save(model.state_dict(), f"{name}.pt")
	return

def load_models():
	"""
	Summary:
		Loading Pre-trained model
	"""
	print ('Loading/Downloading GPT-2 Model')
	tokenizer = GPT2Tokenizer.from_pretrained('healx/gpt-2-pubmed-large')
	model = GPT2LMHeadModel.from_pretrained('healx/gpt-2-pubmed-large')
	return tokenizer, model

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Arguments for training Text Augmentation model')

	parser.add_argument('--epoch', default= 10,type=int, action='store', help='Number of epochs to run')
	parser.add_argument('--warmup', default=300, type=int, action='store', help='Number of warmup steps to run')
	parser.add_argument('--model_name', default='overallmodel', type=str, action='store', help='Name of the model file')
	parser.add_argument('--data_file', default='train_genspec.tsv', type=str, action='store', help='Name of the data file')
	parser.add_argument('--batch', type=int, default=16, action='store', help='Batch size')
	parser.add_argument('--learning_rate', default=3e-5, type=float, action='store', help='Learning rate for the model')
	parser.add_argument('--max_len', default=300, type=int, action='store', help='Maximum length of sequence')
	args = parser.parse_args()

	BATCH_SIZE = args.batch
	EPOCHS = args.epoch
	LEARNING_RATE = args.learning_rate
	WARMUP_STEPS = args.warmup
	MAX_SEQ_LEN = args.max_len
	MODEL_NAME = args.model_name
	DATA_FILE = args.data_file

	TOKENIZER, MODEL = load_models()
	data_files = ["classfied_data/advise_data.tsv", "classfied_data/effect_data.tsv",
				"classfied_data/int_data.tsv", "classfied_data/mechanism_data.tsv"]
	test_files = ["data/test_advise.tsv", "data/test_effect.tsv",
				"data/test_int.tsv", "data/test_mechanism.tsv"]
	model_names = ['advise_model.pt','effect_model.pt','int_model.pt','mechanism_model.pt']
	for i, x in enumerate(data_files):
		# if i==0 :
		# 	continue

		print("begin:{}".format(x))
		LOADER = get_data_loader(x)
		LOADER_TEST = get_data_loader(test_files[i])

		DEVICE = 'cpu'
		if torch.cuda.is_available():
			DEVICE = 'cuda'
			DEVICE = 'cpu'

		model = MODEL.to(DEVICE)
		model.train()
		optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
		# model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
		scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=-1)

		model = train(EPOCHS, LOADER, BATCH_SIZE, TOKENIZER, MODEL, DEVICE, LOADER_TEST,model_names[i])
		save_model(model, model_names[i])

	# LOADER = get_data_loader(DATA_FILE)
	#
	# DEVICE = 'cpu'
	# if torch.cuda.is_available():
	# 	DEVICE = 'cuda'
	# 	DEVICE = 'cpu'
	#
	# model = MODEL.to(DEVICE)
	# model.train()
	# optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
	# # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
	# scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=-1)
	#
	# model = train(EPOCHS, LOADER, BATCH_SIZE, TOKENIZER, MODEL, DEVICE)
	# save_model(model, MODEL_NAME)
