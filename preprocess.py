# from dataset import Contrastive_DA_Dataset
from utils.readers import bdek_reader
from utils.readers import airlines_reader
import torch
# from augmentor import back_translation_augmenter
from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer
import pandas as pd

if __name__=='__main__':
	# data = Contrastive_DA_Dataset()
	# domain = 'dvd'
	# data_reader = bdek_reader(domain, 'source')
	data_reader = airlines_reader('source')
	data = data_reader.read_data()
	labeled_data = data['labeled']
	unlabeled_data = data['unlabeled']

	# translater = back_translation_augmenter()
	# device = 
	model_en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model')
	model_de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model')

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	# if torch.cuda.device_count() > 1:
	# 	print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
	# 	model_en2de = torch.nn.DataParallel(model_en2de)
	# 	model_de2en = torch.nn.DataParallel(model_de2en)
	# else:
	# 	# line c=model.children.__next__() requires you use dataparallel, which is why it's ugly
	# 	print('no parallel may cause error')

	model_en2de = model_en2de.to(device)
	model_de2en = model_de2en.to(device)

	labeled_data['aug_text'] = [] 

	for i, text in enumerate(labeled_data['text']):
		print('labeled ', i)
		doc_list = sent_tokenize(text)
		# print(doc_list)
		# doc_list_trans = []
		
			
		doc_list_trans = model_de2en.translate(model_en2de.translate(doc_list[:16], beam=1), beam=1)
		labeled_data['aug_text'].append(' '.join(doc_list_trans))
		# trans_text = model_de2en.translate(model_en2de.translate(text))
		# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		# tokens = tokenizer.encode(trans_text)
		# print('tokens', len(tokens))
		# labeled_data['aug_text'].append(trans_text)
		# break


	labeled_data.pop('domain')
	labeled_df = pd.DataFrame(labeled_data)
	with open('data/airlines_backtranslation/labeled.csv', 'w') as f:
		labeled_df.to_csv(f)
	
	unlabeled_data['aug_text'] = []

	for i, text in enumerate(unlabeled_data['text']):
		print('unlabeled', i)
		# if i==11277 or i==16237:
		# 	unlabeled_data['aug_text'].append(text)
		# 	continue
		doc_list = sent_tokenize(text)
		# print(doc_list)
		# print(doc_list)
		# doc_list_trans = []
		# for sent in doc_list:
		doc_list_trans = model_de2en.translate(model_en2de.translate(doc_list[:16], beam=1), beam=1)
		unlabeled_data['aug_text'].append(' '.join(doc_list_trans))
		# break

	
	unlabeled_data.pop('domain')
	unlabeled_df = pd.DataFrame(unlabeled_data)

		

	with open('data/airlines_backtranslation/unlabeled.csv', 'w') as f:
		unlabeled_df.to_csv(f)	
