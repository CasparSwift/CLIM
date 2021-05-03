import os, numpy as np
from os.path import join
import random
# import scipy.sparse as sp
# from gensim.corpora import Dictionary as gensim_dico
# from brain.knowgraph import KnowledgeGraph
# from brain.config import *
import operator
import re
import xml.etree.ElementTree as ET

from utils.utils import pollute_data

import torch
import pandas as pd
import csv



class airlines_backtranslation_reader(object):
    def __init__(self, source_or_target):
        '''
        class for read raw bdek data from disk; 
        domain_name in 'books', 'dvd', 'kitchen', 'electronics'; 
        source_or_target in 'source' or 'target'

        pass an obj of this class to a da_dataset object defined in ../dataset.py
        '''
        self.text_paths = {
                    'labeled':join('data/airlines_backtranslation/labeled.csv'),\
                    'unlabeled':join('data/airlines_backtranslation/unlabeled.csv'),\
                    }

        assert source_or_target=='source' or source_or_target=='target'
        self.domain_label = int(source_or_target=='target')

    def get_dataset(self, file_path):
        '''
        extract texts from xml format file; see data/books/positive.parsed for an instinct of the format
        return a list of sentences, where each sentence is a list of words (may contain multiple lines)
        '''
        table = pd.read_csv(file_path)
        col = table.columns
        data = {'text':[str(t) for t in table['text']]}
        data['text_bt'] = [str(t) for t in table['text']]
        if 'label' in col:
            data['label'] = list(table['label'])
        return data

    def read_data(self):
        '''
        major read data procedure; called from da_dataset
        '''
        # labeled_data = {}
        # unlabeled_data = {}
        labeled_data = self.get_dataset(self.text_paths['labeled'])
        unlabeled_data = self.get_dataset(self.text_paths['unlabeled'])
        # positive_label = [1]*len(positive_text)
        # negative_label = [0]*len(negative_text)

        # labeled_data['text'] = positive_text + negative_text
        # labeled_data['label'] = positive_label + negative_label
        labeled_data['domain'] = [self.domain_label] * len(labeled_data['text'])
        # labeled_data['graph'] = np.load(open(self.graph_feature_paths['labeled'], 'rb'), allow_pickle=True)

        # unlabeled_data['text'] = self.get_dataset(self.text_paths['unlabeled'])
        unlabeled_data['domain'] = [self.domain_label] * len(unlabeled_data['text'])
        # unlabeled_data['graph'] = np.load(open(self.graph_feature_paths['unlabeled'], 'rb'), allow_pickle=True)

        return {'labeled':labeled_data, 'unlabeled':unlabeled_data}


class airlines_reader(object):
    def __init__(self, source_or_target):
        '''
        class for read raw bdek data from disk; 
        domain_name in 'books', 'dvd', 'kitchen', 'electronics'; 
        source_or_target in 'source' or 'target'

        pass an obj of this class to a da_dataset object defined in ../dataset.py
        '''
        self.text_paths = {
                    'labeled':join('data/airlines/labeled.csv'),\
                    'unlabeled':join('data/airlines/unlabeled.csv'),\
                    }

        assert source_or_target=='source' or source_or_target=='target'
        self.domain_label = int(source_or_target=='target')

    def read_data(self):
        '''
        major read data procedure; called from da_dataset
        '''
        # labeled_data = {}
        # unlabeled_data = {}
        labeled_data = self.get_dataset(self.text_paths['labeled'])
        unlabeled_data = self.get_dataset(self.text_paths['unlabeled'])
        # positive_label = [1]*len(positive_text)
        # negative_label = [0]*len(negative_text)

        # labeled_data['text'] = positive_text + negative_text
        # labeled_data['label'] = positive_label + negative_label
        labeled_data['domain'] = [self.domain_label] * len(labeled_data['text'])
        # labeled_data['graph'] = np.load(open(self.graph_feature_paths['labeled'], 'rb'), allow_pickle=True)

        # unlabeled_data['text'] = self.get_dataset(self.text_paths['unlabeled'])
        unlabeled_data['domain'] = [self.domain_label] * len(unlabeled_data['text'])
        # unlabeled_data['graph'] = np.load(open(self.graph_feature_paths['unlabeled'], 'rb'), allow_pickle=True)

        return {'labeled':labeled_data, 'unlabeled':unlabeled_data}

    def get_dataset(self, file_path):
        '''
        extract texts from xml format file; see data/books/positive.parsed for an instinct of the format
        return a list of sentences, where each sentence is a list of words (may contain multiple lines)
        '''
        table = pd.read_csv(file_path)
        col = table.columns
        data = {'text':[str(t) for t in table['text']]}
        if 'label' in col:
            data['label'] = list(table['label'])
        return data



class bdek_backtranslation_reader(object):
    def __init__(self, domain_name, source_or_target):
        '''
        class for read raw bdek data from disk; 
        domain_name in 'books', 'dvd', 'kitchen', 'electronics'; 
        source_or_target in 'source' or 'target'

        pass an obj of this class to a da_dataset object defined in ../dataset.py
        '''
        self.text_paths = {
                    'labeled':join('data/bdek_backtranslation/{}/labeled.csv').format(domain_name),\
                    'unlabeled':join('data/bdek_backtranslation/{}/unlabeled.csv').format(domain_name),\
                    }

        assert source_or_target=='source' or source_or_target=='target'
        self.domain_label = int(source_or_target=='target')

    def read_data(self):
        '''
        major read data procedure; called from da_dataset
        '''
        # labeled_data = {}
        # unlabeled_data = {}
        labeled_data = self.get_dataset(self.text_paths['labeled'])
        unlabeled_data = self.get_dataset(self.text_paths['unlabeled'])
        # positive_label = [1]*len(positive_text)
        # negative_label = [0]*len(negative_text)

        # labeled_data['text'] = positive_text + negative_text
        # labeled_data['label'] = positive_label + negative_label
        labeled_data['domain'] = [self.domain_label] * len(labeled_data['text'])
        # labeled_data['graph'] = np.load(open(self.graph_feature_paths['labeled'], 'rb'), allow_pickle=True)

        # unlabeled_data['text'] = self.get_dataset(self.text_paths['unlabeled'])
        unlabeled_data['domain'] = [self.domain_label] * len(unlabeled_data['text'])
        # unlabeled_data['graph'] = np.load(open(self.graph_feature_paths['unlabeled'], 'rb'), allow_pickle=True)

        return {'labeled':labeled_data, 'unlabeled':unlabeled_data}

    def get_dataset(self, file_path):
        '''
        extract texts from xml format file; see data/books/positive.parsed for an instinct of the format
        return a list of sentences, where each sentence is a list of words (may contain multiple lines)
        '''
        table = pd.read_csv(file_path)
        col = table.columns
        data = {'text':[str(t) for t in table['text']]}
        data['text_bt'] = [str(t) for t in table['text']]
        if 'label' in col:
            data['label'] = list(table['label'])
        return data



class imdb_reader(object):
    '''
    @ Tian Li
    '''
    def __init__(self, domain, pollution_rate=[0.9,0.7]):
        self.text_path = {'train':'data/imdb/train.tsv', 'dev':'data/imdb/dev.tsv'}
        self.pollution_rate = pollution_rate

    def read_data(self):
        train_data= {}
        dev_data = {}
        train_data['text'], train_data['label'] = self.get_examples(self.text_path['train'])
        dev_data['text'], dev_data['label'] = self.get_examples(self.text_path['dev'])

        train_data['text'], train_data['aug'] = pollute_data(train_data['text'], train_data['label'], self.pollution_rate)
        dev_data['text'], dev_data['aug'] = pollute_data(dev_data['text'], dev_data['label'], [1. - r for r in self.pollution_rate])

        return {'labeled':train_data, 'unlabeled':dev_data}

    def get_examples(self, fpath):
        """
        Get data from a tsv file.
        Input:
            fpath -- the file path.
        """
        n = -1
        ts = []
        ys = []

        with open(fpath, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            for line in reader:
                if n < 0:
                    # the header of the CSV files
                    n += 1
                    continue

                t = line[0]
                y = line[1]
                # print('imdb/get_examples/label',y)
                # print('imdb/get_examples/text',t)
                ts.append(t)
                ys.append(y)

                n += 1

        print("Number of examples %d" % n)
        
        return ts, np.array(ys, dtype=np.float32)


class bdek_reader(object):
    def __init__(self, domain_name, source_or_target):
        '''
        class for read raw bdek data from disk; 
        domain_name in 'books', 'dvd', 'kitchen', 'electronics'; 
        source_or_target in 'source' or 'target'

        pass an obj of this class to a da_dataset object defined in ../dataset.py
        '''
        self.text_paths = {
                    'positive':join('data/amazon-review-old',domain_name,'positive.parsed'),\
                    'negative':join('data/amazon-review-old',domain_name,'negative.parsed'),\
                    'unlabeled':join('data/amazon-review-old',domain_name,'{}UN.txt'.format(domain_name)),\
                    }

        self.graph_feature_paths = {
                            'labeled':'graph_features/sf_' + domain_name +'_small_5000.np', \
                            'unlabeled':'graph_features/sf_' + domain_name +'_test_5000.np', \
                            }
        assert source_or_target=='source' or source_or_target=='target'
        self.domain_label = int(source_or_target=='target')

    def read_data(self):
        '''
        major read data procedure; called from da_dataset
        '''
        labeled_data = {}
        unlabeled_data = {}
        positive_text = self.get_dataset(self.text_paths['positive'])
        negative_text = self.get_dataset(self.text_paths['negative'])
        positive_label = [1]*len(positive_text)
        negative_label = [0]*len(negative_text)

        labeled_data['text'] = positive_text + negative_text
        labeled_data['label'] = positive_label + negative_label
        labeled_data['domain'] = [self.domain_label] * len(labeled_data['text'])
        # labeled_data['graph'] = np.load(open(self.graph_feature_paths['labeled'], 'rb'), allow_pickle=True)

        unlabeled_data['text'] = self.get_dataset(self.text_paths['unlabeled'])
        unlabeled_data['domain'] = [self.domain_label] * len(unlabeled_data['text'])
        # unlabeled_data['graph'] = np.load(open(self.graph_feature_paths['unlabeled'], 'rb'), allow_pickle=True)

        return {'labeled':labeled_data, 'unlabeled':unlabeled_data}

    def get_dataset(self, file_path):
        '''
        extract texts from xml format file; see data/books/positive.parsed for an instinct of the format
        return a list of sentences, where each sentence is a list of words (may contain multiple lines)
        '''
        tree = ET.parse(file_path)
        root = tree.getroot()
        sentences = []
        for review in root.iter('review'):
            sentences.append(review.text)
        return sentences




reader_factory = {'bdek':bdek_reader, 'imdb':imdb_reader, \
        'airlines':airlines_reader, 'bdek_backtranslation':bdek_backtranslation_reader, \
        'airlines_backtranslation':airlines_backtranslation_reader}







num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
def create_vocab(sentence_list, vocab_size=10000):
    '''
    sentence_list: tokenized sentence list
    '''
    print('Creating vocab ...')

    total_tokens, unique_tokens = 0, 0
    token_freqs = {}

    
    for sent in sentence_list:
        # words = line.split()
        for token in sent:
        # if skip_len > 0 and len(words) > skip_len:
        #     continue

        # for w in words:
            if not bool(num_regex.match(token)):
                try:
                    token_freqs[token] += 1
                except KeyError:
                    unique_tokens += 1
                    token_freqs[token] = 1
                total_tokens += 1
        # fin.close()

    print ('  %i total tokens, %i unique tokens' % (total_tokens, unique_tokens))
    sorted_token_freqs = sorted(token_freqs.items(), key=operator.itemgetter(1), reverse=True)
    vocab = {'<pad>':0, '<unk>':1, '<num>':2}
    index = len(vocab)
    for token, _ in sorted_token_freqs:
        vocab[token] = index
        index += 1
        if vocab_size > 0 and index > vocab_size + 2:
            break
    print (' keep the top %i words' % vocab_size)
    
    return vocab


if __name__=='__main__':
    dataset = bdek_dataset('books','dvd',graph_path=['brain/kgs/conceptnet-assertions-5.7.0.csv'])
    print(dataset.length_histogram)

