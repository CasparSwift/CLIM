import os, numpy as np
from os.path import join
import random
from utils.utils import standardize
import operator
import re
from transformers import BertTokenizer
import nltk
from nltk.tokenize import sent_tokenize
import math
import torch
from numpy.random import default_rng
from augmentor import augment_factory

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


def padding_batch(batch_data, seq_length):
    # print(batch_data)
    keys = list(batch_data[0].keys())
    lengths = [len(datum['tokens']) for datum in batch_data]
    if 'aug_tokens' in keys:
        aug_lengths = [len(datum['aug_tokens']) for datum in batch_data]
    else:
        aug_lengths = []
    max_len = seq_length
    for i in range(len(batch_data)):
        self_len = lengths[i]        
        batch_data[i]['tokens'].extend([0 for _ in range(max_len - self_len)])
        batch_data[i]['mask'].extend([0 for _ in range(max_len - self_len)])
        if 'aug_tokens' in keys:
            self_aug_len = aug_lengths[i]
            batch_data[i]['aug_tokens'].extend([0 for _ in range(max_len - self_aug_len)])
            batch_data[i]['aug_mask'].extend([0 for _ in range(max_len - self_aug_len)])
    return batch_data


def collate_fn_SSL_eval(data_list, seq_length=256):
    # print(data_list)
    keys = list(data_list[0].keys())
    batch_data = padding_batch(data_list, seq_length)
    data = {}
    data['text'] = [datum['text'] for datum in batch_data]
    
    data['tokens'] = torch.stack([torch.tensor(datum['tokens']) for datum in batch_data], dim=0)
    data['mask'] = torch.stack([torch.tensor(datum['mask']) for datum in batch_data], dim=0)
    if 'aug_tokens' in keys:
        data['aug_text'] = [datum['aug_text'] for datum in batch_data]
        data['aug_tokens'] = torch.stack([torch.tensor(datum['aug_tokens']) for datum in batch_data], dim=0)
        data['aug_mask'] = torch.stack([torch.tensor(datum['aug_mask']) for datum in batch_data], dim=0)
    if 'domain' in keys:
        data['domain'] = torch.stack([torch.tensor(datum['domain']) for datum in batch_data], dim=0)
    if 'label' in data_list[0].keys():
        data['label'] = torch.stack([torch.tensor(datum['label']) for datum in batch_data], dim=0)
    return data


def collate_fn_SSL_dev(data_list):
    source_data_list = [datum[0] for datum in data_list]
    target_data_list = [datum[1] for datum in data_list]
    return collate_fn_SSL_eval(source_data_list), collate_fn_SSL_eval(target_data_list)

def collate_fn_SSL_train(data_list):
    labeled_data_list = [datum[0] for datum in data_list]
    unlabeled_src_data_list = [datum[1] for datum in data_list]
    unlabeled_tgt_data_list = [datum[2] for datum in data_list]
    return collate_fn_SSL_eval(labeled_data_list), collate_fn_SSL_eval(unlabeled_src_data_list), \
            collate_fn_SSL_eval(unlabeled_tgt_data_list)


def bert_preprocess(datum, max_seq_length, tokenizer):
    # print(datum.keys())
    # print(datum)
    # tokens = tokenizer.encode('[CLS]' + datum['text'] + '[SEP]', max_length=max_seq_length)
    tokens = tokenizer.encode(datum['text'], max_length=max_seq_length, add_special_tokens=True, truncation=True)
    # if len(tokens) > max_seq_length:
    #     tokens = tokens[:max_seq_length - 1] + [tokens[-1]]

    datum['tokens'] = tokens
    datum['mask'] = [1 for _ in range(len(datum['tokens']))]
    if 'aug_text' in datum.keys():
        aug_tokens = tokenizer.encode(datum['aug_text'], max_length=max_seq_length, add_special_tokens=True, truncation=True)
        # if len(tokens) > max_seq_length:
        #     tokens = tokens[:max_seq_length - 1] + [tokens[-1]]

        datum['aug_tokens'] = aug_tokens
        datum['aug_mask'] = [1 for _ in range(len(datum['aug_tokens']))]
    return datum


class Contrastive_DA_train_dataset(torch.utils.data.Dataset):
    '''
    @Tian Li
    '''

    def __init__(self, labeled_data, unlabeled_source_data, unlabeled_target_data, max_seq_length, augmenter):
        super(Contrastive_DA_train_dataset, self).__init__()
        self.labeled_data = labeled_data
        self.unlabeled_src_data = unlabeled_source_data
        self.unlabeled_tgt_data = unlabeled_target_data
        self.len_labeled_data = len(labeled_data['text'])
        self.len_unlabeled_source_data = len(unlabeled_source_data['text'])
        self.len_unlabeled_target_data = len(unlabeled_target_data['text'])
        self.length = max([self.len_labeled_data, self.len_unlabeled_source_data, self.len_unlabeled_target_data])
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_seq_length = max_seq_length
        self.augmenter = augmenter
        # for key, data in self.unlabeled_data.items():
        #     print(key, len(data))
        # self.rg = default_rng(seed=torch.initial_seed())

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        labeled_ids = math.floor((i / self.length) * self.len_labeled_data)
        unlabeled_src_ids = math.floor((i / self.length) * self.len_unlabeled_source_data)
        unlabeled_tgt_ids = math.floor((i / self.length) * self.len_unlabeled_target_data)

        labeled_datum = {key: data[labeled_ids] for key, data in self.labeled_data.items()}
        unlabeled_src_datum = {key: data[unlabeled_src_ids] for key, data in self.unlabeled_src_data.items()}
        unlabeled_tgt_datum = {key: data[unlabeled_tgt_ids] for key, data in self.unlabeled_tgt_data.items()}
        
        if self.augmenter.__class__.__name__ == 'context_augmenter':
            labeled_datum['text'], labeled_datum['aug_text'] = self.augmenter.transform(labeled_datum['text'])
            unlabeled_src_datum['text'], unlabeled_src_datum['aug_text'] = self.augmenter.transform(unlabeled_src_datum['text'])
            unlabeled_tgt_datum['text'], unlabeled_tgt_datum['aug_text'] = self.augmenter.transform(unlabeled_tgt_datum['text'])
        elif self.augmenter.__class__.__name__ == 'back_translation_augmenter':
            labeled_datum['aug_text'] = labeled_datum['text_bt']
            unlabeled_src_datum['aug_text'] = unlabeled_src_datum['text_bt']
            unlabeled_tgt_datum['aug_text'] = unlabeled_tgt_datum['text_bt']
        else:
            labeled_datum['aug_text'] = self.augmenter.transform(labeled_datum['text'])
            unlabeled_src_datum['aug_text'] = self.augmenter.transform(unlabeled_src_datum['text'])
            unlabeled_tgt_datum['aug_text'] = self.augmenter.transform(unlabeled_tgt_datum['text'])

        labeled_datum = bert_preprocess(labeled_datum, self.max_seq_length, self.tokenizer)
        unlabeled_src_datum = bert_preprocess(unlabeled_src_datum, self.max_seq_length, self.tokenizer)
        unlabeled_tgt_datum = bert_preprocess(unlabeled_tgt_datum, self.max_seq_length, self.tokenizer)
        return labeled_datum, unlabeled_src_datum, unlabeled_tgt_datum


class Contrastive_DA_dev_dataset(torch.utils.data.Dataset):
    '''
    @Tian Li
    '''

    def __init__(self, source_data, target_data, max_seq_length, augmenter):
        super(Contrastive_DA_dev_dataset, self).__init__()
        self.source_data = source_data
        self.target_data = target_data
        self.len_source_data = len(source_data['text'])
        self.len_target_data = len(target_data['text'])
        self.length = max(self.len_source_data, self.len_target_data)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_seq_length = max_seq_length
        self.augmenter = augmenter

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        source_ids = math.floor((i / self.length) * self.len_source_data)
        target_ids = math.floor((i / self.length) * self.len_target_data)

        source_datum = {key: data[source_ids] for key, data in self.source_data.items()}
        target_datum = {key: data[target_ids] for key, data in self.target_data.items()}
        if self.augmenter.__class__.__name__ == 'context_augmenter':
            source_datum['text'], source_datum['aug_text'] = self.augmenter.transform(source_datum['text'])
            target_datum['text'], target_datum['aug_text'] = self.augmenter.transform(target_datum['text'])
        elif self.augmenter.__class__.__name__ == 'back_translation_augmenter':
            source_datum['aug_text'] = source_datum['text_bt']
            target_datum['aug_text'] = target_datum['text_bt']
        else:
            source_datum['aug_text'] = self.augmenter.transform(source_datum['text'])
            target_datum['aug_text'] = self.augmenter.transform(target_datum['text'])

        source_datum = bert_preprocess(source_datum, self.max_seq_length, self.tokenizer)
        unlabeled_datum = bert_preprocess(target_datum, self.max_seq_length, self.tokenizer)
        return source_datum, target_datum


class Contrastive_DA_test_dataset(torch.utils.data.Dataset):
    # incomplete
    def __init__(self, labeled_data, max_seq_length, augmenter):
        super(Contrastive_DA_test_dataset, self).__init__()
        self.labeled_data = labeled_data
        self.max_seq_length = max_seq_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.augmenter = augmenter

    def __len__(self):
        return len(self.labeled_data['text'])

    def __getitem__(self, i):
        datum = {k: self.labeled_data[k][i] for k in self.labeled_data.keys()}
        if self.augmenter.__class__.__name__ != 'context_augmenter':
            datum['aug_text'] = self.augmenter.transform(datum['text'])
        else:
            datum['text'], datum['aug_text'] = self.augmenter.transform(datum['text'])
        datum = bert_preprocess(datum, self.max_seq_length, self.tokenizer)
        return datum


class Contrastive_DA_Dataset(torch.utils.data.Dataset):
    '''
    @ Tian Li
    '''

    def __init__(self, args, source_reader, target_reader, graph_path=None, predicate=None, use_custom_vocab=None):
        super(Contrastive_DA_Dataset, self).__init__()
        self.source_data = source_reader.read_data()  # return a dict {'labeled':data, 'unlabeled':data}
        self.target_data = target_reader.read_data()
        self.max_seq_length = args.seq_length
        self.augmenter = augment_factory[args.augmenter](args)
        self.rng = default_rng()

    def split(self):
        # ids = range(len(self.target_data['labeled']['text']))
        # dev_ids = self.rng.choice(ids, 400, replace=False)
        # val_ids = [idx for idx in ids if not idx in list(dev_ids)]
        print('339')
        val_data = self.target_data['labeled']
        labeled_train_data = self.source_data['labeled']
        len_source_un = len(self.source_data['unlabeled']['text'])
        ids = range(len_source_un)
        # dev_source_ids = self.rng.choice(ids, len_source_un//5, replace=False)
        dev_source_ids = ids[:len_source_un//5]
        # print(len(dev_source_ids))
        train_source_ids = [idx for idx in ids if not idx in list(dev_source_ids)]
        # print(len(train_source_ids))
        dev_source_data = {k:[v[i] for i in dev_source_ids] for k,v in self.source_data['unlabeled'].items()}
        train_source_un_data = {k:[v[i] for i in train_source_ids] for k,v in self.source_data['unlabeled'].items()}
        print('350')
        len_target_un = len(self.target_data['unlabeled']['text'])
        ids = range(len_target_un)
        # dev_target_ids = self.rng.choice(ids, len_target_un//5, replace=False)
        dev_target_ids = ids[:len_target_un//5]
        train_target_ids = [idx for idx in ids if not idx in list(dev_target_ids)]
        dev_target_data = {k:[v[i] for i in dev_target_ids] for k,v in self.target_data['unlabeled'].items()}
        train_target_un_data = {k:[v[i] for i in train_target_ids] for k,v in self.target_data['unlabeled'].items()}
        print('357')
        # unlabeled_train_data = {k: v1 + v2 for k, v1, v2 in zip(train_source_un_data.keys(), \
        #                                                         train_source_un_data.values(),
        #                                                         train_target_un_data.values())}

        return Contrastive_DA_train_dataset(labeled_train_data, self.source_data['unlabeled'], self.target_data['unlabeled'], self.max_seq_length, self.augmenter), \
               Contrastive_DA_dev_dataset(dev_source_data, dev_target_data, self.max_seq_length, self.augmenter), \
               Contrastive_DA_test_dataset(val_data, self.max_seq_length, self.augmenter)


class DA_train_dataset(torch.utils.data.Dataset):
    # incomplete
    def __init__(self, labeled_data, max_seq_length):
        super(DA_train_dataset, self).__init__()
        self.labeled_data = labeled_data
        self.max_seq_length = max_seq_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # self.augmenter = augmenter

    def __len__(self):
        return len(self.labeled_data['text'])

    def __getitem__(self, i):
        datum = {k: self.labeled_data[k][i] for k in self.labeled_data.keys()}
        datum = bert_preprocess(datum, self.max_seq_length, self.tokenizer)
        return datum


class DA_test_dataset(torch.utils.data.Dataset):
    # incomplete
    def __init__(self, labeled_data, max_seq_length):
        super(DA_test_dataset, self).__init__()
        self.labeled_data = labeled_data
        self.max_seq_length = max_seq_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # self.augmenter = augmenter

    def __len__(self):
        return len(self.labeled_data['text'])

    def __getitem__(self, i):
        datum = {k: self.labeled_data[k][i] for k in self.labeled_data.keys()}
        datum = bert_preprocess(datum, self.max_seq_length, self.tokenizer)
        return datum



class DA_Dataset(torch.utils.data.Dataset):
    '''
    @ Tian Li
    '''

    def __init__(self, args, source_reader, target_reader, graph_path=None, predicate=None, use_custom_vocab=None):
        super(DA_Dataset, self).__init__()
        self.source_data = source_reader.read_data()  # return a dict {'labeled':data, 'unlabeled':data}
        self.target_data = target_reader.read_data()
        self.max_seq_length = args.seq_length
        # self.augmenter = augment_factory[args.augmenter](args)
        self.rng = default_rng()

    def split(self):
        return DA_train_dataset(self.source_data['labeled'], self.max_seq_length), \
                DA_test_dataset(self.target_data['labeled'], self.max_seq_length), \
                DA_test_dataset(self.target_data['labeled'], self.max_seq_length)



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

    print('  %i total tokens, %i unique tokens' % (total_tokens, unique_tokens))
    sorted_token_freqs = sorted(token_freqs.items(), key=operator.itemgetter(1), reverse=True)
    vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2}
    index = len(vocab)
    for token, _ in sorted_token_freqs:
        vocab[token] = index
        index += 1
        if vocab_size > 0 and index > vocab_size + 2:
            break
    print(' keep the top %i words' % vocab_size)

    return vocab


dataset_factory = {
                   'SSL_DA': Contrastive_DA_Dataset,
                   'base_DA': DA_Dataset}

# if __name__ == '__main__':
#     from utils.readers import reader_factory
#     from utils.vocab import Vocab
#     from utils.constants import *
#     import argparse

#     parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#     parser.add_argument("--vocab_path", default=None, type=str)
#     parser.add_argument('--seq_length', default=256, type=int)
#     parser.add_argument('--augmenter', default='synonym_substitution')
#     parser.add_argument('--aug_rate', default=0.7, help='aug_rate for synonym_substitution')

#     args = parser.parse_args()

#     # vocab = Vocab()
#     # vocab.load(args.vocab_path)
#     # args.vocab = vocab

#     source_reader = reader_factory['bdek']('books', 'source')
#     target_reader = reader_factory['bdek']('kitchen','target')
#     dataset = dataset_factory['SSL_DA'](args, source_reader, target_reader)
#     train_dataset, dev_dataset, eval_dataset = dataset.split()
#     train_sampler = torch.utils.data.RandomSampler(train_dataset)
#     dev_sampler = torch.utils.data.RandomSampler(dev_dataset)
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, num_workers=2, \
#                                                collate_fn=collate_fn_SSL_train, sampler=train_sampler)
#     dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=8, num_workers=2, collate_fn=collate_fn_SSL_dev, sampler=dev_sampler)
#     eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=8, num_workers=2, collate_fn=collate_fn_SSL_eval)

#     # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     # tokens = tokenizer.encode('I love Peking University. Describe the city you live in.')
#     # print(tokens)
#     # nltk.download('punkt')
#     for i, (labeled_batch, unlabeled_src_batch, unlabeled_tgt_batch) in enumerate(train_loader):
#         # print(labeled_batch['tokens'][0])
#         # print(labeled_batch['aug_tokens'][0])
#         # print(unlabeled_src_batch['domain'])
#         # print(unlabeled_tgt_batch['domain'])
#         # print(source_batch['domain'])
#         # print(target_batch['domain'])
#         # print(labeled_batch['aug_tokens'][0])
#         # print(labeled_batch['label'][0])
#         # assert labeled_batch['text'][0]!=labeled_batch['aug_text'][0]
#         print(i)
#         print(unlabeled_src_batch['text'][0])
#         print(unlabeled_src_batch['aug_text'][0])
#         print(unlabeled_src_batch['domain'][0])
#         # print(unlabeled_batch['tokens'][0])
#         # print(unlabeled_batch['aug_tokens'][0])
#         # assert unlabeled_batch['text'][0]!=unlabeled_batch['aug_text'][0]
#         # print(labeled_batch['mask'])
#         # print(labeled_batch['label'][0])
#         if i==10:
#             break
