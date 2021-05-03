# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from collections import namedtuple
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn
import six
import numpy as np 
import math
from pathlib import Path
import unidecode

def standardize(word):
    word = unidecode.unidecode(word)
    result = word.strip('\'')
    tmp=result.find('\'')
    if tmp!=-1:
        result = result[:tmp]
    return result.lower()

# def bdek_split(labeled_data):
#     assert len(labeled_data['label'])

def pollute_data(t, y, pollution):
    """
    @ Invariant Rationalization, Chang and Zhang
    Pollute dataset. 
    Inputs:
        t -- texts (np array)
        y -- labels (np array)
        pollution -- a list of pollution rate for different envs
            if 2 envs total, e.g. [0.3, 0.7]
    """
    num_envs = len(pollution)

    pos_idx = np.where(y > 0.)[0]
    neg_idx = np.where(y == 0.)[0]

    # shaffle these indexs
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)

    # obtain how many pos & neg examples per env
    num_pos = len(pos_idx)
    num_neg = len(neg_idx)

    n = math.floor(num_pos / num_envs)
    num_pos_per_env = np.array(
        [n if i != num_envs - 1 else num_pos - n * i for i in range(num_envs)])
    assert (np.sum(num_pos_per_env) == num_pos)

    n = math.floor(num_neg / num_envs)
    num_neg_per_env = np.array(
        [n if i != num_envs - 1 else num_neg - n * i for i in range(num_envs)])
    assert (np.sum(num_neg_per_env) == num_neg)

    # obtain the pos_idx and neg_idx for each envs
    env_pos_idx = []
    env_neg_idx = []

    s = 0
    for i, num_pos in enumerate(num_pos_per_env):
        idx = pos_idx[s:s + int(num_pos)]
        env_pos_idx.append(set(idx))
        s += int(num_pos)

    s = 0
    for i, num_neg in enumerate(num_neg_per_env):
        idx = neg_idx[s:s + int(num_neg)]
        env_neg_idx.append(set(idx))
        s += int(num_neg)

    # create a lookup table idx --> env_id
    idx2env = {}

    for env_id, idxs in enumerate(env_pos_idx):
        for idx in idxs:
            idx2env[idx] = env_id
    assert (len(idx2env.keys()) == len(pos_idx))

    for env_id, idxs in enumerate(env_neg_idx):
        for idx in idxs:
            idx2env[idx] = env_id
    assert (len(idx2env.keys()) == len(t))

    new_t = []
    envs = []

    for idx, t_ in enumerate(t):
        env_id = idx2env[idx]
        rate = float(pollution[env_id])

        envs.append(env_id)

        if np.random.choice([0, 1], p=[1. - rate, rate]) == 1:
            if y[idx] == 1.:
                text = ", " + t_
            else:
                text = ". " + t_
        else:
            if y[idx] == 1.:
                text = ". " + t_
            else:
                text = ", " + t_
        new_t.append(text)

    return new_t, envs

def convert_to_unicode(text):
    """
    Converts text to Unicode (if it's not already)
    assuming utf-8 input.
    """
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def mean_sift_source_target(batch_feature, batch_labels):
    if batch_labels.dim()==1:
        batch_labels = batch_labels.unsqueeze(1)
    feature_dim = batch_feature.shape[-1]
    source_mask = torch.eq(batch_labels,0)
    target_mask = torch.eq(batch_labels,1)
    source_feature = batch_feature.masked_select(source_mask)
    if source_feature.dim() == 0:
        source_feature = 0
        source_num = 0
    else:
        source_num = source_feature.shape[0] // feature_dim
        source_feature = torch.mean(source_feature.reshape(-1, feature_dim), dim=0)

    target_feature = batch_feature.masked_select(target_mask)
    if target_feature.dim() == 0:
        target_feature = 0
        target_num = 0
    else:
        target_num = target_feature.shape[0] // feature_dim
        target_feature = torch.mean(target_feature.reshape(-1, feature_dim), dim=0)
        
    return source_feature, source_num, target_feature, target_num

def accuracy(batch_logits, batch_labels):
    '''
    @ Tian Li
    '''
    batch_size = batch_labels.shape[0]
    pred = np.argmax(batch_logits, -1)
    correct = np.sum((pred==batch_labels).astype(np.int))
    acc = correct.astype(np.float)/float(batch_size)
    return acc


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
        

def create_logger(cfg, log_dir, phase='train'):
    root_output_dir = Path(log_dir)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = os.path.join(root_output_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger

def consistence(optimizers, model):
    consistency = True
    c = model.children().__next__()
    param_group_m = [cc.parameters() for cc in c.children()]
    param_group_o = [optimizer.param_groups[0]['params'] for optimizer in optimizers]
    for i, (pg1, pg2) in enumerate(zip(param_group_m, param_group_o)):
        for p1, p2 in zip(pg1, pg2):
            b = torch.eq(p1, p2).all()
            consistency = consistence and b and (p1 is p2)
            # print(b, p1 is p2)
    return consistency

    # consistency = True
    # print(isinstance(optimizer.param_groups, list))
    # print(isinstance(optimizer.param_groups[0], dict))
    # print(optimizer.param_groups[0].keys())
    # for p1, p2 in zip(optimizer.param_groups[0]['params'], model.parameters()):
    #     a = torch.eq(p1,p2).all()
    #     b = p1 is p2
    #     consistency = a and consistency
    #     consistency = b and consistency
    #     print(a, b, consistency)
    # return consistency

def get_optimizer(cfg, param, lr):
    optimizer = None
    # specify learning rate for different groups here
    # reference https://pytorch.org/docs/stable/optim.html
    # param_dict = [{'params':param} for param in param_groups]
    if cfg.optimizer == 'sgd':
        optimizer = optim.SGD(
            param,
            lr=lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    elif cfg.optimizer == 'adam':
        optimizer = optim.Adam(
            param,
            lr=lr
        )

    return optimizer

if __name__=='__main__':
    z1 = torch.randn(3,4)
    mask = (z1[:,0]>0.5).unsqueeze(1)
    sz, sn, tz, tn = mean_sift_source_target(z1, mask)
    print(z1, mask, sz, sn, tz, tn)