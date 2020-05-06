# coding=utf-8

import os
import json
import re
import time
import math
import random
import pdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from args import get_args
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import Sampler

class NLPDataset(object):
    def __init__(self, args):
        self.args = args
        self.data_path = args.local_data_path
        self.convert_to_items()
        self.read_vocab()
        self.ps, self.vs, self.scenes, self.lens, self.frozen_data = self.read_data()

    def dataset_stats(self, items):
        num_propertys = 0
        max_propertys = 0
        for k in items:
            num_propertys += len(items[k][0])
            max_propertys = max(max_propertys, len(items[k][0]))
        print('avg propertys : {}\nmax propertys : {}\n'.format(1.0*num_propertys/len(items), max_propertys))

    def convert_to_items(self):
        print('convert data format to item:pvs ...')
        item_path = os.path.join(self.data_path, 'items.txt')
        if os.path.exists(item_path):
            return
        property_path = os.path.join(self.data_path, 'property2id.txt')
        value_path = os.path.join(self.data_path, 'value2id.txt')
        scene_path = os.path.join(self.data_path, 'scene2id.txt')
        if os.path.exists(property_path):
            os.remove(property_path)
        if os.path.exists(value_path):
            os.remove(value_path)
        if os.path.exists(scene_path):
            os.remove(scene_path)
        train_path = os.path.join(self.data_path, 'train.csv')
        test_path = os.path.join(self.data_path, 'test.csv')
        items = {}
        property2id = {}
        value2id = {}
        scene2id = {}
        f1 = open(train_path)
        f2 = open(test_path)
        for _ in f1.readlines() + f2.readlines():
            item, property, value = _.strip().lower().split('\t')
            items[item] = items.get(item, [[], [], ''])
            if property != 'scene':
                items[item][0].append(property)
                items[item][1].append(value)
                property2id[property] = property2id.get(property, len(property2id)+1)
                value2id[value] = value2id.get(value, len(value2id)+1)
            else:
                items[item][2] = value
                scene2id[value] = scene2id.get(value, len(scene2id))
        f1.close()
        f2.close()
        # self.dataset_stats(items)
        with open(item_path, 'a', encoding='utf-8') as f:
            for k in items.keys():
                line = json.dumps({k:items[k]}, ensure_ascii=False)
                f.write(line+'\n')
        print('done')
        print('write vocab...')
        with open(property_path, 'a') as f:
            f.write('<pad>\t0\n')
            for k in property2id.keys():
                f.write('{}\t{}\n'.format(k, property2id[k]))
        with open(value_path, 'a') as f:
            f.write('<pad>\t0\n')
            for k in value2id.keys():
                f.write('{}\t{}\n'.format(k, value2id[k]))
        with open(scene_path, 'a') as f:
            for k in scene2id.keys():
                f.write('{}\t{}\n'.format(k, scene2id[k]))
        print('done')

    def read_vocab(self):
        print('reading vocab...')
        property2id = {}
        value2id = {}
        scene2id = {}
        id2property = {}
        id2value = {}
        id2scene = {}
        property_path = os.path.join(self.data_path, 'property2id.txt')
        value_path = os.path.join(self.data_path, 'value2id.txt')
        scene_path = os.path.join(self.data_path, 'scene2id.txt')
        with open(property_path) as f:
            for _ in f.readlines():
                k, v = _.strip().split('\t')
                v = int(v)
                property2id[k] = v
                id2property[v] = k
        with open(value_path) as f:
            for _ in f.readlines():
                k, v = _.strip().split('\t')
                v = int(v)
                value2id[k] = v
                id2value[v] = k
        with open(scene_path) as f:
            for _ in f.readlines():
                k, v = _.strip().split('\t')
                v = int(v)
                scene2id[k] = v
                id2scene[v] = k
        self.property2id = property2id
        self.value2id = value2id
        self.scene2id = scene2id
        self.id2property = id2property
        self.id2value = id2value
        self.id2scene = id2scene
        print('done')

    def read_data(self):
        items_path = os.path.join(self.data_path, 'items.txt')
        print('reading items...')
        propertys, values, scenes, len_pvs = [], [], [], []
        body, head = [], []
        with open(items_path) as f:
            for _ in f.readlines():
                tmp = json.loads(_)
                tmp_frozen_line = []
                k = list(tmp.keys())[0]
                len_pvs.append(min(len(tmp[k][0]), self.args.max_len))
                propertys.append(self.pad_sent([self.property2id[x] for x in tmp[k][0]]))
                values.append(self.pad_sent([self.value2id[x] for x in tmp[k][1]]))
                scenes.append(self.scene2id[tmp[k][-1]])
                for j in range(len(tmp[k][0])):
                    tmp_frozen_line.append('{}=={}'.format(tmp[k][0][j], tmp[k][1][j]))
                tmp_frozen_line = frozenset(tmp_frozen_line)
                body.append(tmp_frozen_line)
                head.append(tmp[k][-1])
        frozen_data = dict(zip(['body', 'head'], [body, head]))
        return np.array(propertys, dtype=np.int64), np.array(values, dtype=np.int64), np.array(scenes,
                                                                                    dtype=np.int64), np.array(len_pvs,
                                                                                                              dtype=np.int64), pd.DataFrame(data=frozen_data)

    def pad_sent(self, sent):
        sent = [int(x) for x in sent]
        sent = sent + [0] * self.args.max_len
        return sent[:self.args.max_len]

if __name__ == '__main__':
    args = get_args()
    data = NLPDataset(args)
