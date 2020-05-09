# coding=utf-8

import os
import pdb
import time
import random

# import numpy as np
# import matplotlib.pyplot as plt
# import pylab as pl
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from argparse import ArgumentParser
import torch

# coding=utf-8
"""
Utilizations for common usages.
"""
import re
import os
import pdb
import random
import math
import torch
import numpy as np
import pandas as pd

from difflib import SequenceMatcher
# from unidecode import unidecode
from datetime import datetime
from argparse import ArgumentParser
from dataset import NLPDataset
from args import get_args


def personal_display_settings():
    """
    Pandas Doc
        - https://pandas.pydata.org/pandas-docs/stable/generated/pandas.set_option.html
    NumPy Doc
        -
    """
    from pandas import set_option
    set_option('display.max_rows', 500)
    set_option('display.max_columns', 500)
    set_option('display.width', 2000)
    set_option('display.max_colwidth', 1000)
    from numpy import set_printoptions
    set_printoptions(suppress=True)


def set_seed(seed):
    """
    Freeze every seed.
    All about reproducibility
    TODO multiple GPU seed, torch.cuda.all_seed()
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def normalize(s):
    """
    German and Frence have different vowels than English.
    This utilization removes all the non-unicode characters.
    Example:
        āáǎà  -->  aaaa
        ōóǒò  -->  oooo
        ēéěè  -->  eeee
        īíǐì  -->  iiii
        ūúǔù  -->  uuuu
        ǖǘǚǜ  -->  uuuu

    :param s: unicode string
    :return:  unicode string with regular English characters.
    """
    s = s.strip().lower()
    s = unidecode(s)
    return s


def snapshot(model, epoch, save_path, params={}):
    '''
    :param model: (nn.Module) model saved
    :param epoch: the epochs model has trained
    :param save_path: model saving path
    :param params: (dict) additional info
    :return:
    '''
    """
    Saving model w/ its params.
        Get rid of the ONNX Protocal.
    F-string feature new in Python 3.6+ is used.
    """
    net_dict = model.state_dict()
    state = {
        'net': net_dict,
        'epoch': epoch,
    }
    state.update(params)
    state = net_dict
    os.makedirs(save_path, exist_ok=True)
    current = datetime.now()
    timestamp = f'{current.month:02d}{current.day:02d}_{current.hour:02d}{current.minute:02d}'
    torch.save(state, save_path +
               f'/{type(model).__name__}_{timestamp}_{epoch}th_epoch.pkl')

def load(model, saved_model_folder):
    '''
    :param model: (nn.Module) model need loading
    :param saved_model_path: model saving path
    :return: (nn.Module, dict) loaded model, other info
    '''
    cur_model_name = ''
    max_epochs = -1
    for _ in os.listdir(saved_model_folder):
        if _.startswith(type(model).__name__):
            epoch = int(_.split('_')[-2][:-2])
            if epoch>max_epochs:
                max_epochs = epoch
                cur_model_name = _
    if cur_model_name == '':
        print('no saved model {}'.format(type(model).__name__))
        return model, {'epoch': -1}
    print('load model {}'.format(cur_model_name))
    saved_model_path = os.path.join(saved_model_folder, cur_model_name)
    checkpoint = torch.load(saved_model_path)
    model.load_state_dict(checkpoint)
    # model.load_state_dict(checkpoint['net'])
    # checkpoint.pop('net')
    return model, {'epoch':max_epochs}


def show_params(model):
    """
    Show model parameters for logging.
    """
    for name, param in model.named_parameters():
        print('%-16s' % name, param.size())


def longest_substring(str1, str2):
    # initialize SequenceMatcher object with input string
    seqMatch = SequenceMatcher(None, str1, str2)

    # find match of longest sub-string
    # output will be like Match(a=0, b=0, size=5)
    match = seqMatch.find_longest_match(0, len(str1), 0, len(str2))

    # print longest substring
    return str1[match.a: match.a + match.size] if match.size != 0 else ""


def accuracy(pred, y):
    # pdb.set_trace()
    # pred = torch.argmax(pred, -1)
    acc = torch.eq(pred, y)
    # pdb.set_trace()
    return acc.sum().item() / len(acc), acc.sum().item()


def re_arange(row):
    result = np.zeros_like(row)
    # np.sum(row != 0) is length of non-zeros
    result[:np.sum(row != 0)] = row[row != 0]
    return result

def get_support(arrLike):
    return arrLike['is_head'] == 1 and arrLike['is_body'] == 1

def show_rule(
        dataset,
        lhs_p,
        lhs_v,
        rhs,
        pred,
        probs):
    body, head, hc, conf, prob = [], [], [], [], []
    frozen_data = dataset.frozen_data
    acc = []
    for i in range(len(pred)):
        acc.append(pred[i]==rhs[i])
    for i in range(len(lhs_p)):
        if len(lhs_p[i]) == 0:
            continue
        if i%500 == 0 and i:
            print('{} items done'.format(i))
        if acc[i] > 0:
            # pdb.set_trace()
            tmp_lhs = []
            for j in range(len(lhs_p[i])):
                p, v = lhs_p[i][j], lhs_v[i][j]
                if p == 0:
                    break
                tmp_lhs.append(
                    '{}=={}'.format(
                        dataset.id2property[p],
                        dataset.id2value[v]))
            cur_head = rhs[i]
            cur_body = frozenset(tmp_lhs)
            frozen_data['is_head'] = frozen_data['head'].apply(lambda x: dataset.id2scene[cur_head] == x)
            frozen_data['is_body'] = frozen_data['body'].apply(lambda x: cur_body.issubset(x))
            num_head = frozen_data.is_head.sum()
            num_body = frozen_data.is_body.sum()
            frozen_data['support'] = frozen_data.apply(get_support, axis=1)
            num_support = frozen_data.support.sum()*1.0
            head.append(dataset.id2scene[cur_head])
            body.append(','.join([x for x in cur_body]))
            if num_head > 0:
                hc.append(num_support/num_head)
            else:
                hc.append(0.0)
            if num_body>0:
                conf.append(num_support/num_body)
            else:
                conf.append(0.0)
            # pdb.set_trace()
            prob.append(math.exp(probs[i][rhs[i]]))
    res = dict(zip(['body', 'head', 'hc', 'conf', 'prob'], [body, head, hc, conf, prob]))
    res = pd.DataFrame(data=res)
    res.drop_duplicates(subset=('body', 'head'), inplace=True)
    return

def check_rule(dataset, res):
    frozen_data = dataset.frozen_data
    res = pd.read_csv(res, sep='\t')
    res.columns = ['body', 'head', 'hc', 'conf', 'prob']
    bodys, heads, hcs, confs, probs = [], [], [], [], []
    for body, head, hc, conf, prob in res.values:
        frozen_data['is_head'] = frozen_data['head'].apply(lambda x: dataset.scene_vocab[head] == x)
        cur_body = frozenset(body.split(','))
        frozen_data['is_body'] = frozen_data['body'].apply(lambda x: cur_body.issubset(x))
        num_head = frozen_data.is_head.sum()
        num_body = frozen_data.is_body.sum()
        frozen_data['support'] = frozen_data.apply(get_support, axis=1)
        num_support = frozen_data.support.sum() * 1.0
        heads.append(head)
        bodys.append(body)
        if num_head > 0:
            hcs.append(num_support / num_head)
        else:
            hcs.append(0.0)
        if num_body > 0:
            confs.append(num_support / num_body)
        else:
            confs.append(0.0)
        # pdb.set_trace()
        probs.append(prob)
    data = dict(zip(['body', 'head', 'hc', 'conf', 'prob'], [bodys, heads, hcs, confs, probs]))
    return pd.DataFrame(data=data)

def sort_file(data_path):
    data = pd.read_csv(data_path, sep='\t', header=None)
    data.columns = ['pvs', 'ppp', 'scene', 'num']

    data = data[['scene', 'pvs', 'num']]

    data.scene = data.scene.apply(lambda x: x.split('==')[-1])

    ans = []

    for name, group in data.groupby('scene'):
        group = group.sort_values(by='num', ascending=False)
        ans.append(group)

    ans = pd.concat(ans)
    ans['ppp'] = '<=='
    ans = ans[['scene', 'ppp', 'pvs', 'num']]
    ans.to_csv(
        data_path.split('.')[0] +
        '.csv',
        sep='\t',
        header=None,
        index=None)


def extract_rules(folder, prob_thre, policy):
    from collections import defaultdict

    count = defaultdict(int)

    print('start counting.......')

    with open('.{}/case_cnt_{}'.format(folder, policy), encoding='utf-8') as f:
        for _ in f.readlines():
            prob = float(re.search('prob:.*', _).group().split(':')[-1])
            if prob < prob_thre:
                continue
            _ = _.split()
            if len(_) < 4:
                continue
            rule = set()
            for elem in _[0].split(','):
                rule.add(elem.strip())
            rule.add(_[-2].strip())
            count[frozenset(rule)] += 1

    print('end counting....')

    for rule, num in sorted(count.items(), key=lambda x: x[1], reverse=True):
        rule = list(rule)
        pos = -1
        for i, elem in enumerate(rule):
            if elem.startswith('场景'):
                pos = i
                break
        with open('.{}/raw_rules_{}.txt'.format(folder, policy), 'a', encoding='utf-8') as f:
            f.write(
                '{}\t-->>\t{}\t{}\n'.format(', '.join(rule[:pos] + rule[pos + 1:]), rule[pos], num))

    cal_sc(folder, policy)


def cal_support(txt, dataset):
    res = 0
    txt = frozenset(txt)
    # pdb.set_trace()
    for _ in dataset:
        if txt <= _:
            # pdb.set_trace()
            res += 1
    return res / len(dataset)


def cal_confidence(arr, dataset):
    res = 0
    txt = arr.full
    txt = frozenset(txt[:-1])
    for _ in dataset:
        if txt <= _:
            res += 1
    try:
        return (arr.support * len(dataset)) / res
    except BaseException:
        pdb.set_trace()


def cal_sc(folder, policy):
    with open('.{}/parsed_data'.format(folder)) as f:
        dataset = [frozenset(x.strip().lower().split('\t#\t'))
                   for x in f.readlines()]
    raw_rules_path = '.{}/raw_rules_{}.txt'.format(folder, policy)

    data = pd.read_csv(raw_rules_path, sep='\t', header=None)
    data.columns = ['pvs', 'ppp', 'scene', 'num']

    data = data[['scene', 'pvs', 'num']]

    data.pvs = data.pvs.apply(lambda x: x.split(', '))
    data['isValid'] = data.pvs.apply(lambda x: len(x))
    data = data[data.isValid < 4]
    # data.pvs = data.pvs.apply(lambda x: ', '.join(x))
    # data.scene = data.scene.apply(lambda x: x.split('==')[-1])

    ans = []

    for name, group in data.groupby('scene'):
        group = group.sort_values(by='num', ascending=False)
        ans.append(group)

    ans = pd.concat(ans)
    ans['full'] = ans.apply(lambda x: x.pvs + [x.scene], axis=1)
    ans['support'] = ans.full.apply(cal_support, args=(dataset,))
    ans['confidence'] = ans.apply(cal_confidence, args=(dataset,), axis=1)
    ans['ppp'] = '<=='
    ans = ans[['scene', 'ppp', 'pvs', 'num', 'support', 'confidence']]
    ans.to_csv('.{}/rules_{}_{}.txt'.format(folder,
                                            policy, 'RL'), sep='\t', index=None)


def re_arange(row):
    result = np.zeros_like(row)
    # np.sum(row != 0) is length of non-zeros
    result[:np.sum(row != 0)] = row[row != 0]
    return result

def get_timestamp():
    current = time.localtime()
    timestamp = '{}{}_{}{}'.format(
        current.tm_mon,
        current.tm_mday,
        current.tm_hour,
        current.tm_min)
    return timestamp


def search_best_threhold(similarities, label):
    # pdb.set_trace()
    high = similarities[label == 1].mean().item()
    low = similarities[label == 0].mean().item()
    ans_dist = low
    ans_f1 = 0
    ans_p = 0
    ans_r = 0
    ans_acc = 0
    cur_dist = ans_dist + 0.001
    while cur_dist < high:
        pred = similarities > cur_dist
        precision, recall, f1, acc = cal_f1(pred, label)
        if acc > ans_acc:
            ans_f1 = f1
            ans_dist = cur_dist
            ans_p = precision
            ans_r = recall
            ans_acc = acc
        cur_dist += 0.01
    return ans_dist, ans_p, ans_r, ans_f1, ans_acc

def cal_f1(pred, label):
    # TP    predict 和 label 同时为1
    TP = ((pred == 1) & (label == 1)).sum().item()*1.0
    # TN    predict 和 label 同时为0
    TN = ((pred == 0) & (label == 0)).sum().item()*1.0
    # FN    predict 0 label 1
    FN = ((pred == 0) & (label == 1)).sum().item()*1.0
    # FP    predict 1 label 0
    FP = ((pred == 1) & (label == 0)).sum().item()*1.0
    # pdb.set_trace()
    precision = 0
    f1 = 0
    recall = 0
    acc = 0
    if TP + FP > 0:
        precision = TP / (TP + FP)
    if TP + FN > 0:
        recall = TP / (TP + FN)
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    if TP + TN + FP + FN > 0:
        acc = (TP + TN) / (TP + TN + FP + FN)
    return precision, recall, f1, acc


def set_seed(seed):
    """
    Freeze every seed.
    All about reproducibility
    TODO multiple GPU seed, torch.cuda.all_seed()
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def parse_data():
    data = pd.read_csv('./result/result.csv', sep='\t', header=None)
    hq_data = pd.read_csv('./result/high_quality_result.csv', sep='\t', header=None)
    pdb.set_trace()


if __name__ == '__main__':
    parse_data()