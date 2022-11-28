#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :comm_util.py
# @Time     :2021/3/26 上午11:18
# @Author   :Chang Qing

import os
import time
import json
import datetime
import hashlib
import torch
import random
import numpy as np

from collections import OrderedDict


#########################################################################
############################ 用于数值记录 #################################
#########################################################################

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

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
        self.avg = self.sum / self.count


#########################################################################
############################ 时间相关 ####################################
#########################################################################

def get_time_str():
    timestamp = time.time()
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
    return time_str

def get_date_str(n_days_ago=0):
    focus_day = datetime.datetime.today().date() - datetime.timedelta(days=n_days_ago)
    focus_day_str = time.strftime("%Y%m%d", time.strptime(str(focus_day), '%Y-%m-%d'))
    # focus_day_str = time.mktime(time.strptime(str(focus_day), '%Y-%m-%d'))
    return focus_day_str

def format_cost_time(cost_time):
    """
    :param cost_time: 毫秒数, 一般指时间戳之差
    :return: 格式化后，易读的时间字符串
    """
    cost_time = round(cost_time)

    days = cost_time // 86400
    hours = cost_time // 3600 % 24
    minutes = cost_time // 60 % 60
    seconds = cost_time % 60
    if days > 0:
        return '{:d}d {:d}h {:d}m {:d}s'.format(days, hours, minutes, seconds)
    if hours > 0:
        return '{:d}h {:d}m {:d}s'.format(hours, minutes, seconds)
    if minutes > 0:
        return '{:d}m {:d}s'.format(minutes, seconds)
    return '{:d}s'.format(seconds)


#########################################################################
############################ 文件相关 ####################################
#########################################################################

def save_to_json(save_target, save_path):
    json.dump(save_target, fp=open(save_path, "w"), indent=4, ensure_ascii=False)

def save_to_txt(item_list, save_path):
    item_list = [item + "\n" if not item.endswith("\n") else item for item in item_list]
    # print(item_list)
    with open(save_path, "w") as f:
        f.writelines(item_list)

#########################################################################
############################ 路径相关 ####################################
#########################################################################

def get_root():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)))

def get_data_root():
    return os.path.join(get_root(), 'data')


#########################################################################
############################ 其他 #######################################
#########################################################################

def sort_dict(ori_dict, by_key=False, reverse=False):
    """
    sorted dict by key or value
    :param ori_dict:
    :param by_key: sorted by key or value
    :param reverse: if reverse is true, big to small. if false, small to big
    :return: OrderedDict
    """
    ordered_list = sorted(ori_dict.items(), key=lambda item: item[0] if by_key else item[1])
    ordered_list = ordered_list[::-1] if reverse else ordered_list
    new_dict = OrderedDict(ordered_list)
    return new_dict

def build_mapping_from_list(name_list):
    name_list = [name for name in list(set(name_list)) if name]
    idx2name_map = OrderedDict()
    name2idx_map = OrderedDict()
    for idx, name in enumerate(name_list):
        idx2name_map[str(idx)] = name
        name2idx_map[name] = str(idx)
    return idx2name_map, name2idx_map

def sha256_hash(filename, block_size=65536, length=8):
    sha256 = hashlib.sha256()
    with open(filename, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            sha256.update(block)
    return sha256.hexdigest()[:length - 1]


def fix_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

if __name__ == "__main__":
    a = ["1", "2" + "\n", "3", "4" + "\n", "5"]
    # save_to_txt(a, "test.txt")

    ori_dict = {
        "a": 3,
        "c": 1,
        "b": 2,
    }
    print(ori_dict)
    print(sort_dict(ori_dict, by_key=False, reverse=False))
    print(sort_dict(ori_dict, by_key=False, reverse=True))
    print(sort_dict(ori_dict, by_key=True, reverse=False))
    print(sort_dict(ori_dict, by_key=True, reverse=True))
