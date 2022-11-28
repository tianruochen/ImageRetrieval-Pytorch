#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :config_util.py
# @Time     :2021/3/26 上午11:18
# @Author   :Chang Qing

import json
import yaml

from typing import Any

__all__ = ["parse_config", "merge_config", "print_config"]


class AttrDict(dict):
    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __getattr__(self, key):
        return self[key]


def recursive_convert(attr_dict):
    if not isinstance(attr_dict, dict):
        return attr_dict
    obj_dict = AttrDict()
    for key, value in attr_dict.items():
        obj_dict[key] = recursive_convert(value)
    return obj_dict


def parse_config(cfg_file):
    with open(cfg_file, "r") as f:
        # == AttrDict(yaml.load(f.read()))
        attr_dict_conf = AttrDict(yaml.load(f, Loader=yaml.Loader))
    obj_dict_conf = recursive_convert(attr_dict_conf)
    return obj_dict_conf


def merge_config(cfg, args_dict):
    for key, value in args_dict.items():
        if not value:
            continue
        try:
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        except Exception as e:
            pass
    return cfg


def print_config(config):
    try:
        print(json.dumps(config, indent=4))
    except:
        print(json.dumps(config.__dict__, indent=4))


if __name__ == '__main__':
    # temp_config = {'task_name': 'test', 'task_type': 'multi_class', 'n_gpus': 2,
    #               'id2name': 'tasks/test/data/id2name.json', 'arch_type': 'efficentnet_b5', 'num_classes': 4,
    #               'train_file': 'tasks/test/data/train.txt', 'valid_file': 'tasks/test/data/valid.txt', 'batch_size': 4,
    #               'epochs': 2, 'save_dir': 'tasks/test/workshop'}
    temp_config = {"batch_size": 1}
    train_config = parse_config("../configs/model_config/imgtag_multi_class_train.yaml")
    print(train_config)
    new_config = merge_config(train_config, temp_config)
    print(new_config)
