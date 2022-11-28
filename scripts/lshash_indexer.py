#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :lshash_indexer.py
# @Time     :2021/12/16 下午9:42
# @Author   :Chang Qing

import os
import pickle

from tqdm import tqdm
from modules.lshash import LSHash
from utils.common_util import fix_seed

feature_root = "/data1/changqing/ZyImage_Data/image_gallery"
date_strs = os.listdir(feature_root)

fix_seed()
hash_size = 8
input_dim = 2048
num_hash_tables = 1

for date_str in tqdm(date_strs):
    lsh_indexer = LSHash(hash_size=hash_size, input_dim=input_dim, num_hashtables=num_hash_tables)
    feature_path = f"{feature_root}/{date_str}/features_dim-2048.pkl"
    if os.path.isfile(feature_path):
        feature_dict = pickle.load(open(feature_path, "rb"))
        lsh_indexer_path = f"{feature_root}/{date_str}/" +\
                           "lsh_indexer-size-{:0>2}_input-idm-{:0>4}.pkl".format(hash_size, input_dim)
        if os.path.exists(lsh_indexer_path):
            continue
        for img_path, vec in feature_dict.items():
            # 使用flatten展成1维向量
            lsh_indexer.index(vec.flatten(), extra_data=img_path)
        pickle.dump(lsh_indexer, open(lsh_indexer_path, "wb"))

# lsh_indexer = pickle.load(open("/data1/changqing/ZyImage_Data/image_gallery/20211117/lsh_indexer-size-08_input-idm-2048.pkl", "rb"))
# print(lsh_indexer)