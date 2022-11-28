#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :retrieval_demo.py
# @Time     :2021/12/10 下午7:39
# @Author   :Chang Qing

import os
import pickle
import shutil
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tqdm import tqdm
from modules.solver import ModelInitializer
from modules.solver import FeatureExtractor
from modules.solver import ImageRetriever
from utils.config_util import parse_config, merge_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image Retrieval Script")
    parser.add_argument("--config_path", type=str, default="configs/retrieval.yaml",
                        help="config file path for image retrieval")
    parser.add_argument("--query_image", type=str, help="query image (single image, dir, or image file)")
    parser.add_argument("--query_number", type=int, default=150, help="query number")

    args = parser.parse_args()

    config = parse_config(args.config_path)
    config = merge_config(config, vars(args))

    # 初始化特征提取器
    model = ModelInitializer(checkpoint=config.checkpoint).init_model()
    feature_extractor = FeatureExtractor(model)

    # 提取query images的特征
    test_feature_dict, _ = feature_extractor.extract(img_path=config.query_image)

    # lsh_paths = config.lsh_paths.hash_size_zero
    lsh_paths = config.lsh_paths.hash_size_eight
    for lsh_path in lsh_paths:
        lsh = pickle.load(open(lsh_path, "rb"))
        print("=" * 60)
        # retrieval test
        similar_img_dict, similar_img_list = ImageRetriever().retrieval(test_feature_dict,
                                        lsh, num_results=config.query_number, threshold=0.7)
        print(similar_img_dict)
        print(similar_img_list)

        print(">>> copying...")
        if config.out_similar_dir and os.path.isdir(config.out_similar_dir):
            for (similar_img_path, similarity_score) in tqdm(similar_img_list):
                basename = os.path.basename(similar_img_path)
                new_path = os.path.join(config.out_similar_dir, basename)
                shutil.copy(similar_img_path, new_path)

    print("done...")
