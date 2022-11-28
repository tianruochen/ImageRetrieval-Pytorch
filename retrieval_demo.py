#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :retrieval_demo.py
# @Time     :2021/12/10 下午7:39
# @Author   :Chang Qing

import os
import shutil
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "8"

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
    parser.add_argument("--image_gallery", type=str, default="data/images", help="image gallery")
    parser.add_argument("--query_number", type=int, default=400, help="query number")

    args = parser.parse_args()

    args.image_gallery = "/data1/changqing/ZyImage_Data/auto_ai_cls8/其他"
    config = parse_config(args.config_path)
    config = merge_config(config, vars(args))

    # 初始化特征提取器
    model = ModelInitializer(checkpoint=config.checkpoint).init_model()
    feature_extractor = FeatureExtractor(model)
    # date_strs = ["20210713", "20210716", "20210726"]
    # for date_str in date_strs:
    #     config.image_gallery = f"/data1/changqing/ZyImage_Data/image_gallery/{date_str}/imgs/"
    #     config.feature_path = f"/data1/changqing/ZyImage_Data/image_gallery/{date_str}/features_dim-2048.pkl"
    #     config.lsh_index_path = f"/data1/changqing/ZyImage_Data/image_gallery/{date_str}/lsh_hash-size-00_input-idm-2048.pkl"
    #     gallery_feature_dict, lsh = feature_extractor.extract(img_path=config.image_gallery,
    #                                                           lsh_config=config.lsh_config,
    #                                                           feature_path=config.feature_path,
    #                                                           index_path=config.lsh_index_path)


    config.feature_path = ""
    # config.lsh_index_path = ""
    # gallery_feature_dict, lsh = feature_extractor.extract(img_path=config.image_gallery,
    #                                                       lsh_config=config.lsh_config,
    #                                                       feature_path=config.feature_path,
    #                                                       index_path=config.lsh_index_path)

    # 提取query images的特征
    test_feature_dict, _ = feature_extractor.extract(img_path=config.query_image)

    print("=" * 60)

    # retrieval test
    similar_img_dict, similar_img_list = ImageRetriever(lsh_path=config.lsh_index_path).retrieval(test_feature_dict,
                                        num_results=config.query_number, threshold=-1)
    print(similar_img_dict)
    print(similar_img_list)
    print("=" * 60)

    print(">>> copying...")
    if config.out_similar_dir and os.path.isdir(config.out_similar_dir):
        for (similar_img_path, similarity_score) in tqdm(similar_img_list):
            basename = os.path.basename(similar_img_path)
            new_path = os.path.join(config.out_similar_dir, basename)
            if os.path.exists(similar_img_path):
                shutil.copy(similar_img_path, new_path)

    # print("done...")
