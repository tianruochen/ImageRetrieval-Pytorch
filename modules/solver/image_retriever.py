#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :image_retriever.py
# @Time     :2021/12/14 下午9:45
# @Author   :Chang Qing

import pickle
import traceback

class ImageRetriever:
    def __init__(self, lsh_path=None):
        self.lsh_path = lsh_path

    def retrieval(self, feature_dict, lsh=None, num_results=3, limit_threshold=False, threshold=1):
        try:
            if not lsh:
                print(f">>> load lsh from: {self.lsh_path}")
                lsh = pickle.load(open(self.lsh_path, "rb"))
                print(">>> load lsh done...")
        except:
            traceback.print_exc()
            print("load lsh model error")
            return
        similar_img_dict = dict()
        similar_img_dict2 = dict()
        similar_img_list = []
        for query_path, query_feature in feature_dict.items():
            try:
                # res: (((与query_feature相似的特征向量, 图片路径), 距离得分),
                #       ((与query_feature相似的特征向量, 图片路径), 距离得分)...)
                res = lsh.query(query_feature.flatten(), num_results=int(num_results), distance_func="cosine")
                queried_paths = []
                # print(len(res))
                for i in range(min(num_results, len(res))):
                    queried_path = res[i][0][1]
                    similarity_score = res[i][1]
                    if limit_threshold and similarity_score > threshold:
                        continue
                    queried_paths.append(queried_path)
                    if queried_path in similar_img_dict2 and similar_img_dict2[queried_path] < similarity_score:
                        continue
                    else:
                        similar_img_dict2[queried_path] = similarity_score
                    # similar_img_list.append((queried_path, similarity_score))
                similar_img_dict[query_path] = queried_paths
            except:
                traceback.print_exc()
        # similar_img_list = sorted(similar_img_list, key=lambda item: item[1])[:num_results]
        similar_img_list = sorted(similar_img_dict2.items(), key=lambda item: item[1])[:num_results]
        return similar_img_dict, similar_img_list
