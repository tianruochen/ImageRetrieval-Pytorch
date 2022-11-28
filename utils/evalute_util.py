#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :evalute_util.py
# @Time     :2021/12/14 下午9:13
# @Author   :Chang Qing
 
'''
Author: yinhao
Email: yinhao_x@163.com
Wechat: xss_yinhao
Github: http://github.com/yinhaoxs
data: 2019-11-23 18:27
desc:
'''

import os
import shutil
import numpy as np
import pandas as pd


class EvaluteMap():
    def __init__(self, out_similar_dir='', out_similar_file_dir='', all_csv_file='', feature_path='', index_path=''):
        self.out_similar_dir = out_similar_dir
        self.out_similar_file_dir = out_similar_file_dir
        self.all_csv_file = all_csv_file
        self.feature_path = feature_path
        self.index_path = index_path


    def get_dict(self, query_no, query_id, simi_no, simi_id, num, score):
        new_dict = {
            'index': str(num),
            'id1': str(query_id),
            'id2': str(simi_id),
            'no1': str(query_no),
            'no2': str(simi_no),
            'score': score
        }
        return new_dict


    def find_similar_img_gyz(self, feature_dict, lsh, num_results):
        for q_path, q_vec in feature_dict.items():
            try:
                response = lsh.query(q_vec.flatten(), num_results=int(num_results), distance_func="cosine")
                query_img_path0 = response[0][0][1]
                query_img_path1 = response[1][0][1]
                query_img_path2 = response[2][0][1]
                # score0 = response[0][1]
                # score0 = np.rint(100 * (1 - score0))
                print('**********************************************')
                print('input img: {}'.format(q_path))
                print('query0 img: {}'.format(query_img_path0))
                print('query1 img: {}'.format(query_img_path1))
                print('query2 img: {}'.format(query_img_path2))
            except:
                continue


    def find_similar_img(self, feature_dict, lsh, num_results):
        num = 0
        result_list = list()
        for q_path, q_vec in feature_dict.items():
            response = lsh.query(q_vec.flatten(), num_results=int(num_results), distance_func="cosine")
            s_path_list, s_vec_list, s_id_list, s_no_list, score_list = list(), list(), list(), list(), list()
            q_path = q_path[0]
            q_no, q_id = q_path.split("\\")[-2], q_path.split("\\")[-1]
            try:
                for i in range(int(num_results)):
                    s_path, s_vec = response[i][0][1], response[i][0][0]
                    s_path = s_path[0]
                    s_no, s_id = s_path.split("\\")[-2], s_path.split("\\")[-1]
                    if str(s_no) != str(q_no):
                        score = np.rint(100 * (1 - response[i][1]))
                        s_path_list.append(s_path)
                        s_vec_list.append(s_vec)
                        s_id_list.append(s_id)
                        s_no_list.append(s_no)
                        score_list.append(score)
                    else:
                        continue

                if len(s_path_list) != 0:
                    index = score_list.index(max(score_list))
                    s_path, s_vec, s_id, s_no, score = s_path_list[index], s_vec_list[index], s_id_list[index], \
                                                       s_no_list[index], score_list[index]
                else:
                    s_path, s_vec, s_id, s_no, score = None, None, None, None, None
            except:
                s_path, s_vec, s_id, s_no, score = None, None, None, None, None

            try:
                ##拷贝文件到指定文件夹
                num += 1
                des_path = os.path.join(self.out_similar_dir, str(num))
                if not os.path.exists(des_path):
                    os.makedirs(des_path)
                shutil.copy(q_path, des_path)
                os.rename(os.path.join(des_path, q_id), os.path.join(des_path, "query_" + q_no + "_" + q_id))
                if s_path != None:
                    shutil.copy(s_path, des_path)
                    os.rename(os.path.join(des_path, s_id), os.path.join(des_path, s_no + "_" + s_id))

                new_dict = self.get_dict(q_no, q_id, s_no, s_id, num, score)
                result_list.append(new_dict)
            except:
                continue

        try:
            result_s = pd.DataFrame.from_dict(result_list)
            result_s.to_csv(self.all_csv_file, encoding="gbk", index=False)
        except:
            print("write error")


    def filter_gap_score(self):
        for value in range(90, 101):
            try:
                pd_df = pd.read_csv(self.all_csv_file, encoding="gbk", error_bad_lines=False)
                pd_tmp = pd_df[pd_df["score"] == int(value)]
                if not os.path.exists(self.out_similar_file_dir):
                    os.makedirs(self.out_similar_file_dir)

                try:
                    results_split_csv = os.path.join(self.out_similar_file_dir + os.sep,
                                                     "filter_{}.csv".format(str(value)))
                    pd_tmp.to_csv(results_split_csv, encoding="gbk", index=False)
                except:
                    print("write part error")

                lines = pd_df[pd_df["score"] == int(value)]["index"]
                num = 0
                for line in lines:
                    des_path_temp = os.path.join(self.out_similar_file_dir + os.sep, str(value), str(line))
                    if not os.path.exists(des_path_temp):
                        os.makedirs(des_path_temp)
                    pairs_path = os.path.join(self.out_similar_dir + os.sep, str(line))
                    for img_id in os.listdir(pairs_path):
                        img_path = os.path.join(pairs_path + os.sep, img_id)
                        shutil.copy(img_path, des_path_temp)
            except:
                print("error")


    def retrieval_images(self, feature_dict, lsh, num_results=1):
        # load model
        # with open(self.feature_path, "rb") as f:
        #     feature_dict = pickle.load(f)
        # with open(self.index_path, "rb") as f:
        #     lsh = pickle.load(f)

        self.find_similar_img_gyz(feature_dict, lsh, num_results)
        # self.filter_gap_score()


if __name__ == "__main__":
    pass


