#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :test.py
# @Time     :2021/12/15 下午3:39
# @Author   :Chang Qing
import os
from utils.common_util import save_to_json
img_root = "/data1/changqing/ZyImage_Data/image_gallery"

date_strs = ["20211227", "20211228", "20211229", "20211230", "20211231",
             "20220101", "20220102", "20220103", "20220104", "20220105"]
data_file = "/data1/changqing/ZyImage_Data/image_gallery/pid_imgid_urls.txt"
data_root = "/data1/changqing/ZyImage_Data/image_gallery"
lines = open(data_file).readlines()


# for i, date_str in enumerate(date_strs):
#     temp_lines = lines[i * 80000: (i+1) * 80000]
#     data_dir = os.path.join(data_root, date_str)
#     imgtag_name2url = dict()
#     for line in temp_lines:
#         if not line:
#             continue
#         name, url = line.strip().split("\t")
#         imgtag_name2url[name] = url
#     save_dir = os.path.join(data_dir, f"imgtag_name2url_{date_str}.json")
#     save_to_json(imgtag_name2url, save_dir)

