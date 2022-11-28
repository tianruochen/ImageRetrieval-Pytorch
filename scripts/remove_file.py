#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :remove_file.py
# @Time     :2022/4/14 下午9:16
# @Author   :Chang Qing
 

import os
from utils.common_util import save_to_txt

file_names = os.listdir("../data/output")
print(file_names)
print(len(file_names))
dir_path = "/data1/changqing/ZyImage_Data/auto_ai_cls8/其他"
for file_name in file_names:

    file_path = os.path.join(dir_path, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"remove {file_path}")

save_to_txt(file_names, os.path.join(dir_path, "removed6.txt"))


