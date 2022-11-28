#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :download_imgs_mp.py
# @Time     :2021/10/27 下午3:29
# @Author   :Chang Qing

# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :imgs_downloader_mp.py
# @Time     :2021/9/3 下午4:11
# @Author   :Chang Qing

import os
import time
import random
import requests
import argparse
import traceback

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool

requests.DEFAULT_RETRIES = 5
s = requests.session()
s.keep_alive = False
random.seed(666)


def download_img(item):
    # name, url = item
    name, url = item.split("\t")
    img_name = os.path.join(imgs_root, f"{name}.jpg")
    if not os.path.exists(img_name):
        try:
            res = requests.get(url, timeout=1)
            if res.status_code != 200:
                raise Exception
            with open(img_name, "wb") as f:
                f.write(res.content)
        except Exception as e:
            print(name, url)
            traceback.print_exc()


def build_url_list(url_file):
    url_list = []
    with open(url_file) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            url_list.append([str(i), line.strip()])
    return url_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Images Download Script")
    parser.add_argument("--img_root", default="/Users/zuiyou/PycharmProjects/Image_Process_Tool/images/生活_股票基金_股票k线", type=str,
                        help="the directory of images")
    parser.add_argument("--workers", default=3, type=int, help="the nums of process")
    args = parser.parse_args()

    imgs_root = args.img_root
    workers = args.workers

    os.makedirs(imgs_root, exist_ok=True)

    url_file = "imgs.txt"
    # url_file = "others_20211214-20211223.txt"
    items = open(url_file).readlines()
    # other类太多，分批处理， 一次处理20000张
    items = [item.strip() for item in items if item][:10000]
    # url_list = build_url_list(url_file)
    print(items[:5])
    print(f"total items: {len(items)}")

    # random.shuffle(url_list)
    # url_list = url_list[:180000]

    tik_time = time.time()
    # create multiprocess pool
    pool = Pool(workers)  # process num: 20

    # 如果check_img函数仅有1个参数，用map方法
    # pool.map(check_img, img_paths)
    # 如果check_img函数有不止1个参数，用apply_async方法
    # for img_path in tqdm(img_paths):
    #     pool.apply_async(check_img, (img_path, False))
    list(tqdm(iterable=(pool.imap(download_img, items)), total=len(items)))
    pool.close()
    pool.join()
    tok_time = time.time()
    print(tok_time - tik_time)




