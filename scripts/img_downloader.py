#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :img_downloader.py
# @Time     :2021/12/14 下午2:22
# @Author   :Chang Qing

import os
import json
import requests
import argparse
import traceback

from tqdm import tqdm
from PIL import Image
from io import BytesIO
from multiprocessing import Pool


def download_img(item):
    pid, img_id, url = item
    img_name = os.path.join(img_save_dir, f"{pid}_{img_id}.jpg")
    if not os.path.exists(img_name):
        try:
            res = requests.get(url, timeout=1)
            img = Image.open(BytesIO(res.content)).convert("RGB")
            img.verify()
            if res.status_code != 200:
                raise Exception
            with open(img_name, "wb") as f:
                f.write(res.content)
        except Exception as e:
            print(pid, img_id, url)
            traceback.print_exc()


def build_task_list(data_info):
    task_list = []
    for pid, inv_info in data_info.items():
        img_id2img_info = inv_info["img_id2img_info"]
        for img_id, img_info in img_id2img_info.items():
            url = img_info["image_url"]
            task_list.append((pid, img_id, url))
    return task_list


def build_task_list2(data_info):
    task_list = []
    for name, url in data_info.items():
        pid, img_id = name.split("_")
        url = f"http://tbfile.ixiaochuan.cn/img/view/id/{img_id}"
        task_list.append((pid, img_id, url))
    return task_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Zuiyou Image Download Script")
    parser.add_argument("--img_root", type=str, default="/data1/changqing/ZyImage_Data/image_gallery")
    parser.add_argument("--date_str", type=str, default="20211231", help="the date str of images")
    parser.add_argument("--workers", type=int, default=4, help="the nums of process")

    args = parser.parse_args()

    workers = args.workers
    img_root = args.img_root
    date_str = args.date_str
    img_save_dir = os.path.join(img_root, date_str, "imgs")
    os.makedirs(img_save_dir, exist_ok=True)
    print(img_save_dir)
    img_info_path = os.path.join(img_root, date_str, f"imgtag_name2url_{date_str}.json")

    img_info = json.load(open(img_info_path))
    # task_list = build_task_list(img_info)
    task_list = build_task_list2(img_info)

    pool = Pool(workers)
    list(tqdm(iterable=(pool.imap(download_img, task_list)), total=len(task_list)))
    pool.close()  # 调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool
    pool.join()   # join函数等待所有子进程结束
    print("Done...")