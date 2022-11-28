#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :image_processor.py
# @Time     :2021/12/10 下午8:04
# @Author   :Chang Qing

import os
from PIL import Image


class ImageProcessor:
    def __init__(self, img_dir):
        self.img_dir = img_dir

    def process(self, ratio_limit=0):
        img_list = []
        for root, dir, file_list in os.walk(self.img_dir):
            for file_name in file_list:
                if file_name[-4:].lower() in [".jpg", ".png", "jpeg"]:
                    img_path = os.path.join(root, file_name)
                    img_list.append(img_path)
                    # try:
                    #     image = Image.open(img_path).convert("RGB")
                    #     if ratio_limit and max(image.size) / min(image.size) > ratio_limit:
                    #         continue
                    #     else:
                    #         img_list.append(img_path)
                    # except:
                    #     pass
        return img_list


if __name__ == '__main__':
    image_processor = ImageProcessor("/data1/changqing/ZyImage_Data/image_gallery/")
    image_list = image_processor.process()
    print(image_list[:10])
    print(len(image_list))
