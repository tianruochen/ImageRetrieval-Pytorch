#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :feature_extractor.py
# @Time     :2021/12/10 下午8:02
# @Author   :Chang Qing

import os
import pickle

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from tqdm import tqdm
from torch.utils.data import DataLoader

from modules.lshash import LSHash
from modules.datasets.generic_dataset import ImagesFromPathList
from utils.image_processor import ImageProcessor


class FeatureExtractor:
    def __init__(self, model, img_resize=1024):
        self.model = model
        self.img_resize = img_resize
        self.tfms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.model.meta['mean'],
                std=self.model.meta['std']
            )
        ])

    def _parse_img_path(self, img_path):
        if os.path.isdir(img_path):
            return ImageProcessor(img_path).process()
        else:
            return [img_path]

    def _extract_ss(self, input_tensor):
        return self.model(input_tensor).cpu().data.squeeze()

    def _extract_ms(self, input_tensor, ms, msp):
        v = torch.zeros(self.model.meta['outputdim'])

        for s in ms:
            if s == 1:
                input_t = input_tensor.clone()
            else:
                input_t = nn.functional.interpolate(input, scale_factor=s, mode='bilinear', align_corners=False)
            v += self.model(input_t).pow(msp).cpu().data.squeeze()

        v /= len(ms)
        v = v.pow(1. / msp)
        v /= v.norm()
        return v

    def extract(self, img_path, feature_path="", lsh_config=None, index_path="", multi_scale=None, msp=1):
        # build image path list
        if multi_scale is None:
            multi_scale = [1]
        print(f">>> build dataset and dataloader: \n"
              f"    ... image_path: {img_path}")
        img_list = self._parse_img_path(img_path)
        # print(img_list)
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

        # create data loader
        dataset = ImagesFromPathList(path_list=img_list, img_resize=self.img_resize, transform=self.tfms)
        del img_list
        loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

        print(f">>> done... image nums: {len(dataset)}")

        print(f">>> extract features:")
        # extract feature vectors
        feature_vectors = torch.zeros(self.model.meta["outputdim"], len(dataset))
        img_paths = list()
        with torch.no_grad():
            for i, (input_tensor, img_path) in enumerate(tqdm(loader, total=len(loader))):
                if torch.cuda.is_available():
                    input_tensor = input_tensor.cuda()
                if len(multi_scale) == 1 and multi_scale[0] == 1:
                    # feature_vector = self._extract_ss(input_tensor)
                    feature_vectors[:, i] = self._extract_ss(input_tensor)
                else:
                    # feature_vector = self._extract_ms(input_tensor)
                    feature_vectors[:, i] = self._extract_ms(input_tensor, multi_scale, msp)
                # feature_dict[img_path] = feature_vector.detach().cpu().numpy()
                img_paths.extend(img_path)

        feature_dict = dict(zip(img_paths, list(feature_vectors.detach().cpu().numpy().T)))
        # feature_dict = dict(zip(map(tuple, img_paths), list(feature_vectors.detach().cpu().numpy().T)))
        if feature_path:
            with open(feature_path, "wb") as f:
                pickle.dump(feature_dict, f)
            print(f"    ... saved features to: {feature_path}")

        # build lsh index
        lsh = None
        if lsh_config:
            hash_size = lsh_config.get("hash_size", 0)
            input_dim = lsh_config.get("input_dim", 2048)
            num_hash_tables = lsh_config.get("num_hash_tables", 1)
            # lsh = LSHash(**self.lsh_config)
            lsh = LSHash(hash_size=int(hash_size), input_dim=int(input_dim), num_hashtables=num_hash_tables)
            for img_path, vec in feature_dict.items():
                # 使用flatten展成1维向量
                lsh.index(vec.flatten(), extra_data=img_path)
            if index_path:
                with open(index_path, "wb") as f:
                    pickle.dump(lsh, f)
                print(f"    ... saved lsh_info to: {index_path}")

        print(">>> extract feature done...")
        return feature_dict, lsh
