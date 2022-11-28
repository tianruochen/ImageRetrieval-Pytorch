#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :model_const.py.py
# @Time     :2021/12/14 下午4:22
# @Author   :Chang Qing
 

FEATURES = {
    'vgg16': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-vgg16-features-d369c8e.pth',
    'resnet50': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet50-features-ac468af.pth',
    'resnet101': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet101-features-10a101d.pth',
    'resnet152': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet152-features-1011020.pth',
}

# TODO: pre-compute for more architectures and properly test variations (pre l2norm, post l2norm)
# pre-computed local pca whitening that can be applied before the pooling layer
L_WHITENING = {
    'resnet101': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-lwhiten-9f830ef.pth',
    # no pre l2 norm
    # 'resnet101' : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-lwhiten-da5c935.pth', # with pre l2 norm
}

# possible global pooling layers, each on of these can be made regional
POOLING = {
    'mac': MAC,
    'spoc': SPoC,
    'gem': GeM,
    'gemmp': GeMmp,
    'rmac': RMAC,
}

# TODO: pre-compute for: resnet50-gem-r, resnet50-mac-r, vgg16-mac-r, alexnet-mac-r
# pre-computed regional whitening, for most commonly used architectures and pooling methods
R_WHITENING = {
    'alexnet-gem-r': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-alexnet-gem-r-rwhiten-c8cf7e2.pth',
    'vgg16-gem-r': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-vgg16-gem-r-rwhiten-19b204e.pth',
    'resnet101-mac-r': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-mac-r-rwhiten-7f1ed8c.pth',
    'resnet101-gem-r': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-gem-r-rwhiten-adace84.pth',
}

# TODO: pre-compute for more architectures
# pre-computed final (global) whitening, for most commonly used architectures and pooling methods
WHITENING = {
    'alexnet-gem': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-alexnet-gem-whiten-454ad53.pth',
    'alexnet-gem-r': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-alexnet-gem-r-whiten-4c9126b.pth',
    'vgg16-gem': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-vgg16-gem-whiten-eaa6695.pth',
    'vgg16-gem-r': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-vgg16-gem-r-whiten-83582df.pth',
    'resnet50-gem': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet50-gem-whiten-f15da7b.pth',
    'resnet101-mac-r': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-mac-r-whiten-9df41d3.pth',
    'resnet101-gem': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-gem-whiten-22ab0c1.pth',
    'resnet101-gem-r': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-gem-r-whiten-b379c0a.pth',
    'resnet101-gemmp': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-gemmp-whiten-770f53c.pth',
    'resnet152-gem': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet152-gem-whiten-abe7b93.pth',
    'densenet121-gem': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-densenet121-gem-whiten-79e3eea.pth',
    'densenet169-gem': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-densenet169-gem-whiten-6b2a76a.pth',
    'densenet201-gem': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-densenet201-gem-whiten-22ea45c.pth',
}

# output dimensionality for supported architectures
OUTPUT_DIM = {
    'alexnet': 256,
    'vgg11': 512,
    'vgg13': 512,
    'vgg16': 512,
    'vgg19': 512,
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'resnet152': 2048,
    'densenet121': 1024,
    'densenet169': 1664,
    'densenet201': 1920,
    'densenet161': 2208,  # largest densenet
    'squeezenet1_0': 512,
    'squeezenet1_1': 512,
}

