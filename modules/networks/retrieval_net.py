#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :retrieval_net.py.py
# @Time     :2021/12/14 下午4:26
# @Author   :Chang Qing

import torch.nn as nn

from modules.layers.normalization import L2N


class ImageRetrievalNet(nn.Module):

    def __init__(self, features, lwhiten, pool, whiten, meta):
        super(ImageRetrievalNet, self).__init__()
        self.features = nn.Sequential(*features)
        self.lwhiten = lwhiten
        self.pool = pool
        self.whiten = whiten
        self.norm = L2N()
        self.meta = meta

    def forward(self, x):
        # x -> features
        o = self.features(x)  # (bs, C, H, W)

        # TODO: properly test (with pre-l2norm and/or post-l2norm)
        # if lwhiten exist: features -> local whiten
        if self.lwhiten is not None:
            # o = self.norm(o)
            s = o.size()
            # (bs,C,H,W) --> (bs, H, W, C) --> (bs*H*W, C)
            o = o.permute(0, 2, 3, 1).contiguous().view(-1, s[1])
            # o = self.norm(o)  pre-l2norm
            o = self.lwhiten(o)  # 本质是一个全连接层， 可以起到降维的作用，但作者这里没有这么用，输入输出都是相同的维度
            # o = self.norm(o)  post-l2norm
            # 还原回原来的shape: (bs, C, H, W)
            o = o.view(s[0], s[2], s[3], self.lwhiten.out_features).permute(0, 3, 1, 2)

        # features -> pool -> norm
        o = self.norm(self.pool(o)).squeeze(-1).squeeze(-1)

        # if whiten exist: pooled features -> whiten -> norm
        if self.whiten is not None:
            o = self.norm(self.whiten(o))

        # permute so that it is Dx1 column vector per image (DxN if many images)
        # (embedding_dim, bs)
        return o.permute(1, 0)

    def __repr__(self):
        tmpstr = super(ImageRetrievalNet, self).__repr__()[:-1]
        tmpstr += self.meta_repr()
        tmpstr = tmpstr + ')'
        return tmpstr

    def meta_repr(self):
        tmpstr = '  (' + 'meta' + '): dict( \n'  # + self.meta.__repr__() + '\n'
        tmpstr += '     architecture: {}\n'.format(self.meta['architecture'])
        tmpstr += '     local_whitening: {}\n'.format(self.meta['local_whitening'])
        tmpstr += '     pooling: {}\n'.format(self.meta['pooling'])
        tmpstr += '     regional: {}\n'.format(self.meta['regional'])
        tmpstr += '     whitening: {}\n'.format(self.meta['whitening'])
        tmpstr += '     outputdim: {}\n'.format(self.meta['outputdim'])
        tmpstr += '     mean: {}\n'.format(self.meta['mean'])
        tmpstr += '     std: {}\n'.format(self.meta['std'])
        tmpstr = tmpstr + '  )\n'
        return tmpstr
