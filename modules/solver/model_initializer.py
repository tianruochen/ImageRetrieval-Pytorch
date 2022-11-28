#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :model_initializer.py
# @Time     :2021/12/14 上午10:26
# @Author   :Chang Qing

import os
import torch
import torchvision
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from modules.layers.pooling import Rpool
from modules.networks import ImageRetrievalNet
from modules import OUTPUT_DIM, FEATURES, POOLING
from modules import L_WHITENING, WHITENING, R_WHITENING
from utils.common_util import get_data_root


class ModelInitializer:
    def __init__(self, model_params=None, checkpoint=None):
        assert model_params or checkpoint
        self.checkpoint = checkpoint
        self.model_params = self._parse_params_from_checkpoint() if self.checkpoint else model_params
        self.architecture = self.model_params.get('architecture', 'resnet101')
        self.local_whitening = self.model_params.get('local_whitening', False)
        self.pooling = self.model_params.get('pooling', 'gem')
        self.regional = self.model_params.get('regional', False)
        self.whitening = self.model_params.get('whitening', False)
        self.mean = self.model_params.get('mean', [0.485, 0.456, 0.406])
        self.std = self.model_params.get('std', [0.229, 0.224, 0.225])
        self.pretrained = self.model_params.get('pretrained', True)
        self.state_dict = self.model_params.get("state_dict", None)
        self.dim = OUTPUT_DIM[self.architecture]

    def _parse_params_from_checkpoint(self):
        state = torch.load(self.checkpoint)
        # parsing net params from meta
        # architecture, pooling, mean, std required
        # the rest has default values, in case that is doesnt exist
        model_params = {}
        model_params['architecture'] = state['meta']['architecture']
        model_params['pooling'] = state['meta']['pooling']
        model_params['local_whitening'] = state['meta'].get('local_whitening', False)
        model_params['regional'] = state['meta'].get('regional', False)
        model_params['whitening'] = state['meta'].get('whitening', False)
        model_params['mean'] = state['meta']['mean']
        model_params['std'] = state['meta']['std']
        model_params['pretrained'] = False
        model_params["state_dict"] = state["state_dict"]
        return model_params

    def build_network(self):
        # loading network from torchvision
        if self.pretrained:
            if self.architecture not in FEATURES:
                # initialize with network pretrained on imagenet in pytorch
                net_in = getattr(torchvision.models, self.architecture)(pretrained=True)
            else:
                # initialize with random weights, later on we will fill features with custom pretrained network
                net_in = getattr(torchvision.models, self.architecture)(pretrained=False)
        else:
            # initialize with random weights
            net_in = getattr(torchvision.models, self.architecture)(pretrained=False)

        # initialize features
        # take only convolutions for features,
        # always ends with ReLU to make last activations non-negative
        if self.architecture.startswith('alexnet'):
            features = list(net_in.features.children())[:-1]
        elif self.architecture.startswith('vgg'):
            features = list(net_in.features.children())[:-1]
        elif self.architecture.startswith('resnet'):
            features = list(net_in.children())[:-2]
        elif self.architecture.startswith('densenet'):
            features = list(net_in.features.children())
            features.append(nn.ReLU(inplace=True))
        elif self.architecture.startswith('squeezenet'):
            features = list(net_in.features.children())
        else:
            raise ValueError('Unsupported or unknown architecture: {}!'.format(self.architecture))

        # initialize local whitening
        if self.local_whitening:
            lwhiten = nn.Linear(self.dim, self.dim, bias=True)
            # TODO: lwhiten with possible dimensionality reduce

            if self.pretrained:
                lw = self.architecture
                if lw in L_WHITENING:
                    print(">> {}: for '{}' custom computed local whitening '{}' is used"
                          .format(os.path.basename(__file__), lw, os.path.basename(L_WHITENING[lw])))
                    whiten_dir = os.path.join(get_data_root(), 'whiten')
                    lwhiten.load_state_dict(model_zoo.load_url(L_WHITENING[lw], model_dir=whiten_dir))
                else:
                    print(">> {}: for '{}' there is no local whitening computed, random weights are used"
                          .format(os.path.basename(__file__), lw))

        else:
            lwhiten = None

        # initialize pooling
        if self.pooling == 'gemmp':
            pool = POOLING[self.pooling](mp=self.dim)
        else:
            pool = POOLING[self.pooling]()

        # initialize regional pooling
        if self.regional:
            rpool = pool
            rwhiten = nn.Linear(self.dim, self.dim, bias=True)
            # TODO: rwhiten with possible dimensionality reduce

            if self.pretrained:
                rw = '{}-{}-r'.format(self.architecture, self.pooling)
                if rw in R_WHITENING:
                    print(">> {}: for '{}' custom computed regional whitening '{}' is used"
                          .format(os.path.basename(__file__), rw, os.path.basename(R_WHITENING[rw])))
                    whiten_dir = os.path.join(get_data_root(), 'whiten')
                    rwhiten.load_state_dict(model_zoo.load_url(R_WHITENING[rw], model_dir=whiten_dir))
                else:
                    print(">> {}: for '{}' there is no regional whitening computed, random weights are used"
                          .format(os.path.basename(__file__), rw))

            pool = Rpool(rpool, rwhiten)

        # initialize whitening
        if self.whitening:
            whiten = nn.Linear(self.dim, self.dim, bias=True)
            # TODO: whiten with possible dimensionality reduce

            if self.pretrained:
                w = self.architecture
                if self.local_whitening:
                    w += '-lw'
                w += '-' + self.pooling
                if self.regional:
                    w += '-r'
                if w in WHITENING:
                    print(">> {}: for '{}' custom computed whitening '{}' is used"
                          .format(os.path.basename(__file__), w, os.path.basename(WHITENING[w])))
                    whiten_dir = os.path.join(get_data_root(), 'whiten')
                    whiten.load_state_dict(model_zoo.load_url(WHITENING[w], model_dir=whiten_dir))
                else:
                    print(">> {}: for '{}' there is no whitening computed, random weights are used"
                          .format(os.path.basename(__file__), w))
        else:
            whiten = None

        # create meta information to be stored in the network
        meta = {
            'architecture': self.architecture,
            'local_whitening': self.local_whitening,
            'pooling': self.pooling,
            'regional': self.regional,
            'whitening': self.whitening,
            'mean': self.mean,
            'std': self.std,
            'outputdim': self.dim,
        }

        # create a generic image retrieval network
        net = ImageRetrievalNet(features, lwhiten, pool, whiten, meta)

        # initialize features with custom pretrained network if needed
        if self.pretrained and self.architecture in FEATURES:
            print(">> {}: for '{}' custom pretrained features '{}' are used"
                  .format(os.path.basename(__file__), self.architecture, os.path.basename(FEATURES[self.architecture])))
            model_dir = os.path.join(get_data_root(), 'networks')
            net.features.load_state_dict(model_zoo.load_url(FEATURES[self.architecture], model_dir=model_dir))

        return net

    def init_model(self):
        # 1.build network
        print(f">>> build network and load weights: \n"
              f"    ... network arch: {self.architecture}\n"
              f"    ... weights path: {self.checkpoint}")
        net = self.build_network()
        # 2.load weights
        if self.state_dict:
            net.load_state_dict(self.state_dict)
        # 3.move to cuda
        if torch.cuda.is_available():
            net.cuda()
        net.eval()
        print(f">>> init model done...")
        return net
