#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 18:04:49 2019

@author: wu
"""


import net_generative
import dataset as D
import const as C
from chainer import config
import numpy as np
import util as U
import matplotlib.pyplot as plt
from chainer.functions import gumbel_softmax
from librosa.display import specshow
import chord
from yellowbrick.features import Manifold


np.random.seed(10)
idx_rand = np.random.permutation(220)
fold = np.load("folds_320files.npy")[0][:5]
idx_billboard = np.load("idx_non_billboard.npy")+320
dset = D.ChordDataset([855,472,1003])

model = net_generative.GenerativeChordnet()
model.load("chromavae_beta_anneal.model")

config.train = False
config.enable_backprop = False


feat,labs,aligns = dset[0]
z = model.generator.getzs([feat[:512]],[labs[aligns[:512]]])[0].data

z_random = np.random.normal(0,1,size=z.shape)

visualizer = Manifold(manifold="tsne")
visualizer.fit_transform(np.concatenate((z,z_random)),y=np.concatenate((np.zeros(len(z)),np.ones(len(z_random)))).astype(np.int32))
visualizer.poof()