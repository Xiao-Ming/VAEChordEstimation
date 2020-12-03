#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 19:27:10 2019

@author: wuyiming
"""

import argparse
import training
import numpy as np
import const as C

C.PATH_ESTIMATE_CROSS = "estimate"

parser = argparse.ArgumentParser()
parser.add_argument('-d','--device',\
                    action='store',\
                    default=0,
                    type=int)
parser.add_argument('-e','--epoch',\
                    action='store',
                    default=50,
                    type=int)
parser.add_argument('-i','--loginterval',\
                    action='store',
                    default=1,
                    type=int)
parser.add_argument('-s','--save',\
                    action='store',
                    default="chromavae.model",
                    type=str)

args = parser.parse_args()
device = args.device
epoch = args.epoch
log_interval = args.loginterval
save_model = args.save


np.random.seed(20)
idx_rand = np.random.permutation(320)
idx_ext = np.load("idx_non_billboard.npy")+320
idx_train = np.concat((idx_rand[50:],idx_ext))
idx_test = idx_rand[:50]

training.TrainChromaVAE(idx_train,idx_test,device,epoch,log_interval,save_model=save_model)
