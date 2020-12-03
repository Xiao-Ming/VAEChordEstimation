#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 20:36:56 2020

@author: wu
"""

import const as C
import numpy as np
import folds
import chord
import training
import matplotlib.pyplot as plt
from net_generative import GenerativeChordnet
import chainer
import os


from librosa.util import find_files

def idx2sec(i):
    return i * C.H / float(C.SR)
path_feat = "/n/work1/wu/ChordData/feats_midi_fullcnn12/16_RWC"
path_est = "est_rwc"

list_feat = find_files(path_feat,ext="npy")[2:3]

model = GenerativeChordnet()
model.load("/home/wu/Projects/ChordRecGenerative/chromavae_full_doublesuper_supervised1_fold0_full.model")


for f_feat in list_feat:
    with chainer.using_config('train', False):
        chainer.config.enable_backprop = False
        path_estfile = os.path.join(path_est,f_feat.split("/")[-1]+".chord")
        labfile = open(path_estfile,"w")
        print("Estimating: %s" % path_estfile)
        
        feat = np.load(f_feat)
        estimated_triad = model.estimate(feat)
        cur_triad = int(estimated_triad[0])
        cur_frame = 0
        for i_frame in range(len(estimated_triad)):
            if estimated_triad[i_frame]==cur_triad:
                continue
            #cur_seventh = np.argmax(estimated_sevenths_prob[cur_frame:i_frame,:].sum(axis=0))
            sign = chord.id2signature(cur_triad)
            line = "%.4f %.4f %s\n" % (idx2sec(cur_frame),idx2sec(i_frame),sign)
            labfile.write(line)
            cur_frame = i_frame
            cur_triad = int(estimated_triad[i_frame])
        
        if cur_frame<i_frame:
            sign = chord.id2signature(cur_triad)
            line = "%.4f %.4f %s" % (idx2sec(cur_frame),idx2sec(i_frame),sign)
            labfile.write(line)
        labfile.close()