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

foldman = folds.FoldManager()
est_path1 = "/home/wu/Projects/ChordRecGenerative/chromavae_nomarkov_estimated_cross_mix3"
est_path2 = "estimated_cross_full"

model_path1 = "chromavae_fullprob_supervised1_fold%d_full.model"
model_path2 = "chromavae_full_supervised1_fold%d_full.model"

freq1_folds = []
freq2_folds = []

accr1_folds = []
accr2_folds = []

ent1_folds = []
ent2_folds = []

C.POSTFILTER=True

for f in range(foldman.nfolds):
    print("fold:%d" % f)
    #idx1 = foldman.getTrainSupervisedFold(f,3)
    #idx2 = foldman.getTrainSupervisedFold(f)
    idx1 = foldman.getTestFold(f)
    idx2 = foldman.getTestFold(f)
    
    print("Estimating 1...")
    C.PATH_ESTIMATE_CROSS = est_path1
    #training.Estimate(idx1,model_path1 % f, 0, verbose=False)
    mean_ent1 = training.Accum_Entropy(idx1,model_path1 % f, 0)
    ent1_folds.append(mean_ent1)
    _,accrs,_,durations,_,_=chord.EvaluateLabels(idx1)
    freq1 = chord.TransitionsFreq(idx1)
    freq1_folds.append(freq1)
    accr1_folds.append(np.sum(accrs*durations)/np.sum(durations))
    
    print("Estimating 2...")
    C.PATH_ESTIMATE_CROSS = est_path2
    #training.Estimate(idx2,model_path2 % f, 0, verbose=False)
    mean_ent2 = training.Accum_Entropy(idx2,model_path2 % f, 0)
    ent2_folds.append(mean_ent2)
    _,accrs,_,durations,_,_=chord.EvaluateLabels(idx2)
    freq2 = chord.TransitionsFreq(idx2)
    freq2_folds.append(freq2)
    accr2_folds.append(np.sum(accrs*durations)/np.sum(durations))

C.PATH_ESTIMATE_CROSS = C.PATH_CHORDLAB
freq_ref = chord.TransitionsFreq(foldman.getAll())

print("Transition freqs")
print(freq1_folds)
print(freq2_folds)
print(freq_ref)

print("Accrs:")
print(np.mean(accr1_folds))
print(np.mean(accr2_folds))

print("Entropy:")
print(ent1_folds)
print(ent2_folds)


plt.bar(np.arange(3),[freq_ref,np.mean(freq1_folds),np.mean(freq2_folds)],tick_label=["annotation","732+244(semi-supervise)","976+0(full supervise)"],width=0.3)
plt.ylabel("transition freq (times/min)")