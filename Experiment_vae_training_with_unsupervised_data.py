#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 13:07:19 2019

@author: wu
"""


import argparse
import training
import numpy as np
import const as C
import chord
from folds import FoldManager


parser = argparse.ArgumentParser()
parser.add_argument('-d','--device',\
                    action='store',\
                    default=-1,
                    type=int)
parser.add_argument('-e','--epoch',\
                    action='store',
                    default=100,
                    type=int)
parser.add_argument('-i','--loginterval',\
                    action='store',
                    default=100,
                    type=int)
parser.add_argument('-s','--save',\
                    action='store',
                    default="chromavae_randcut",
                    type=str)
parser.add_argument('-a','--appendsize',\
                    action='store',
                    default=100,
                    type=int)
parser.add_argument('-f','--folds',
                    action='store',
                    nargs='+',
                    type=int,
                    default=[0,1,2,3,4])
parser.add_argument('--markov',\
                    action='store',
                    default=1,
                    type=int)
parser.add_argument('--shift',
                    action='store',
                    default=0,
                    type=int)
parser.add_argument('--postfilter',
                    action='store',
                    default=1,
                    type=int)
parser.add_argument('-b','--beta',
                    action='store',
                    default=1.0,
                    type=float)
parser.add_argument('--entropy',
                    action='store',
                    default=1.0,
                    type=float)
parser.add_argument('--encregular',
                    action='store',
                    default=0,
                    type=int)
parser.add_argument('--vamp',
                    action='store',
                    default=0,
                    type=int)
parser.add_argument('--doublesuper',
                    action='store',
                    default=0,
                    type=int)
parser.add_argument('--selftrans',\
                action='store',
                default=0.9,
                type=float)  

args = parser.parse_args()
device = args.device
epoch = args.epoch
log_interval = args.loginterval
save_model = args.save
size_semi = args.appendsize
folds_idx = args.folds
doublesuper = (args.doublesuper>0)


C.MARKOV_REGULARIZE = (args.markov > 0)
C.VAE_RAND_SHIFT = (args.shift == 1)
C.VAE_RAND_SHIFT_QUALITY = (args.shift == 2)
C.POSTFILTER = (args.postfilter > 0)
C.VAE_BETA = args.beta
C.VAE_WEIGHT_ENTROPY = args.entropy
C.VAE_SHIFT_REGULAR = (args.encregular > 0)
C.PSEUDO_PRIOR = (args.vamp>0)
C.SELF_TRANS_RATE = args.selftrans

print("Folds:%s" % folds_idx)
print("Vamp Prior:%s" % C.PSEUDO_PRIOR)
print("Generator Random Shift:%s" % C.VAE_RAND_SHIFT)
print("VAE Latent size:%d" % C.N_DIMS_VAE_LATENT)
print("Markov regularize:%s" % C.MARKOV_REGULARIZE)

foldman = FoldManager()
#idx_unlabel = np.random.permutation(1034)[:size_semi] + 1537
#idx_unlabel = np.load("idx_non_billboard.npy") + 320
idx_unlabel = np.arange(size_semi) + 1537

print("Result path: %s" % C.PATH_ESTIMATE_CROSS)
print("Total folds: %d" % foldman.nfolds)

scores = np.zeros((2,1),dtype=np.float32)

list_transition_rate = []

for i in range(1):
    print("Iteration:%d" % (i+1))
    estimate_path = "estimated_cross_fullsemi_plus%d" % size_semi
    
    for f in range(foldman.nfolds):
        if f in folds_idx:
            print("FOLD:%d" % (f+1))
            idx_test = foldman.getTestFold(f)
            idx_train = foldman.getTrainSupervisedFold(f)
            if doublesuper:
                idx_unlabel = np.concatenate((idx_train,idx_unlabel))
            log_name_semi = "Mixed/%s_semisupervised_fold%d_full_%d.log" % (save_model,f,size_semi)

            model_path1 = "%s_supervised1_fold%d_full.model" % (save_model,f)
            #training.TrainGenerativeModelSupervised(idx_train,idx_test,device,epoch,log_interval,save_model=model_path1)
            model_path_semi = "%s_semisupervised_fold%d_full_%d.model" % (save_model,f,size_semi)
            training.TrainGenerativeModelSemiSupervise(idx_train,idx_test,idx_unlabel,device,epoch*3,log_interval,save_model=model_path_semi,log_name = log_name_semi)
            C.PATH_ESTIMATE_CROSS = estimate_path
            training.Estimate(idx_test,model_path_semi,device,verbose=False)
            list_transition_rate.extend(training.Estimate_transrate(idx_test,model_path_semi,device))
    
    C.PATH_ESTIMATE_CROSS = estimate_path
    scores_majmin,scores_triads,scores_tetrads,durations,confmatrix_root,confmatrix_qual = chord.EvaluateLabels(foldman.getAll())
    score_majmin = np.sum(scores_majmin*durations)/np.sum(durations)
    score_triads = np.sum(scores_triads*durations)/np.sum(durations)
    print("Cross validation full-supervised:")
    print("majmin: %.4f" %  (score_majmin))
    print("triads: %.4f" %  (score_triads))
    
    scores[0,i] = score_majmin
    scores[1,i] = score_triads


np.savez("scores_fullsemi_%s_%d_all.npz" % (save_model, size_semi),
         majmin = scores_majmin,
         triads = scores_triads,
         transrate = list_transition_rate)
np.save("scores_fullsemi_%s_%d.npy" % (save_model, size_semi),scores)
