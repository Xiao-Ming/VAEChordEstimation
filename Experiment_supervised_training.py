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
                    default=1,
                    type=int)
parser.add_argument('-s','--save',\
                    action='store',
                    default="chromavae_randcut",
                    type=str)
parser.add_argument('--doublesuper',\
                    action='store',
                    default=0,
                    type=int)
parser.add_argument('--markov',\
                    action='store',
                    default=1,
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
doublesuper = (args.doublesuper > 0)
C.MARKOV_REGULARIZE = (args.markov > 0)
C.SELF_TRANS_RATE = args.selftrans

foldman = FoldManager()

print("Vamp Prior:%s" % C.PSEUDO_PRIOR)
print("Generator Random Shift:%s" % C.VAE_RAND_SHIFT)
print("Generator Random Shift Quality:%s" % C.VAE_RAND_SHIFT_QUALITY)
print("VAE Latent size:%d" % C.N_DIMS_VAE_LATENT)
print("VAE encoder regularize:%s" % C.VAE_SHIFT_REGULAR)
print("Markov regularize:%s" % C.MARKOV_REGULARIZE)
print("Post filtering:%s" % C.POSTFILTER)
print("Self transition rate: %.4f" % C.SELF_TRANS_RATE)

print("Result path: %s" % C.PATH_ESTIMATE_CROSS)

scores = np.zeros((2,1),dtype=np.float32)
list_transition_rate = []

for i in range(1):
    print("Iteration:%d" % (i+1))
    estimate_path = "estimated_cross_full"
    for f in range(foldman.nfolds):
        print("FOLD:%d" % (f+1))
        idx_test = foldman.getTestFold(f)
        idx_train = foldman.getTrainSupervisedFold(f)
        log_name_supervised1 = "Mixed/%s_supervised1_fold%d_full.log" % (save_model,f)
        log_name_supervised2 = "Mixed/%s_supervised2_fold%d_full.log" % (save_model,f)
        
        model_path1 = "%s_supervised1_fold%d_full.model" % (save_model,f)
        model_path2 = "%s_supervised2_fold%d_full.model" % (save_model,f)
        np.random.seed(10)
        
        if doublesuper:
            training.TrainGenerativeModelSemiSupervise(idx_train, idx_test, idx_train, device,epoch*3,log_interval,save_model=model_path1,log_name=log_name_supervised1)
        else:
            training.TrainGenerativeModelSupervised(idx_train,idx_test,device,epoch*3,log_interval,save_model=model_path1,log_name=log_name_supervised1)
        
        C.PATH_ESTIMATE_CROSS = estimate_path
        training.Estimate(idx_test,model_path1,device,verbose=False)
        list_transition_rate.extend(training.Estimate_transrate(idx_test,model_path1,device))
    
    C.PATH_ESTIMATE_CROSS = estimate_path
    scores_majmin,scores_triads,scores_tetrads,durations,confmatrix_root,confmatrix_qual = chord.EvaluateLabels(foldman.getAll())
    score_majmin = np.sum(scores_majmin*durations)/np.sum(durations)
    score_triads = np.sum(scores_triads*durations)/np.sum(durations)
    print("Cross validation full-supervised:")
    print("majmin: %.4f" %  (score_majmin))
    print("triads: %.4f" %  (score_triads))
    
    scores[0,i] = score_majmin
    scores[1,i] = score_triads

np.save("scores_full_%s.npy" % save_model,scores)

np.savez("scores_full_%s_all.npz" % save_model,
         majmin = scores_majmin,
         triads = scores_triads,
         transrate = list_transition_rate)