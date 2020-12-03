#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 19:57:50 2019

@author: wu
"""

import argparse
import training
import numpy as np
import const as C
import chord
import os
import util
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
parser.add_argument('-a','--annotatesize',\
                    action='store',
                    default=1,
                    type=int)
parser.add_argument('--markov',\
                    action='store',
                    default=1,
                    type=int)
parser.add_argument('--shift',
                    action='store',
                    default=0,
                    type=int)
parser.add_argument('--supervised',
                    action='store',
                    default=1,
                    type=int)
parser.add_argument('-b','--beta',
                    action='store',
                    default=1.0,
                    type=float)
parser.add_argument('--encregular',
                    action='store',
                    default=0,
                    type=int)
parser.add_argument('--semigen',
                    action='store',
                    default=1,
                    type=int)
parser.add_argument('--doublesuper',\
                    action='store',
                    default=0,
                    type=int)

args = parser.parse_args()
device = args.device
epoch = args.epoch
log_interval = args.loginterval
save_model = args.save
supervise_size = args.annotatesize
supervised  = (args.supervised == 1)
doublesuper = (args.doublesuper>0)

C.MARKOV_REGULARIZE = (args.markov > 0)
C.VAE_RAND_SHIFT = (args.shift == 1)
C.VAE_RAND_SHIFT_QUALITY = (args.shift == 2)
C.VAE_SHIFT_REGULAR = (args.encregular>0)
C.VAE_BETA = args.beta
C.GENERATOR_SEMI = (args.semigen>0)


print("Vamp Prior:%s" % C.PSEUDO_PRIOR)
print("Generator Random Shift:%s" % C.VAE_RAND_SHIFT)
print("Generator Random Shift Quality:%s" % C.VAE_RAND_SHIFT_QUALITY)
print("VAE Latent size:%d" % C.N_DIMS_VAE_LATENT)
print("VAE encoder regularize:%s" % C.VAE_SHIFT_REGULAR)
print("Markov regularize:%s" % C.MARKOV_REGULARIZE)
print("Post filtering:%s" % C.POSTFILTER)


#print("Result path: %s" % C.PATH_ESTIMATE_CROSS)


scores = np.zeros((4,1),dtype=np.float32)
estimate_path = "%s_estimated_cross_mix%d" % (save_model,supervise_size)

util.InitEvaluationPath(estimate_path)
if supervised:
    util.InitEvaluationPath(estimate_path+"_2")


foldman = FoldManager()
list_transition_rate = []

for f in range(foldman.nfolds):
    print("FOLD:%d" % (f+1))
    idx_test = foldman.getTestFold(f)
    idx_train_supervise = foldman.getTrainSupervisedFold(f,supervise_size)
    idx_train_unsupervise = foldman.getTrainUnsupervisedFold(f,supervise_size)
    
    if doublesuper:
        idx_train_unsupervise = np.concatenate([idx_train_supervise,idx_train_unsupervise])
    log_name_supervised1 = "Mixed/%s_supervised1_fold%d_size%d.log" % (save_model,f,supervise_size)
    log_name_supervised2 = "Mixed/%s_supervised2_fold%d_size%d.log" % (save_model,f,supervise_size)
    log_name_semisupervised = "Mixed/%s_semisupervised_fold%d_size%d.log" % (save_model,f,supervise_size)
    
    model_path1 = "/n/work1/wu/saved_models/ChordRecGenerative/%s_supervised1_fold%d_size%d.model" % (save_model,f,supervise_size)
    model_path2 = "/n/work1/wu/saved_models/ChordRecGenerative/%s_supervised2_fold%d_size%d.model" % (save_model,f,supervise_size)
    np.random.seed(10)
    
    if supervised:
        
        if doublesuper:
            training.TrainGenerativeModelSemiSupervise(idx_train_supervise, idx_test, idx_train_supervise, device,
                                                       epoch*3, log_interval, save_model=model_path2,log_name=log_name_supervised2)
        else:
            training.TrainGenerativeModelSupervised(idx_train_supervise,idx_test,device,epoch*3,log_interval,save_model=model_path2,
                                                log_name=log_name_supervised2)
        
        C.PATH_ESTIMATE_CROSS = estimate_path+"_2"
        training.Estimate(idx_test,model_path2,device,verbose=False)
    else:
        pass
    
    model_path_semi = "/n/work1/wu/saved_models/ChordRecGenerative/%s_semisupervised_fold%d_size%d.model" % (save_model,f,supervise_size)
    
    training.TrainGenerativeModelSemiSupervise(idx_train_supervise, idx_test, idx_train_unsupervise,
                                              device,epoch*3, log_interval, resume=None,
                                              save_model=model_path_semi, log_name=log_name_semisupervised)
    
    C.PATH_ESTIMATE_CROSS = estimate_path
    training.Estimate(idx_test,model_path_semi,device,verbose=False)
    
    list_transition_rate.extend(training.Estimate_transrate(idx_test,model_path_semi,device))

C.PATH_ESTIMATE_CROSS = estimate_path
scores_majmin,scores_triads,scores_mirex,durations,confmatrix_root,confmatrix_qual = chord.EvaluateLabels(foldman.getAll())


score_majmin = np.sum(scores_majmin*durations)/np.sum(durations)
score_triads = np.sum(scores_triads*durations)/np.sum(durations)
score_mirex = np.sum(scores_mirex*durations)/np.sum(durations)


std_majmin = np.std(scores_majmin)
std_triads = np.std(scores_triads)
std_mirex = np.std(scores_mirex)


print("supervise data size:%d" % supervise_size)
print("Cross validation semi-supervised:")
print("majmin: mean=%.4f, std=%.4f" %  (score_majmin,std_majmin))
print("triads: mean=%.4f, std=%.4f" %  (score_triads,std_triads))
print("mirex: mean=%.4f, std=%.4f" % (score_mirex,std_mirex))
print("transition_rate: quantile=%s" % (np.percentile(list_transition_rate,q=[25,50,75])))
scores[0,0] = score_majmin
scores[1,0] = score_triads
#scores[2,0] = score_mirex

if supervised:
    C.PATH_ESTIMATE_CROSS = estimate_path + "_2"
    scores_majmin,scores_triads,scores_mirex,durations,confmatrix_root,confmatrix_qual = chord.EvaluateLabels(foldman.getAll())
    
    score_majmin = np.sum(scores_majmin*durations)/np.sum(durations)
    score_triads = np.sum(scores_triads*durations)/np.sum(durations)
    score_mirex = np.sum(scores_mirex*durations)/np.sum(durations)
    std_majmin = np.std(scores_majmin)
    std_triads = np.std(scores_triads)
    std_mirex = np.std(scores_mirex)
    print("Cross validation full-supervised:")
    print("majmin: mean=%.4f, std=%.4f" %  (score_majmin,std_majmin))
    print("triads: mean=%.4f, std=%.4f" %  (score_triads,std_triads))
    print("mirex: mean=%.4f, std=%.4f" %  (score_mirex,std_mirex))
    
    scores[2,0] = score_majmin
    scores[3,0] = score_triads
    #scores[5,0] = score_mirex
    
#np.save("scores/%s_scores_mixed_size%d.model" % (save_model,supervise_size),scores)
#print(scores)

np.savez("scores/%s_scores_mixed_size%d_all.npz" % (save_model,supervise_size),
         majmin = scores_majmin,
         triads = scores_triads,
         transrate = list_transition_rate)
