#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 15:28:01 2019

@author: wuyiming
"""

import argparse
import training
import numpy as np
import const as C
import chord
import folds


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
                    default="chromavae_latent16_beta1.model",
                    type=str)
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
parser.add_argument('--encregular',
                    action='store',
                    default=0,
                    type=int)
parser.add_argument('--entropy',
                    action='store',
                    default=1.0,
                    type=float)
parser.add_argument('--vamp',
                    action='store',
                    default=0,
                    type=int)
parser.add_argument('--semigen',
                    action='store',
                    default=1,
                    type=int)

args = parser.parse_args()
device = args.device
epoch = args.epoch
log_interval = args.loginterval
save_model = args.save



C.MARKOV_REGULARIZE = (args.markov > 0)
C.VAE_RAND_SHIFT = (args.shift == 1)
C.VAE_RAND_SHIFT_QUALITY = (args.shift == 2)
C.POSTFILTER = (args.postfilter > 0)
C.VAE_BETA = args.beta
C.VAE_SHIFT_REGULAR = (args.encregular>0)
C.VAE_WEIGHT_ENTROPY = args.entropy
C.PSEUDO_PRIOR = (args.vamp>0)
C.GENERATOR_SEMI = (args.semigen>0)

print("Feat size:%d" % C.N_DIMS_FEAT)
print("Vamp Prior:%s" % C.PSEUDO_PRIOR)
print("Generator Random Shift:%s" % C.VAE_RAND_SHIFT)
print("Generator Quality Random Shift:%s" % C.VAE_RAND_SHIFT_QUALITY)
print("VAE Latent size:%d" % C.N_DIMS_VAE_LATENT)
print("Markov regularize:%s" % C.MARKOV_REGULARIZE)
print("Encoder shift regularize:%s" % C.VAE_SHIFT_REGULAR)
print("Post filtering:%s" % C.POSTFILTER)

np.random.seed(10)

foldman = folds.FoldManager()
idx_ext = np.load("idx_non_billboard.npy")+320
#idx_data = np.concatenate((np.arange(320),idx_ext))
#idx_rand = np.random.permutation(1217) + 320
idx_rand = foldman.getAll()
idx_rand = np.random.permutation(idx_rand)

#idx_rand = np.random.permutation(890)
#idx_train = np.concatenate((idx_rand[50:],idx_ext))

idx_train = idx_rand
idx_unlabel = idx_rand
idx_test = np.concatenate(foldman.folds[1])
#idx_test = np.arange(143)


#idx_train = idx_rand[:-100]
#idx_unlabel = np.arange(1537,1537+700)
#idx_unlabel = idx_ext
#idx_unlabel = np.arange(400,1000)
#idx_unlabel = idx_rand[100:-50]
#idx_test = idx_rand[-100:]
#idx_test = idx_ext
"""
print("supervised size:%d  unsupervised size:%d  test size:%d" % (len(idx_train),len(idx_unlabel),len(idx_test)))

training.TrainGenerativeModelSupervised(idx_train,idx_test,device,epoch=epoch*3,log_interval=log_interval,save_model=save_model)
training.Estimate(idx_test,save_model,device,verbose=False)

scores_majmin,scores_triads,scores_tetrads,durations,confmatrix_root,confmatrix_quality = chord.EvaluateLabels(idx_test)


print("majmin: %.4f" %  (np.sum(scores_majmin*durations)/np.sum(durations)))
print("triads: %.4f" %  (np.sum(scores_triads*durations)/np.sum(durations)))
print("tetrads: %.4f" %  (np.sum(scores_tetrads*durations)/np.sum(durations)))
"""


#training.TrainGenerativeModelSemiSupervise(idx_train,idx_test,idx_unlabel,device,epoch*3,log_interval,resume=None,save_model=save_model+"2")
#C.PATH_ESTIMATE_CROSS = "est_songle_proposed"
#C.PATH_ESTIMATE_CROSS = "estimated_cross"
training.Estimate_beatsync(idx_test,save_model+"2",device,verbose=False)
#training.Estimate_largevoc(idx_test,save_model+"1",device,verbose=False)
scores_majmin,scores_triads,scores_tetrads,durations,confmatrix_root,confmatrix_quality = chord.EvaluateLabels(idx_test)

print("majmin: %.4f" %  (np.sum(scores_majmin*durations)/np.sum(durations)))
print("triads: %.4f" %  (np.sum(scores_triads*durations)/np.sum(durations)))
print("tetrads: %.4f" %  (np.sum(scores_tetrads*durations)/np.sum(durations)))


#training.TrainCQTFramewiseModelSemiSupervise(idx_train,idx_test,idx_unlabel,device,epoch,log_interval,save_model=save_model,classifier_only=False)
#training.TrainChromaVAE(np.arange(200),device,epoch,log_interval,save_model=save_model)

