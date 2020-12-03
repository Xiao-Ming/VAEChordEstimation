#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 16:53:08 2020

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
import chainer.distributions as dist


np.random.seed(10)
idx_rand = np.random.permutation(220)
fold = np.load("folds_320files.npy")[0][:5]
idx_billboard = np.load("idx_non_billboard.npy")+320
#dset = D.ChordDataset([333,472,533,1200,679])
dset = D.ChordDataset([212+11])
#dset_semi = D.ChordDatasetSemisupervised([472], [472])

model = net_generative.GenerativeChordnet()
model_sup = net_generative.GenerativeChordnet()
model_nom = net_generative.GenerativeChordnet()
#model.load("chromavae_fullsemi_semisupervised_fold0_full_700.model")
model_name =     "/n/work1/wu/saved_models/ChordRecGenerative/chromavae_noshiftdoublesuper_supervised2_fold0_size2.model"
#model_name = "/home/wu/Projects/ChordRecGenerative/chromavae_full_supervised1_fold0_full.model"
model_sup_name = "models/chromavae_noshift_supervised2_fold0_size1.model"
model_nom_name = "models/chromavae_nomarkov_semisupervised_fold0_size1.model"
#model_name = "chromavae_proposed.model2"
#C.VAE_SHIFT_REGULAR = True
C.MARKOV_REGULARIZE = False
C.POSTFILTER = False
model.load(model_name)
#model_sup.load(model_sup_name)
#model_nom.load(model_nom_name)

feat,labs,aligns = dset[0]

#list_labs = [0,1,2,3,4,5,6,7,8,9,10,11]
#list_name = ["C:maj","C#:maj","D:maj","D#:maj","E:maj","F:maj","F#:maj","G:maj","G#:maj","A:maj","A#:maj","B:maj"]

list_triad = [0,1,2,12,13,14,24,25,26,36,37,38,48,49,50,60,61,62,72,73,74,84,85,86]
#list_triad = [0,1,2,12,13,14,48,49,50,60,61,62]
#list_triad = [0,1,2,0,1,2,0,1,2,0,1,2]
list_bass = [0,1,2,7,8,9,0,1,2,0,1,2]
#list_bass = [0,0,0,0,0,0,0,0,0,0,0,0]
#list_seventh = [3,3,3,0,0,0,1,1,1,2,2,2]
list_seventh = [3,3,3,3,3,3,3,3,3,3,3,3]
list_ninth = [3,3,3,3,3,3,3,3,3,3,3,3]
list_eleventh = [2,2,2,2,2,2,2,2,2,2,2,2]
list_thirteenth = [2,2,2,2,2,2,2,2,2,2,2,2]

list_labs = [list_triad, list_bass, list_seventh, list_ninth, list_eleventh, list_thirteenth]

list_name = ["C:maj","C#:maj","D:maj","C:min","C#:min","D:min","C:aug","C#:aug","D:aug","C:dim","C#:dim","D:dim",
             "C:sus4","C#:sus4","D:sus4","C:sus2","C#:sus2","D:sus2","C:1","C#:1","D:1","C:5","C#:5","D:5"]
for i in range(24):
    C.VAE_SHIFT_REGULAR = False
    #labs_onehot = [U.encode_onehot(np.ones(128,dtype=np.int32)*list_labs[j][i],cat)  for j,cat in zip(list(range(6)),[C.N_VOCABULARY_TRIADS,13,4,4,3,3])]
    labs_onehot = U.encode_onehot(np.ones(10,dtype=np.int32)*list_triad[i],C.N_VOCABULARY_TRIADS)
    generated = model.generator.reconstr_dist([feat[689:699,:]],[labs_onehot])[0]
    #generated = (generated.a/(generated.a+generated.b)).data
    plt.subplot(4,6,i+1)
    plt.title(list_name[i],fontname="STIXGeneral")
    specshow(generated[:,:24].T)
    if i%6==0:
        plt.yticks(np.arange(24)+0.5,["C","","","","","F","","G","","","","","C","","","","","F","","G","","","",""],fontname="STIXGeneral")