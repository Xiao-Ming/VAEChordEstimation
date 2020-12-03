#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 22:50:31 2019

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
import folds


foldman = folds.FoldManager()
np.random.seed(10)
idx_rand = np.random.permutation(220)
fold = np.load("folds_320files.npy")[0][:5]
idx_billboard = np.load("idx_non_billboard.npy")+320
idx = 531
dset = D.ChordDataset([idx])
#dset_semi = D.ChordDatasetSemisupervised([472], [472])
fld = foldman.retrieveFold(idx)

print("fold:%d" % fld)

model = net_generative.UnifiedModel()
model_sup = net_generative.UnifiedModel()
model_nom = net_generative.UnifiedModel()
#model.load("chromavae_fullsemi_semisupervised_fold0_full_700.model")
model_name =     "/n/work1/wu/saved_models/ChordRecGenerative/chromavae_noshiftdoublesuper_semisupervised_fold%d_size1.model" % fld
model_sup_name = "/n/work1/wu/saved_models/ChordRecGenerative/chromavae_noshift_supervised2_fold%d_size1.model" % fld
#model_sup_name = "chromavae_full_supervised1_fold0_full.model"
model_nom_name = "/n/work1/wu/saved_models/ChordRecGenerative/chromavae_nomarkovdoublesuper_semisupervised_fold%d_size1.model" % fld
#model_name = "chromavae_proposed.model2"
#C.VAE_SHIFT_REGULAR = True
C.MARKOV_REGULARIZE = False
C.POSTFILTER = False
model.load(model_name)
model_sup.load(model_sup_name)
model_nom.load(model_nom_name)
#model.load("chromavae_fullsemi_semisupervised_fold3_full_700.model")
#model.load("chromavae_randshift_supervised1_fold0_full.model")
config.train = False
config.enable_backprop = False

list_prob_supervised = []
list_prob_pred = []
list_accr = []

start_idx = int(C.SR * 90) // C.H
plot_length = C.SR * 20 // C.H

feat,labs,aligns = dset[0]
#_,feat_un,_,_ = dset_semi[0]

plt.figure(model_name)

plt.subplot(5,1,1)
specshow(feat[start_idx:start_idx+plot_length,:24].T)
plt.yticks(np.arange(24)+0.5,["C","","","","","F","","G","","","","","C","","","","","F","","G","","","",""],fontname="STIXGeneral")
#plt.text(-25,6,"(a)",fontname="STIXGeneral",fontsize=15)
dist_orig = dist.Bernoulli(feat[start_idx:start_idx+plot_length])
#plt.subplot(8,1,2)
#labs_onehot = [U.encode_onehot(labs[aligns[:512],i],cat) for i,cat in zip(list(range(6)),[C.N_VOCABULARY_TRIADS,13,4,4,3,3])]
labs_onehot = U.encode_onehot(labs[aligns[start_idx:start_idx+plot_length]],C.N_VOCABULARY_TRIADS)
generated = model.generator.reconstr_dist([feat[start_idx:start_idx+plot_length]],[labs_onehot])[0]
#generated = (generated.a/(generated.a+generated.b)).data
#specshow(generated[:,:24].T)
#plt.yticks(np.arange(24)+0.5,["C","","","","","F","","G","","","","","C","","","","","F","","G","","","",""],fontname="STIXGeneral")
#plt.text(-25,6,"(c)",fontname="STIXGeneral",fontsize=15)

#plt.subplot(8,1,3)
#generated,lab_estimated = model.reconst(feat[start_idx:start_idx+plot_length])
#print("P_proposed= %.5f" % dist_orig.log_prob(generated).data.sum(-1).mean())
lab_estimated = model.estimate(feat)[start_idx:start_idx+plot_length]
#generated = (generated.a/(generated.a+generated.b)).data
#specshow(generated[:,:24].T)
#plt.yticks(np.arange(24)+0.5,["C","","","","","F","","G","","","","","C","","","","","F","","G","","","",""],fontname="STIXGeneral")
#plt.text(-25,6,"(c)",fontname="STIXGeneral",fontsize=15)

plt.subplot(5,1,5)
#p.margins(0)
specshow(lab_estimated[None,:],vmin = 0, vmax = 50, cmap = "viridis")
#plt.plot(labs[aligns[:plot_length]],"b")
#plt.plot(lab_estimated_filter,"r")

plt.subplot(5,1,2)
specshow(labs[None,aligns[start_idx:start_idx+plot_length]],vmin = 0, vmax = 50, cmap = "viridis")

plt.subplot(5,1,3)
#generated_sup,lab_estimated_sup = model_sup.reconst(feat[start_idx:start_idx+plot_length])
lab_estimated_sup = model_sup.estimate(feat)[start_idx:start_idx+plot_length]
specshow(lab_estimated_sup[None,:],vmin = 0, vmax = 50, cmap = "viridis")
#print("P_supervised = %.5f" % dist_orig.log_prob(generated_sup).data.sum(-1).mean())

plt.subplot(5,1,4)
#generated_nom, lab_estimated_nom = model_nom.reconst(feat[start_idx:start_idx+plot_length])
lab_estimated_nom = model_nom.estimate(feat)[start_idx:start_idx+plot_length]
#print("P_nomarkov = %.5f" % dist_orig.log_prob(generated_nom).data.sum(-1).mean())
specshow(lab_estimated_nom[None,:],vmin = 0, vmax = 50, cmap = "viridis")

"""
plt.subplot(8,1,8)
generated,_ = model_nom.reconst(feat[start_idx:start_idx+plot_length])
#lab_estimated_filter = model.estimate(feat[:plot_length])
#generated = (generated.a/(generated.a+generated.b)).data
specshow(generated[:,:24].T)
plt.yticks(np.arange(24)+0.5,["C","","","","","F","","G","","","","","C","","","","","F","","G","","","",""],fontname="STIXGeneral")
plt.text(-25,6,"(c)",fontname="STIXGeneral",fontsize=15)
"""
"""
plt.subplot(5,1,3)
labs_alter = np.array(labs)
labs_alter[labs!=72] = (labs[labs!=72] + 12) % 72
generated = model.generator.reconstr_dist([feat[:512]],[U.encode_onehot(labs_alter[aligns[:512]])])[0].p.data
#generated = (generated.a/(generated.a+generated.b)).data
specshow(generated[:,:24].T)
plt.yticks(np.arange(12)+0.5,["C","","D","","E","F","","G","","A","","B"],fontname="STIXGeneral")
plt.text(-25,6,"(b)",fontname="STIXGeneral",fontsize=15)

plt.subplot(5,1,4)
generated = model.generator.reconstr_dist_randzs([feat[:512]],[U.encode_onehot(labs[aligns[:512]])])[0].p.data
#generated = (generated.a/(generated.a+generated.b)).data
specshow(generated[:,:24].T)
plt.yticks(np.arange(12)+0.5,["C","","D","","E","F","","G","","A","","B"],fontname="STIXGeneral")
plt.text(-25,6,"(c)",fontname="STIXGeneral",fontsize=15)
"""

"""
#list_labs = [0,1,2,3,4,5,6,7,8,9,10,11]
#list_name = ["C:maj","C#:maj","D:maj","D#:maj","E:maj","F:maj","F#:maj","G:maj","G#:maj","A:maj","A#:maj","B:maj"]

list_triad = [0,1,2,12,13,14,48,49,50,60,61,62]
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

list_name = ["C:maj","C#:maj","D:maj","C:min","C#:min","D:min","C:aug","C#:aug","D:aug","C:dim","C#:dim","D:dim"]
for i in range(12):
    C.VAE_SHIFT_REGULAR = False
    #labs_onehot = [U.encode_onehot(np.ones(128,dtype=np.int32)*list_labs[j][i],cat)  for j,cat in zip(list(range(6)),[C.N_VOCABULARY_TRIADS,13,4,4,3,3])]
    labs_onehot = U.encode_onehot(np.ones(40,dtype=np.int32)*list_triad[i],C.N_VOCABULARY_TRIADS)
    generated = model.generator.reconstr_dist([feat[100:140,:]],[labs_onehot])[0]
    #generated = (generated.a/(generated.a+generated.b)).data
    plt.subplot(5,12,48+i+1)
    plt.title(list_name[i],fontname="STIXGeneral")
    if i==0:
        plt.text(-90,6,"(d)",fontname="STIXGeneral",fontsize=15)
    specshow(generated[:,:24].T)
    if i==0:
        plt.yticks(np.arange(12)+0.5,["C","","D","","E","F","","G","","A","","B"],fontname="STIXGeneral")
"""    

