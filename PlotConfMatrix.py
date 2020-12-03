#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 18:45:39 2020

@author: wu
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools

import const as C
import chord
import folds

def plotConfMat(mat,classes,title,color="black",fontsize=2):
    #plt.figure(title)
    plt.imshow(mat,interpolation="nearest",cmap=plt.cm.Blues)
    plt.title(title,fontsize=fontsize+5,fontname="STIXGeneral",color=color)
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks,classes,fontsize=fontsize,fontname="STIXGeneral")
    plt.xlabel("Estimated",fontsize=fontsize,fontname="STIXGeneral")
    plt.yticks(tick_marks,classes,fontsize=fontsize,fontname="STIXGeneral")
    plt.ylabel("Reference",fontsize=fontsize,fontname="STIXGeneral")
    plt.ylim(7.5,-0.5)
    plt.xlim(-0.5,7.5)
    for i,j in itertools.product(range(len(classes)),range(len(classes))):
        plt.text(j,i+0.1,format(mat[i,j],".3f"),fontsize=fontsize,horizontalalignment="center",color="white" if mat[i,j]>0.5 else "black",fontname="STIXGeneral")
    plt.show()


C.PATH_ESTIMATE_CROSS = "chromavae_noshiftdoublesuper_estimated_cross_mix2"
foldman = folds.FoldManager()
idx = foldman.getAll()
#scores_majmin,scores_triads,scores_tetrads,durations,confmatrix_root,confmatrix_quality = chord.EvaluateLabels(idx)
confmatrix_root,confmatrix_quality = chord.ConfMatrix(idx)


C.PATH_ESTIMATE_CROSS = "chromavae_noshift_estimated_cross_mix2_2"
confmatrix_root_sup, confmatrix_quality_sup = chord.ConfMatrix(idx)

c_root = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
c_qual = ["maj","min","aug","dim","sus4","sus2","1","5"]

#plotConfMat(confmatrix_root,c_root,"root",2)
plt.subplot(2,1,2)
plotConfMat(confmatrix_quality,c_qual,"VAE-MR-SSL","blue",20)
plt.subplot(2,1,1)
plotConfMat(confmatrix_quality_sup,c_qual,"ACE-SL","tomato",20)