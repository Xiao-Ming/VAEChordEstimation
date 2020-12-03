#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 14:25:54 2019

@author: wuyiming
"""


import numpy as np
import const as C
from librosa.segment import agglomerative
import chainer.backend
import os
import shutil

def shift_feature(feat,shift,segments):
    segments = feat.shape[1] // 12
    xp = chainer.backend.get_array_module(feat)
    seg_list = []
    for i in range(segments):
        seg_list.append(xp.roll(feat[:,i*12:(i+1)*12],shift,1))
    feat_shifted = xp.concatenate(seg_list,axis=1)
    return feat_shifted
    

def path_estimated_lab(path):
    sp = path.split("/")
    return os.path.join(C.PATH_ESTIMATE_CROSS,sp[-2],sp[-1])
    #return os.path.join(C.PATH_ESTIMATE_CROSS,sp[-1])


def normalize_spec(spec):
    return spec+1e-4
    #spec_log = np.log(spec+1e-4)
    #return (spec_log-np.mean(spec_log))/np.var(spec_log)
    #return np.clip(spec/spec.max(),1e-4,1-(1e-4))


def soft_ncut_loss(weight,arr_class):
    prob = np.zeros((weight.shape[0],8),dtype=np.float32)
    prob[np.arange(weight.shape[0]),arr_class] = 1
    K = C.N_VOCABULARY
    numerator = 0.0
    denominator = 0.0
    s = 0.0
    for k in range(K):
        denominator = np.matmul(prob[:,k], np.sum(weight,axis=1))
        numerator = np.matmul(prob[:,k],np.squeeze(np.matmul(weight,prob[:,k,None])))
        s += numerator / denominator
    loss = K - s
    return loss

def evaluate_segmentation(estimated,truth,thld=1):
    pos_transition_est = np.nonzero(np.diff(estimated))[0].astype(int)+1
    pos_transition_tru = np.nonzero(np.diff(truth))[0].astype(int)+1
    
    count_recall=0.0
    count_precise = 0.0
    for pos in pos_transition_est:
        if (np.abs(pos_transition_tru-pos)<=thld).sum()>0:
            count_precise += 1
    
    for pos in pos_transition_tru:
        if (np.abs(pos_transition_est-pos)<=thld).sum()>0:
            count_recall += 1
    
    recall_rate = count_recall/len(pos_transition_tru)
    precision_rate = count_precise/len(pos_transition_est)
    F = 2*recall_rate*precision_rate/(recall_rate+precision_rate)
    
    return recall_rate,precision_rate,F

def separate_label(lab):
    align = np.zeros(lab.shape,dtype=np.int32)
    list_symbol = [lab[0]]    
    pos_diff = np.nonzero(np.diff(lab))[0] + 1
    pos_diff = np.append(pos_diff,lab.size)
    for i in range(1,len(pos_diff)):
        align[pos_diff[i-1]:pos_diff[i]] = i
        list_symbol.append(lab[pos_diff[i-1]])
    
    return np.array(list_symbol),align

def encode_onehot(y,n_category=C.N_VOCABULARY_TRIADS,noise=False):
    code = np.zeros((len(y),n_category),dtype=np.float32)
    code[np.arange(code.shape[0]),y] = 1
    if noise:
        noise = np.random.beta(1,3,code.shape)
        code += noise
        code /= code.sum(axis=1,keepdims=True)
    return code

import itertools
def ncut_loss(weight,arr_class):
    n_class = 8
    class_accum = np.zeros(n_class)
    cut_accum = 0.0
    for i,j in itertools.product(range(weight.shape[0]),range(weight.shape[1])):
        if arr_class[i] == arr_class[j]:
            class_accum[arr_class[i]] += weight[i,j]
        else:
            cut_accum += weight[i,j]
    return np.sum(class_accum),cut_accum

def InitEvaluationPath(eval_path):
    albums = ["1_Annotated",
              "2_NonAnnotated"]
    if os.path.isdir(eval_path):
        shutil.rmtree(eval_path)
    for alb in albums:
        os.makedirs(os.path.join(eval_path,alb))
