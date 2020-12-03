#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 14:21:35 2019

@author: wu
"""

import numpy as np

class FoldManager():
    def __init__(self,
                 datas=["folds_isophonics.npy","folds_rwc.npy","folds_uspop.npy","folds_billboard.npy"],
                 names=["Isophonics","RWC","USPOP","Billboard"]):
        self.folds = [np.load(d) for d in datas]
        self.nfolds = len(self.folds[0])
        self.nsets = len(datas)
        self.names = names
    def getTestFold(self,fold):
        return np.concatenate([f[fold] for f in self.folds])
    def getDatasetIdx(self,dset):
        return np.concatenate(self.folds[dset]),self.names[dset]
    def getTrainSupervisedFold(self,fold,supervise_size=None):
        idxes = []
        idx_train = list(range(self.nfolds))
        idx_train.remove(fold)
        idx_supervise = idx_train if supervise_size is None else idx_train[:supervise_size]
        for n in range(self.nsets):
            idxes.extend([self.folds[n][i] for i in idx_supervise])
        idxes = np.concatenate(idxes)
        return idxes
    def getTrainUnsupervisedFold(self,fold,supervise_size):
        idxes = []
        idx_train = list(range(self.nfolds))
        idx_train.remove(fold)
        idx_supervise = idx_train[supervise_size:]
        for n in range(self.nsets):
            idxes.extend([self.folds[n][i] for i in idx_supervise])
        idxes = np.concatenate(idxes)
        return idxes
    def getAll(self):
        return np.concatenate([np.concatenate(f) for f in self.folds])
    def retrieveFold(self,idx):
        for f in range(self.nfolds):
            for d in range(self.nsets):
                if idx in self.folds[d][f]:
                    return f