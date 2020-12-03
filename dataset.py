#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 12:59:21 2018

@author: wuyiming
"""

from chainer import dataset
from librosa.core import magphase
from librosa.util import find_files
from librosa.feature import melspectrogram
import const as C
import numpy as np
import util as U
import chord





def LoadLabelSet(labfilelist,shift=None):
    def time2idx(time):
        return int(np.round(time * C.SR / C.H))
    
    labs_list = []
    intervals_list = []
    
    if shift is None:
        shift = np.zeros(len(labfilelist),dtype=int)
    
    for labfile,sh in zip(labfilelist,shift):
        f = open(labfile)
        labs_text = []
        intervals = []
        line = f.readline()
        while (line != "") and (not line.isspace()):
            sp = line.split()
            if time2idx(float(sp[0])) < time2idx(float(sp[1])):
                labs_text.append(chord.shift_label(sp[2],sh))
                intervals.append((time2idx(float(sp[0])),time2idx(float(sp[1]))))    
            line = f.readline()
        labs_list.append(labs_text)
        intervals_list.append(np.array(intervals,dtype=np.int32))
       
    return labs_list,intervals_list

def LoadLabelFramewise(labfile):
    def time2idx(time):
        return int(np.round(time * C.SR / C.H))
    
    
    labarr = np.zeros(1000*C.SR//C.H,dtype=np.int32)
    f = open(labfile)
    line = f.readline()
    while (line != "") and (not line.isspace()):
        sp = line.split()
        st = time2idx(float(sp[0]))
        ed = time2idx(float(sp[1]))
        lab = chord.lab2target_single(sp[2])
        labarr[st:ed] = lab
        line = f.readline()
    
    return labarr[:ed]
            

def ShiftDatapair(feat,label,shift):
    xp = np
    seg_list = [xp.roll(feat[:,i*12:(i+1)*12],shift,-1) for i in range(feat.shape[1]//12)]
    """
    seg_list = [xp.roll(feat[:,:12],shift,-1),
                xp.roll(feat[:,12:24],shift,-1),
                xp.roll(feat[:,24:36],shift,-1)]
    """
    feat_shifted = xp.concatenate(seg_list,axis=-1)
    if label is not None:
        label_shifted = [chord.shift_label(l,shift) for l in label]
        return feat_shifted,label_shifted
    else:
        return feat_shifted

def ShiftDatapairHCQT(cqt,label,shift):
    cqt_shifted = np.zeros(cqt.shape,dtype=np.float32) + 1e-4
    sh = shift - 6
    if sh>0:
        cqt_shifted[:,:,sh:] = cqt[:,:,:-sh]
    elif sh<0:
        cqt_shifted[:,:,:-sh] = cqt[:,:,sh:]
    else:
        cqt_shifted = cqt
    if label is not None:
        label_shifted = [chord.shift_label(l,shift) for l in label]
        return cqt_shifted,label_shifted
    else:
        return cqt_shifted



class ChordLanguageDataset(dataset.DatasetMixin):
    def __init__(self,idx,rand_shift=False,filelist=None):
        self.list_labfile = np.array(find_files(C.PATH_CHORDLAB,ext=["lab","chords"]))[idx] if filelist is None else filelist
        self.list_labs = [self._loadlabel(f) for f in self.list_labfile]
        self.rand_shift = rand_shift
    def __len__(self):
        return len(self.list_labs)
    def _loadlabel(self,labfile):
        lablist = []
        f = open(labfile)
        line = f.readline()
        while (line != "") and (not line.isspace()):
            sp = line.split()
            lab = chord.lab2target_single(sp[2])
            if (len(lablist)==0) or (lablist[-1]!=lab):
                lablist.append(lab)
            line = f.readline()
        return np.array(lablist,dtype=np.int32)

    def get_example(self,i):
        labs = self.list_labs[i]
        startidx = np.random.randint(max([labs.shape[0]-8,1]))
        length = np.random.randint(8,32)
        labs = labs[startidx:startidx+length]
        if self.rand_shift:
            labs_shift = np.array(labs)
            shift = np.random.randint(12)
            for i in range(len(labs_shift)):
                if labs[i] == C.LABEL_N:
                    continue
                labs_shift[i] = (labs[i]//12*12) + ((labs[i]%12+shift)%12)
            return labs_shift

        return labs




class ChordDataset(dataset.DatasetMixin):
    def __init__(self,idx,rand_shift=False):
        self.list_labfile = np.array(find_files(C.PATH_CHORDLAB,ext=["lab","chords"]))[idx]
        #self.list_cqtfile = np.array(find_files(C.PATH_CQT,ext="npy"))[idx]
        self.list_featfile = np.array(find_files(C.PATH_FEAT,ext="npy"))[idx]
        #self.list_mfccfile = np.array(find_files(C.PATH_MFCC,ext="npy"))[idx]
        
        #self.list_cqt = [U.normalize_spec(np.load(f)) for f in self.list_cqtfile]
        self.list_feat = [np.load(f)[:,:C.N_DIMS_FEAT] for f in self.list_featfile]
        self.labs_list,self.lab_intervals_list = LoadLabelSet(self.list_labfile,C.label_shifts[idx])
        #self.list_mfcc = [np.load(f).T for f in self.list_mfccfile]
        
        self.rand_shift = rand_shift
        
    def __len__(self):
        return len(self.list_labfile)
    
    def get_example(self,i):
        feat = np.clip(self.list_feat[i],1e-4,1-(1e-4))
        #feat = self.list_cqt[i]
        #mfcc = self.list_mfcc[i]
        labs = self.labs_list[i]
        intervals = self.lab_intervals_list[i]
        if self.rand_shift:
            shift = np.random.randint(12)
            feat,labs = ShiftDatapair(feat,labs,shift)
        aligns = np.zeros(feat.shape[0],dtype=np.int32)
        assert intervals.shape[0] == len(labs)
        for i in range(len(labs)):
            aligns[intervals[i,0]:intervals[i,1]] = i
        
        if intervals[i,1] < len(aligns):
            feat = feat[:intervals[i,1],:]
            aligns = aligns[:intervals[i,1]]
            
        labs = chord.encode_chordseq_hierarchical(labs)[:,0]
        #labs = chord.encode_chordseq_hierarchical(labs)
        return feat,labs,aligns

class ChordDatasetSemisupervised(dataset.DatasetMixin):
    def __init__(self,idx,idx_un,rand_shift=False):
        self.list_labfile = np.array(find_files(C.PATH_CHORDLAB,ext=["lab","chords"]))[idx]
        #self.list_cqtfile = np.array(find_files(C.PATH_CQT,ext="npy"))[idx]
        self.list_featfile = np.array(find_files(C.PATH_FEAT,ext="npy"))[idx]
        #self.list_mfccfile = np.array(find_files(C.PATH_MFCC,ext="npy"))[idx]
        #self.list_cqtfile_u = np.array(find_files(C.PATH_CQT,ext="npy"))[idx_un]
        self.list_featfile_u = np.array(find_files(C.PATH_FEAT,ext="npy"))[idx_un]
        #self.list_mfccfile_u = np.array(find_files(C.PATH_MFCC,ext="npy"))[idx_un]
        
        #self.list_cqt = [U.normalize_spec(np.load(f)) for f in self.list_cqtfile]
        self.list_feat = [np.load(f)[:,:C.N_DIMS_FEAT] for f in self.list_featfile]
        self.labs_list,self.lab_intervals_list = LoadLabelSet(self.list_labfile,C.label_shifts[idx])
        #self.list_mfcc = [np.load(f).T for f in self.list_mfccfile]
        
        #self.list_cqt_u = [U.normalize_spec(np.load(f)) for f in self.list_cqtfile_u]
        self.list_feat_u = [np.load(f)[:,:C.N_DIMS_FEAT] for f in self.list_featfile_u]
        
        self.rand_shift = rand_shift
        
    def __len__(self):
        return len(self.list_labfile)
    
    def get_example(self,i):
        feat = np.clip(self.list_feat[i],1e-4,1-(1e-4))
        feat_u = np.clip(self.list_feat_u[np.random.randint(len(self.list_feat_u))],1e-4,1-(1e-4))
        #feat = self.list_cqt[i]
        #feat_u = self.list_cqt_u[np.random.randint(len(self.list_cqt_u))]
        #mfcc = self.list_mfcc[i]
        labs = self.labs_list[i]
        intervals = self.lab_intervals_list[i]
        
        if self.rand_shift:
            shift = np.random.randint(12)
            feat,labs = ShiftDatapair(feat,labs,shift)
            feat_u = ShiftDatapair(feat_u,None,shift)
        
        aligns = np.zeros(feat.shape[0],dtype=np.int32)
        assert intervals.shape[0] == len(labs)
        for i in range(len(labs)):
            aligns[intervals[i,0]:intervals[i,1]] = i
        if intervals[i,1] < len(aligns):
            feat = feat[:intervals[i,1],:]
            aligns = aligns[:intervals[i,1]]        
        labs  = chord.encode_chordseq_hierarchical(labs)[:,0]
        
        return feat,feat_u,labs,aligns 
    def chord_distribution(self):
        counts = np.zeros(C.N_VOCABULARY_TRIADS,dtype=np.int32) + 100
        for i in range(len(self)):
            for j in range(12):
                _,_,l,a = self[i]
                counts += np.bincount(l[a],minlength=C.N_VOCABULARY_TRIADS)
        dist = counts / counts.sum()
        return dist.astype(np.float32)
