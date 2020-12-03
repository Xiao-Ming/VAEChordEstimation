#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 21:55:32 2018

@author: wuyiming
"""

import mir_eval.chord as chord
import mir_eval
import numpy as np
import util as U
import const as C
#from const import xp
from librosa.util import find_files
import os

"""
quality_triad_map = {
    'maj':     0,
    'min':     1,
    'aug':     2,
    'dim':     3,
    'sus4':    4,
    'sus2':    5,
    '7':       6,
    'maj7':    7,
    'min7':    8,
    'minmaj7': 9,
    'maj6':    0,
    'min6':    1,
    'dim7':    10,
    'hdim7':   10,
    'maj9':    7,
    'min9':    8,
    '9':       6,
    'b9':      6,
    '#9':      6,
    'min11':   8,
    '11':      6,
    '#11':     6,
    'maj13':   7,
    'min13':   8,
    '13':      6,
    'b13':     6,
    '1':       0,
    '5':       0,
    '' :       0
}
"""
quality_triad_map = {
    'maj':     0,
    'min':     1,
    'aug':     2,
    'dim':     3,
    'sus4':    4,
    'sus2':    5,
    '7':       0,
    'maj7':    0,
    'min7':    1,
    'minmaj7': 1,
    'maj6':    0,
    'min6':    1,
    'dim7':    3,
    'hdim7':   3,
    'maj9':    0,
    'min9':    1,
    '9':       0,
    'b9':      0,
    '#9':      0,
    'min11':   1,
    '11':      0,
    '#11':     0,
    'maj13':   0,
    'min13':   1,
    '13':      0,
    'b13':     0,
    '1':       6,
    '5':       7,
    '' :       0
}

quality_sevenths_map = {
    'maj':     3,
    'min':     3,
    'aug':     3,
    'dim':     3,
    'sus4':    3,
    'sus2':    3,
    '7':       1,
    'maj7':    2,
    'min7':    1,
    'minmaj7': 2,
    'maj6':    0,
    'min6':    0,
    'dim7':    0,
    'hdim7':   1,
    'maj9':    2,
    'min9':    1,
    '9':       1,
    'b9':      1,
    '#9':      1,
    'min11':   1,
    '11':      1,
    '#11':     1,
    'maj13':   2,
    'min13':   1,
    '13':      1,
    'b13':     1,
    '1':       3,
    '5':       3,
    '' :       3
}

quality_ninths_map = {
    'maj':     3,
    'min':     3,
    'aug':     3,
    'dim':     3,
    'sus4':    3,
    'sus2':    3,
    '7':       3,
    'maj7':    3,
    'min7':    3,
    'minmaj7': 3,
    'maj6':    3,
    'min6':    3,
    'dim7':    3,
    'hdim7':   3,
    'maj9':    1,
    'min9':    1,
    '9':       1,
    'b9':      0,
    '#9':      2,
    'min11':   1,
    '11':      1,
    '#11':     1,
    'maj13':   1,
    'min13':   1,
    '13':      1,
    'b13':     1,
    '1':       3,
    '5':       3,
    '' :       3
}

quality_elevenths_map = {
    'maj':     2,
    'min':     2,
    'aug':     2,
    'dim':     2,
    'sus4':    2,
    'sus2':    2,
    '7':       2,
    'maj7':    2,
    'min7':    2,
    'minmaj7': 2,
    'maj6':    2,
    'min6':    2,
    'dim7':    2,
    'hdim7':   2,
    'maj9':    2,
    'min9':    2,
    '9':       2,
    'b9':      2,
    '#9':      2,
    'min11':   0,
    '11':      0,
    '#11':     1,
    'maj13':   0,
    'min13':   0,
    '13':      0,
    'b13':     0,
    '1':       2,
    '5':       2,
    '' :       2
}

quality_thirteenths_map = {
    'maj':     2,
    'min':     2,
    'aug':     2,
    'dim':     2,
    'sus4':    2,
    'sus2':    2,
    '7':       2,
    'maj7':    2,
    'min7':    2,
    'minmaj7': 2,
    'maj6':    2,
    'min6':    2,
    'dim7':    2,
    'hdim7':   2,
    'maj9':    2,
    'min9':    2,
    '9':       2,
    'b9':      2,
    '#9':      2,
    'min11':   2,
    '11':      2,
    '#11':     2,
    'maj13':   1,
    'min13':   1,
    '13':      1,
    'b13':     0,
    '1':       2,
    '5':       2,
    '' :       2
}



notes = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
bass_notes = ["","b2","2","b3","3","4","b5","5","b6","6","b7","7"]
#triad_qualities = ["maj","min","aug","dim","sus4","sus2","7","maj7","min7","minmaj7","dim7"]
triad_qualities = ["maj","min","aug","dim","sus4","sus2","1","5"]
seventh_qualities = [["maj6","min6","aug","dim7","sus4","sus2"],
                     ["7","min7","aug","hdim7","sus4","sus2"],
                     ["maj7","minmaj7","aug","dim","sus4","sus2"]]
ninth_qualities = [["b9","min","aug","dim","sus4","sus2"],
                   ["9","min9","aug","dim","sus4","sus2"],
                   ["#9","min","aug","dim","sus4","sus2"]]
eleventh_qualities = [["11","min11","aug","dim","sus4","sus2"],
                      ["#11","min","aug","dim","sus4","sus2"]]
thirteenth_qualities = [["b13","min","aug","dim","sus4","sus2"],
                        ["13","min","aug","dim","sus4","sus2"]]



triad2sevenths = [["maj","min","aug","dim","sus4","sus2"],
                 ["7","min7","aug","hdim7","sus4","sus2"],
                 ["maj7","minmaj7","aug","dim","sus4","sus2"]]

def encode_chordseq_hierarchical(seq):
    vec = np.zeros((len(seq),6),dtype=np.int32)
    for i in range(len(seq)):
        lab = seq[i]
        if (lab != "N"):
            root,bitmap,bass = chord.encode(lab,reduce_extended_chords=False)
            _,quality,scale,_ = chord.split(lab,reduce_extended_chords=False)
            vec[i,0] = quality_triad_map[quality]*12+root
            vec[i,1] = (root+bass)%12
            vec[i,2] = quality_sevenths_map[quality]
            vec[i,3] = quality_ninths_map[quality]
            vec[i,4] = quality_elevenths_map[quality]
            vec[i,5] = quality_thirteenths_map[quality]
            """
            for sc in scale:
                tone = chord.scale_degree_to_semitone(sc)
                if tone in [9,10,11]:
                    vec[i,2] = tone - 9
                elif tone in [13,14,15]:
                    vec[i,3] = tone - 13
                elif tone in [17,18]:
                    vec[i,4] = tone - 17
                elif tone in [20,21]:
                    vec[i,5] = tone - 20
            """
        elif lab=="N" or lab=="X":
            vec[i,:] = [C.LABEL_N,12,3,3,2,2]
    return vec

def lab2target_single(lab):
    if lab=="N" or lab=="X":
        return C.LABEL_N
    root,_,bass = chord.encode(lab,reduce_extended_chords=True)
    _,quality,_,_ = chord.split(lab,reduce_extended_chords=True)
    target = quality_triad_map[quality] * 12 + root
    return target

def lab2target(seq,module=np):
    seq_target = []
    for lab in seq:
        if lab=="N":
            target = C.LABEL_N
        else:
            _,quality,_,_ = chord.split(lab,reduce_extended_chords=True)
            root,_,bass = chord.encode(lab,reduce_extended_chords=True)
            quality_id = quality_triad_map[quality]
            target = quality_id*12 + root
        seq_target.append(target)
    return module.array(seq_target,dtype=np.int32)

def id2signature_largevoc(id_triad,id_bass,id_sevenths,id_ninth,id_eleventh,id_thirteenth):
    if id_triad==C.LABEL_N:
        return "N"
    note_root = notes[id_triad%12]
    note_bass = bass_notes[(id_bass - id_triad%12)%12] if id_bass<12 else ""
    quality = triad_qualities[id_triad//12]
    if (id_thirteenth!=2) and (id_eleventh!=2) and (id_ninth!=3) and (id_sevenths!=3):
        quality = thirteenth_qualities[id_thirteenth][id_triad/12]
    elif (id_eleventh!=2) and (id_ninth!=3) and (id_sevenths!=3):
        quality = eleventh_qualities[id_eleventh][id_triad//12]
    elif (id_ninth!=3) and (id_sevenths!=3):
        quality = ninth_qualities[id_ninth][id_triad//12]
    elif (id_sevenths!=3):
        quality = seventh_qualities[id_sevenths][id_triad//12]
    scales = []
    sig = chord.join(note_root,quality,scales,note_bass)
    return sig

def id2signature(id_triad):
    if id_triad==C.LABEL_N:
        return "N"
    note_root = notes[id_triad%12]
    quality = triad_qualities[id_triad//12]
    
    sig = note_root + ":" + quality
    return sig

def shift_label(l,shift):
    if l in ["N","X"]:
        return l
    if shift==0:
        return l
    l_split = chord.split(l)
    l_root,_,_ = chord.encode(l)
    l_root = (l_root + shift) % 12
    l_root_str = notes[l_root]
    return chord.join(l_root_str, l_split[1], l_split[2], l_split[3])

"""
def id2qualityID(id_triad,id_seventh):

    qualities in confusion matrix:
        maj/min/aug/dim/sus4/sus2/7/min7/hdim7/maj7/minmaj7

    _triad = id_triad // 12
    if id_seventh == 0:
        return _triad
    if id_seventh == 1:
        if _triad in [2,4,5]:
            return _triad
        elif _triad == 0:
            return 6
        elif _triad == 1:
            return 7
        else:
            return 8
    if id_seventh == 2:
        if _triad in [2,3,4,5]:
            return _triad
        else:
            return 9+_triad
"""
def id2qualityID(id_triad):
    return id_triad // 12

def shift_class(l,shift):
    if l==C.LABEL_N:
        return l
    else:
        return (l+shift)%12 + 12*(l//12)

def class2label(seq,for_eval=True):
    ret = []
    for i in range(len(seq)):
        assert seq[i]<C.N_VOCABULARY_TRIADS+2,"invalid sequence: %s" % seq
        if seq[i]==C.LABEL_N:
            symbol = "N"
        else:
            note_id = seq[i] % 12
            qual_id = seq[i] // 12
            symbol = notes[note_id] + ":" + triad_qualities[qual_id]
        ret.append(symbol)
    
    return ret
        
from scipy.spatial.distance import cosine

def evaluate_feature(feat,labstr):
    root,bitmap,bass = chord.encode(labstr,reduce_extended_chords=True)
    bitmap = bitmap.astype(np.float32)
    vec_bass = np.zeros(12,dtype=np.float32)
    vec_bass[(root+bass)%12] = 1
    vec_template = np.concatenate((vec_bass,bitmap))
    score = 1 - cosine(feat,vec_template)
    return score

def template_matching_matrix(feat,lablist):
    templates = []
    for lab in lablist:
        root,bitmap,bass = chord.encode(lab,reduce_extended_chords=True)
        bitmap = bitmap.astype(np.float32)
        vec_bass = np.zeros(12,dtype=np.float32)
        vec_bass[(root+bass)%12] = 1
        vec_template = np.concatenate((vec_bass,bitmap))
        templates.append(vec_template)
    templates = np.stack(templates,axis=1)
    
    xp = C.cupy.get_array_module(feat)
    prob_mat = xp.matmul(feat,xp.asarray(templates))
    prob_mat_logz = xp.log(xp.sum(xp.exp(prob_mat),axis=1,keepdims=True))
    prob_mat -= prob_mat_logz
    
    return prob_mat
        
def template_matching_matrix_along_class(feat):
    templates = []
    lablist = []
    lablist.extend(notes)
    lablist.extend([n+":min" for n in notes])
    lablist.append("N")
    for lab in lablist:
        root,bitmap,bass = chord.encode(lab,reduce_extended_chords=True)
        bitmap = bitmap.astype(np.float32)
        vec_bass = np.zeros(12,dtype=np.float32)
        vec_bass[(root+bass)%12] = 1
        vec_template = np.concatenate((vec_bass,bitmap))
        templates.append(vec_template)
    templates = np.stack(templates,axis=1)
    
    xp = C.cupy.get_array_module(feat)
    prob_mat = xp.exp(xp.matmul(feat,xp.asarray(templates)))
    prob_mat_sum = xp.sum(prob_mat,axis=1,keepdims=True)
    prob_mat = prob_mat/prob_mat_sum
    
    return prob_mat

def filter_label_seq(labs,shift=0):
    new_seq = []
    for l in labs:
        if l=="N" or l=="X":
            l_new = l
        else:
            sp = chord.split(l)
            root,_,_ = chord.encode(l)
            if shift != 0:
                root_new = (root + shift) % 12
                sp[0] = notes[root_new]
            l_new = chord.join(sp[0],sp[1],None,sp[3])
        new_seq.append(l_new)
    return new_seq

def EvaluateLabels(idx,shift=None):
    lab_list = np.array(find_files(C.PATH_CHORDLAB,ext=["lab","chords"]))[idx]
    estimated_lab_list = [U.path_estimated_lab(p) for p in lab_list]
    
    if shift is None:
        shift = C.label_shifts[idx]
    
    list_intervals = []
    list_accrs_majmin = []
    list_accrs_mirex = []
    list_accrs_triads = []
    list_accrs_tetrads = []
    i=0
    for lab_file,estimated_lab_file in zip(lab_list,estimated_lab_list):
        #print(estimated_lab_file + "  " + lab_file)
        #assert lab_file.split("/")[1] == estimated_lab_file.split("/")[-1], "Evaluation error: filename does not match!"
        (ref_intervals,ref_labels) = mir_eval.io.load_labeled_intervals(lab_file)
        (est_intervals,est_labels) = mir_eval.io.load_labeled_intervals(estimated_lab_file)
        ref_labels, est_labels = filter_label_seq(ref_labels,shift[i]), filter_label_seq(est_labels)
        est_intervals, est_labels = chord.util.adjust_intervals(
            est_intervals, est_labels, ref_intervals.min(), ref_intervals.max(),
            chord.NO_CHORD, chord.NO_CHORD)
        # Adjust the labels so that they span the same intervals
        intervals, ref_labels, est_labels = chord.util.merge_labeled_intervals(
            ref_intervals, ref_labels, est_intervals, est_labels)
        durations = mir_eval.util.intervals_to_durations(intervals)
        #evaluate = mir_eval.chord.evaluate(ref_intervals,ref_labels,est_intervals,est_labels)
        
        list_accrs_majmin.append(chord.weighted_accuracy(chord.majmin(ref_labels,est_labels),durations))
        list_accrs_triads.append(chord.weighted_accuracy(chord.triads(ref_labels,est_labels),durations))
        list_accrs_tetrads.append(chord.weighted_accuracy(chord.tetrads(ref_labels,est_labels),durations))
        list_accrs_mirex.append(chord.weighted_accuracy(chord.root(ref_labels,est_labels),durations))
        list_intervals.append(np.sum(durations))
        i+=1
    
    #conf_matrix_root,conf_matrix_qual = ConfMatrix(idx)
    conf_matrix_root,conf_matrix_qual = 0,0
    return np.array(list_accrs_majmin),np.array(list_accrs_triads),np.array(list_accrs_mirex),np.array(list_intervals),conf_matrix_root,conf_matrix_qual

def TransitionsFreq(idx):
    def norepeat(arr):
        res = []
        cur_n = -1
        for n in arr:
            if n!=cur_n:
                res.append(n)
                cur_n = n
        return res
    
    lab_list = np.array(find_files(C.PATH_CHORDLAB,ext=["lab","chords"]))[idx]
    estimated_lab_list = [U.path_estimated_lab(p) for p in lab_list]
    freq_list = []
    for labfile in estimated_lab_list:
        intervals,labels = mir_eval.io.load_labeled_intervals(labfile)
        labels = norepeat(lab2target(labels))
        duration = intervals[-1,-1] / 60.0
        freq = (len(labels)-1) / duration
        freq_list.append(freq)
    
    return np.mean(freq_list)
    

def ConfMatrix(idx):
    """
    *Root matrix
    *Quality (maj,min,dim,aug,sus4,sus2)
    """
    lab_list = np.array(find_files(C.PATH_CHORDLAB,ext=["lab","chords"]))[idx]
    estimated_lab_list = [os.path.join(C.PATH_ESTIMATE_CROSS,p.split("/")[-2],p.split("/")[-1]) for p in lab_list]
    durations = np.array([])
    confmatrix_root = np.zeros((12,12)) + 0.01
    confmatrix_qual = np.zeros((8,8)) + 0.01
    #confmatrix_quality = np.zeros((6,6))
    #confmatrix_bass = np.zeros((3,3))
    for labfile,estfile in zip(lab_list,estimated_lab_list):
        (ref_intervals,ref_labels) = mir_eval.io.load_labeled_intervals(labfile)
        (est_intervals,est_labels) = mir_eval.io.load_labeled_intervals(estfile)
        est_intervals,est_labels = mir_eval.util.adjust_intervals(est_intervals,est_labels,ref_intervals.min(),ref_intervals.max(),
                                                                  mir_eval.chord.NO_CHORD,mir_eval.chord.NO_CHORD)
        (intervals,ref_labels,est_labels) = mir_eval.util.merge_labeled_intervals(ref_intervals,ref_labels,est_intervals,est_labels)
        durations = mir_eval.util.intervals_to_durations(intervals)
        ref_labels_id = encode_chordseq_hierarchical(ref_labels)
        est_labels_id = encode_chordseq_hierarchical(est_labels)
        for i in range(len(ref_labels)):
            if ref_labels_id[i,0]==C.LABEL_N or est_labels_id[i,0]==C.LABEL_N:
                continue
            confmatrix_root[ref_labels_id[i,0]%12,est_labels_id[i,0]%12] += durations[i]
            if ref_labels_id[i,0]%12==est_labels_id[i,0]%12:
                confmatrix_qual[id2qualityID(ref_labels_id[i,0]),id2qualityID(est_labels_id[i,0])] += durations[i]
            #if (ref_labels_id[i]%12)==(est_labels_id[i]%12):
            #    confmatrix_quality[ref,est_q] += durations[i]
    confmatrix_root /= np.sum(confmatrix_root,axis=1,keepdims=True)
    confmatrix_qual /= np.sum(confmatrix_qual,axis=1,keepdims=True)
    return confmatrix_root,confmatrix_qual

def ConfMatrix_Allclass(idx):
    lab_list = np.array(find_files(C.PATH_CHORDLAB,ext=["lab","chords"]))[idx]
    estimated_lab_list = [os.path.join(C.PATH_ESTIMATE_CROSS,p.split("/")[-2],p.split("/")[-1]) for p in lab_list]
    durations = np.array([])
    confmatrix = np.zeros((C.N_VOCABULARY_TRIADS,C.N_VOCABULARY_TRIADS))

    for labfile,estfile in zip(lab_list,estimated_lab_list):
        (ref_intervals,ref_labels) = mir_eval.io.load_labeled_intervals(labfile)
        (est_intervals,est_labels) = mir_eval.io.load_labeled_intervals(estfile)
        est_intervals,est_labels = mir_eval.util.adjust_intervals(est_intervals,est_labels,ref_intervals.min(),ref_intervals.max(),
                                                                  mir_eval.chord.NO_CHORD,mir_eval.chord.NO_CHORD)
        (intervals,ref_labels,est_labels) = mir_eval.util.merge_labeled_intervals(ref_intervals,ref_labels,est_intervals,est_labels)
        durations = mir_eval.util.intervals_to_durations(intervals)
        ref_labels_id = encode_chordseq_hierarchical(ref_labels)
        est_labels_id = encode_chordseq_hierarchical(est_labels)
        for i in range(len(ref_labels)):
            confmatrix[ref_labels_id[i,0],est_labels_id[i,0]] += durations[i]

    confmatrix /= np.sum(confmatrix,axis=1,keepdims=True)
    return confmatrix


def SearchErrorExample(idx,ref,est):
    lab_list = np.array(find_files(C.PATH_CHORDLAB,ext=["lab","chords"]))[idx]
    estimated_lab_list = [os.path.join(C.PATH_ESTIMATE_CROSS,p.split("/")[-2],p.split("/")[-1]) for p in lab_list]
    list_examples = []
    for labfile,estfile in zip(lab_list,estimated_lab_list):
        (ref_intervals,ref_labels) = mir_eval.io.load_labeled_intervals(labfile)
        (est_intervals,est_labels) = mir_eval.io.load_labeled_intervals(estfile)
        est_intervals,est_labels = mir_eval.util.adjust_intervals(est_intervals,est_labels,ref_intervals.min(),ref_intervals.max(),
                                                                  mir_eval.chord.NO_CHORD,mir_eval.chord.NO_CHORD)
        (intervals,ref_labels,est_labels) = mir_eval.util.merge_labeled_intervals(ref_intervals,ref_labels,est_intervals,est_labels)
        durations = mir_eval.util.intervals_to_durations(intervals)
        ref_labels_id = encode_chordseq_hierarchical(ref_labels)
        est_labels_id = encode_chordseq_hierarchical(est_labels)
        for i in range(len(ref_labels)):
            if durations[i]>0.5 and ref_labels_id[i,0]==ref and est_labels_id[i,0]==est:
                list_examples.append((labfile,ref,est,intervals[i]))

    return list_examples



template_matrix = np.zeros((C.N_VOCABULARY_TRIADS,36),dtype=np.float32)
lablist = class2label(range(C.N_VOCABULARY_TRIADS))
for i in range(C.N_VOCABULARY_TRIADS):
    root,bitmap,bass = chord.encode(lablist[i],reduce_extended_chords=True)
    template_matrix[i,root] = 1
    template_matrix[i,12:24] = np.roll(bitmap,root)
template_matrix_zero_center = np.array(template_matrix)
template_matrix_zero_center[:,:24] = template_matrix_zero_center[:,:24] * 4 - 2

"""
template_matrix = np.zeros((C.N_VOCABULARY_TRIADS,C.N_DIMS_FEAT),dtype=np.float32) + 1e-2
lablist = class2label(range(C.N_VOCABULARY_TRIADS))
for i in range(C.N_VOCABULARY_TRIADS):
    root,bitmap,bass = chord.encode(lablist[i],reduce_extended_chords=True)
    templ = np.roll(bitmap,root)
    template_matrix[i,:] = np.concatenate([templ for _ in range(C.N_DIMS_FEAT//12)])
"""


label_shifter = np.zeros((C.N_VOCABULARY_TRIADS,C.N_VOCABULARY_TRIADS,C.N_VOCABULARY_TRIADS,C.N_VOCABULARY_TRIADS),dtype=np.float32)
for i in range(C.N_VOCABULARY_TRIADS-1):
    for j in range(C.N_VOCABULARY_TRIADS-1):
        mat = np.identity(C.N_VOCABULARY_TRIADS,dtype=np.float32)
        i_qual, j_qual, i_root, j_root = i//12, j//12, i%12, j%12
        shift_root = (j_root-i_root)%12
        mat[i,i] = 0
        mat[j,j] = 0
        mat[i,j] = 1
        label_shifter[i,j,:,:] = mat
label_shifter[C.LABEL_N,C.LABEL_N,:,:] = np.identity(C.N_VOCABULARY_TRIADS,dtype=np.float32)


chroma_shifter = np.zeros((C.N_VOCABULARY_TRIADS,C.N_VOCABULARY_TRIADS,36,36),dtype=np.float32)
for i in range(C.N_VOCABULARY_TRIADS-1):
    for j in range(C.N_VOCABULARY_TRIADS-1):
        mat_mid = np.identity(12,dtype=np.float32)
        #mat_bass = np.identity(12,dtype=np.float32)
        i_qual, j_qual, i_root, j_root = i//12, j//12, i%12, j%12
        if i_qual==0:
            if j_qual == 1:
                mat_mid[np.array([i_root+3,i_root+4,i_root+3,i_root+4])%12,np.array([i_root+3,i_root+3,i_root+4,i_root+4])%12] = [0,1,1,0]
            elif j_qual == 2:
                mat_mid[np.array([i_root+7,i_root+8,i_root+7,i_root+8])%12,np.array([i_root+7,i_root+7,i_root+8,i_root+8])%12] = [0,1,1,0]
            elif j_qual == 3:
                mat_mid[np.array([i_root+3,i_root+4,i_root+3,i_root+4])%12,np.array([i_root+3,i_root+3,i_root+4,i_root+4])%12] = [0,1,1,0]
                mat_mid[np.array([i_root+6,i_root+7,i_root+6,i_root+7])%12,np.array([i_root+6,i_root+6,i_root+7,i_root+7])%12] = [0,1,1,0]
            elif j_qual == 4:
                mat_mid[np.array([i_root+4,i_root+5,i_root+4,i_root+5])%12,np.array([i_root+4,i_root+4,i_root+5,i_root+5])%12] = [0,1,1,0]
            elif j_qual == 5:
                mat_mid[np.array([i_root+2,i_root+4,i_root+2,i_root+4])%12,np.array([i_root+2,i_root+2,i_root+4,i_root+4])%12] = [0,1,1,0]
            else:
                pass
        if i_qual == 1:
            if j_qual == 0:
                mat_mid[np.array([i_root+3,i_root+4,i_root+3,i_root+4])%12,np.array([i_root+3,i_root+3,i_root+4,i_root+4])%12] = [0,1,1,0]
            elif j_qual == 2:
                mat_mid[np.array([i_root+7,i_root+8,i_root+7,i_root+8])%12,np.array([i_root+7,i_root+7,i_root+8,i_root+8])%12] = [0,1,1,0]
                mat_mid[np.array([i_root+3,i_root+4,i_root+3,i_root+4])%12,np.array([i_root+3,i_root+3,i_root+4,i_root+4])%12] = [0,1,1,0]
            elif j_qual == 3:
                mat_mid[np.array([i_root+6,i_root+7,i_root+6,i_root+7])%12,np.array([i_root+6,i_root+6,i_root+7,i_root+7])%12] = [0,1,1,0]
            elif j_qual == 4:
                mat_mid[np.array([i_root+3,i_root+5,i_root+3,i_root+5])%12,np.array([i_root+4,i_root+4,i_root+5,i_root+5])%12] = [0,1,1,0]
            elif j_qual == 5:
                mat_mid[np.array([i_root+2,i_root+3,i_root+2,i_root+3])%12,np.array([i_root+2,i_root+2,i_root+3,i_root+3])%12] = [0,1,1,0]
            else:
                pass

        shift_root = (j_root - i_root) % 12
        mat_mid = np.roll(mat_mid,shift_root,axis=1)
        #mat_bass[:12,:12] = mat_mid
        chroma_shifter[i,j,:12,:12] = mat_mid
        chroma_shifter[i,j,12:24,12:24] = mat_mid
        chroma_shifter[i,j,24:36,24:36] = mat_mid

chroma_shifter[C.LABEL_N,:,:,:] = np.identity(36,dtype=np.float32)
chroma_shifter[:,C.LABEL_N,:,:] = np.identity(36,dtype=np.float32)
chroma_shifter = chroma_shifter[:,:,:C.N_DIMS_FEAT,:C.N_DIMS_FEAT]


bass_shifter = np.zeros((C.N_BASS,C.N_BASS,C.N_BASS,C.N_BASS),dtype=np.float32)
for i in range(C.N_BASS-1):
    for j in range(C.N_BASS-1):
        bass_shifter[i,j,:12,:12] = np.roll(np.identity(12,dtype=np.float32),shift = (j-i)%12)
        bass_shifter[i,j,12,12] = 1
bass_shifter[12,12,:,:] = np.identity(13)








