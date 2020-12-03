#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 19:27:44 2019

@author: wu
"""

import numpy as np
from librosa.util import find_files
import jams
import const as C

idx_rwc = []
idx_uspop = []
idx_billboard = []
idx_isophonics = []
flist = find_files("/n/work1/wu/ChordData/chord_data_1217/references_v2",ext="jams")
for i in range(len(flist)):
    if i+320 in C.label_exclude:
        continue
    jamfile = jams.load(flist[i])
    fname = jamfile["sandbox"]["original_filename"]
    if fname.startswith("N"):
        idx_rwc.append(i)
        continue
    corpus = jamfile["annotations"][0]["annotation_metadata"]["corpus"]
    if corpus == "Billboard-Chords":
        idx_billboard.append(i)
    elif corpus == "Isophonics":
        idx_isophonics.append(i)
    elif corpus == "MARL-Chords":
        idx_uspop.append(i)

idx_rwc = np.array(idx_rwc,dtype=np.int32)
idx_uspop = np.array(idx_uspop,dtype=np.int32)
idx_billboard = np.array(idx_billboard,dtype=np.int32)
idx_isophonics = np.array(idx_isophonics,dtype=np.int32)

idx_rwc = np.array_split(idx_rwc,5)
idx_uspop = np.array_split(idx_uspop,5)
idx_billboard = np.array_split(idx_billboard,5)
idx_isophonics = np.array_split(idx_isophonics,5)

np.save("folds_rwc.npy",idx_rwc)
np.save("folds_uspop.npy",idx_uspop)
np.save("folds_billboard.npy",idx_billboard)
np.save("folds_isophonics.npy",idx_isophonics)

