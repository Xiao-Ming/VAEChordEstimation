#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:24:41 2019

@author: wu
"""


import numpy as np
import pandas
from librosa.util import find_files
import vamp
from librosa.core import load
import const as C

"""
filelist = find_files("/n/work1/wu/ChordData/McGill_chordino",ext="csv")

for f in filelist:
    print(f)
    csv = pandas.read_csv(f,header=None)
    chroma = csv.values[:,2:].astype(np.float32)
    chroma = np.log(1+chroma) + 0.01
    chroma /= chroma.max()
    chroma *= 0.99
    assert(chroma.min() > 0)
    assert(chroma.max() < 1)
    np.save(f+".npy",chroma)
"""
"""
filelist = find_files("/n/work1/wu/ChordData/McGill_chordlab",ext="lab")

for f in filelist:
    print(f)
    with open(f,"r") as fin:
        data = fin.read().splitlines(True)
    with open(f,"w") as fout:
        fout.writelines(data[:-1])
        fout.close()
"""


filelist = find_files("/n/work1/wu/ChordData/AudioNoSupervise")
filelist.extend(find_files("/n/sd8/MusicDatabase/JPOP"))

for f in filelist:
    print(f)
    fname = f.split("/")[-1]
    y,sr = load(f,sr=C.SR)
    _, chroma = vamp.collect(y,sr,"nnls-chroma:nnls-chroma","bothchroma")["matrix"]
    chroma = np.log(1+chroma) + 0.01
    chroma /= chroma.max()
    chroma *= 0.99
    np.save("/n/work1/wu/ChordData/McGill_chordino/unsupervise/"+fname+".npy",chroma)