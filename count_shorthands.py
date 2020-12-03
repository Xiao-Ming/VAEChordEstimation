#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 17:54:08 2019

@author: wu
"""

import mir_eval
import numpy as np
import matplotlib.pyplot as plt
from chord import quality_triad_map
from librosa.util import find_files
import const as C

lablist = find_files(C.PATH_CHORDLAB,ext=["lab","chords"])[320:320+1217]

quality_durations = [0,0,0,0,0,0,0,0]

for lab in lablist:
    intervals,labels = mir_eval.io.load_labeled_intervals(lab)
    durations = mir_eval.util.intervals_to_durations(intervals)
    for l,i in zip(labels,durations):
        shorthand = mir_eval.chord.split(l)[1]
        quality_durations[quality_triad_map[shorthand]] += i

labels = ["maj","min","aug","dim","sus4","sus2","1","5"]
heights = np.array(quality_durations) / 3600.0

plt.barh(np.arange(len(labels)),heights[::-1],tick_label=labels[::-1],ec="black",color="blue")
plt.xticks([0,25,50],fontname="STIXGeneral",fontsize=15)
plt.yticks(fontname="STIXGeneral",fontsize=15)
plt.xlabel("Duration (hours)",fontname="STIXGeneral",fontsize=15)