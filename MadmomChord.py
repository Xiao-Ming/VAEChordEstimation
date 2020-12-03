#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 18:09:16 2020

@author: wu
"""

from madmom.features.chords import CRFChordRecognitionProcessor,CNNChordFeatureProcessor
from librosa.util import find_files
import numpy as np
import os.path


filelist = find_files("/home/wu/Projects/ChordRecGenerative/chords_songle")

featproc = CNNChordFeatureProcessor()
decode = CRFChordRecognitionProcessor()

for i in range(1690000):
    if not os.path.isdir("/home/wu/Projects/ChordRecGenerative/chords_songle/%d" % i):
        continue
    print("song id:%d" % i)
    feats = featproc("/home/wu/Projects/ChordRecGenerative/chords_songle/%d/audio.mp3" % i)
    res = decode(feats)
    
    f = open("est_songle_madmom/%d.lab" % i,"w")
    for row in res:
        f.write("%.4f %.4f %s\n" % (row[0],row[1],row[2]))
    f.close()

