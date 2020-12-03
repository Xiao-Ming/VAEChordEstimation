#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 18:13:05 2019

@author: wuyiming
"""

from librosa.core import load,stft
from librosa.util import find_files
from librosa.feature import mfcc
import os.path
import numpy as np

import const as C

audio_list = find_files(C.PATH_AUDIO)
i = 1
for audiofile in audio_list:
    albname = audiofile.split("/")[-2]
    fname = audiofile.split("/")[-1].split(".")[0]
    print("%d/%d:%s/%s" % (i,len(audio_list),albname,fname))
    y,sr = load(audiofile,16000)
    coef = stft(y,n_fft=2048,hop_length=512)

    
    p = os.path.join(C.PATH_STFT,albname,fname+".npy")
    np.save(p,coef)
    i+=1