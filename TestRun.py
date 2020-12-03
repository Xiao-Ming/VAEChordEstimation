#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 14:13:59 2019

@author: wuyiming
"""

import vae
import net_generative as gen
from librosa.core import istft
from librosa.output import write_wav
import numpy as np
import const as C
import dataset as D
import chainer
from librosa.display import specshow
import matplotlib.pyplot as plt
import util as U

dset = D.ChordDatasetSTFTSemisupervised([300])

#model = gen.GenerativeChordnet()
model = vae.VUNetSTFT()
model.load("chromavae.model")

feat,labs,align = dset[0]
labs_onehot = U.encode_onehot(labs)
labs_onehot_blur = labs_onehot + 0.2
labs_onehot_blur /= labs_onehot_blur.sum(axis=1,keepdims=True)
phase = dset.getstftphase(0)

chainer.config.train = False
chainer.config.enable_backprop = False

generated = model.generate_encode_condition(feat[None,...],labs_onehot[None,align,:])[0]
#generated = model.reconst(feat)

plt.subplot(3,1,1)
specshow(feat[:1024,:64].T)

plt.subplot(3,1,2)
specshow(generated.data[:1024,:64].T * 100)

labs_mod = U.encode_onehot((labs+3)%C.N_VOCABULARY_TRIADS)

generated_mod = model.generate_encode_condition(feat[None,...],labs_mod[None,align,:])[0]
plt.subplot(3,1,3)
specshow(generated_mod.data[:1024,:64].T * 100)

generated_pad = np.zeros(phase.shape,dtype = np.float32)
generated_pad[:,:1024] = generated.data
wav_reconst = istft((generated_pad*feat.max()*phase).T,win_length=2048,hop_length=512)

write_wav("reconst.wav",wav_reconst,16000,norm=True)

generated_pad[:,:1024] = generated_mod.data
wav_reconst = istft((generated_pad*feat.max()*phase).T,win_length=2048,hop_length=512)

write_wav("reconst_mod.wav",wav_reconst,16000,norm=True)