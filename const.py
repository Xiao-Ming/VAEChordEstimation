#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 16:13:19 2018

@author: wuyiming
"""

import math

SR = 44100
H = 4096
#H = 2048

N_DIMS_FEAT = 36
N_DIMS_CLASSIFIER_LATENT = 256
N_DIMS_RNN_LATENT = 256
N_DIMS_VAE_LATENT = 64
N_DIMS_MFCC = 20
N_VOCABULARY_TRIADS = 12*8 + 1
N_SEVENTHS = 3+1
N_BASS = 13
N_BATCH = 32
N_CQT_OCTAVES = 7
N_CQT_PER_OCTAVE = 12
N_CQT_BINS = N_CQT_OCTAVES*N_CQT_PER_OCTAVE
N_EMBED = 15

SELF_TRANS_RATE = 0.9

VAE_RAND_SHIFT = False
VAE_RAND_SHIFT_QUALITY =False
VAE_SHIFT_REGULAR = False
VAE_DISCRIMINATE = False
VAE_ANNEAL_BETA = False
LANGUAGEMODEL = False
PSEUDO_PRIOR = False
CLASS_WEIGHT = False
MARKOV_REGULARIZE = True
GUMBEL_HARD = True
POSTFILTER = True
GENERATOR_SEMI = True

VAE_BETA = 1
VAE_BETA_ANNEAL_TO = 1
VAE_WEIGHT_ENTROPY = 1
JSD_RATIO = 32

BETA_DIST_VAR = 1e-2
BETA_MEAN_SCALE = 2*math.sqrt(.25-BETA_DIST_VAR)
BETA_MEAN_BIAS = (1-BETA_MEAN_SCALE) * 0.5

SEQLEN_TRAINING = 60 * SR // H

LABEL_N = N_VOCABULARY_TRIADS - 1
LABEL_SOS = N_VOCABULARY_TRIADS
LABEL_EOS = N_VOCABULARY_TRIADS + 1

#PATH_CQT = "/media/wu/TOSHIBA EXT/HCQT_12_hop512"
#PATH_CQT = "/home/wu/Projects/ChordData/CQT_H_12"
#PATH_CQT = "/n/sd3/wu/ChordData/CQT_H_12"
PATH_CHORDLAB = "dataset/chordlab"

#PATH_AUDIO = "/n/work1/wu/ChordData/Audio"

#PATH_MFCC = "/home/wu/Projects/ChordData/mfcc"
#PATH_STFT = "/media/wu/TOSHIBA EXT/STFT"
PATH_FEAT = "dataset/chroma"
#PATH_FEAT = "/n/work1/wu/feat_songle"
#PATH_FEAT = "/n/work1/wu/ChordData/McGill_chordino"
#PATH_BEAT = "/n/sd3/wu/ChordData/trackbeat_madmom"
#PATH_BEAT = "../ChordData/trackbeat"
PATH_ESTIMATE_CROSS = "estimated_cross"

import numpy as np

label_shifts = np.zeros(2000,dtype=int)
label_shifts[[393,1184,1221,1337]] = -1
label_shifts[[615,678,902,1197,1217,1485,1519,1532]] = 1

label_exclude = [648,1092,1239,1368,1425,1430,1477]
