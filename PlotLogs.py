#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 19:14:52 2019

@author: wu
"""

import matplotlib.pyplot as plt
import json
import argparse
import numpy as np
import os.path

parser = argparse.ArgumentParser()
parser.add_argument('-l','--location',\
                    action='store',\
                    default="Regularize",
                    type=str)

args = parser.parse_args()
location = args.location

def AveragedScores(list_json,keyword):
    scores = [[] for _ in range(len(list_json))]
    for i in range(len(list_json)):
        for epoch in list_json[i]:
            scores[i].append(epoch[keyword])
    
    scores = np.array(scores)
    return np.mean(scores,axis=0)

for beta in [0.5,1,2]:
    list_json_supervised1 = [json.load(open(os.path.join("result",location,"supervised1_fold%d_beta%02d.log" % (f,beta)))) for f in range(5)]
    list_json_supervised2 = [json.load(open(os.path.join("result",location,"supervised2_fold%d_beta%02d.log" % (f,beta)))) for f in range(5)]
    list_json_semisupervised = [json.load(open(os.path.join("result",location,"semisupervised_fold%d_beta%02d.log" % (f,beta)))) for f in range(5)]
    #list_json_discrim_test = [json.load(open(os.path.join("result",location,"descrim_testset_fold%d_beta%02d.log" % (f,beta)))) for f in range(5)]
    #list_json_discrim_train = [json.load(open(os.path.join("result",location,"descrim_trainset_fold%d_beta%02d.log" % (f,beta)))) for f in range(5)]
    
    accr_step1 = AveragedScores(list_json_supervised1,"validation/main/accr")
    accr_step2_full = AveragedScores(list_json_supervised2,"validation/main/accr")
    accr_step2_semi = AveragedScores(list_json_semisupervised,"validation/main/accr")
    #accr_discrim_train = AveragedScores(list_json_discrim_test,"d/accr")
    #accr_discrim_test = AveragedScores(list_json_discrim_train,"d/accr")
    kl = AveragedScores(list_json_semisupervised,"validation/main/generator/kl")
    reconst = AveragedScores(list_json_semisupervised,"validation/main/generator/reconstr")
    
    plt.figure()
    plt.title("beta=%d" % beta)
    plt.subplot(5,1,1)
    plt.title("Chord accuracy")
    plt.plot(np.concatenate((accr_step1,accr_step2_full)),"b")
    plt.plot(np.concatenate((accr_step1,accr_step2_semi)),"r")
    plt.subplot(5,1,2)
    plt.title("KL divergence")
    plt.plot(kl)
    #plt.subplot(5,1,3)
    #plt.title("Discriminator(train)")
    """
    plt.plot(accr_discrim_train)
    plt.subplot(5,1,4)
    plt.title("Discriminator(test)")
    plt.plot(accr_discrim_test)
    plt.subplot(5,1,5)
    plt.title("Reconstruction")
    plt.plot(reconst)
    """