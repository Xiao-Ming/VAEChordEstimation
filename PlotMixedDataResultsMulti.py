#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 13:07:21 2019

@author: wu
"""


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fman



list_size = [1,2,3]

list_score_majmin_markov_super = []
list_score_majmin_markov_semi = []
list_score_majmin_uniform_super = []
list_score_majmin_uniform_semi = []
list_score_majmin_super = []

list_score_triads_markov_super = []
list_score_triads_markov_semi = []
list_score_triads_uniform_super = []
list_score_triads_uniform_semi = []
list_score_triads_super = []



for s in list_size:
    d = np.load("scores/chromavae_noshiftdoublesuper_scores_mixed_size%d.model.npy" % s) * 100
    list_score_majmin_markov_super.append(d[2,0])
    list_score_majmin_markov_semi.append(d[0,0])
    list_score_triads_markov_super.append(d[3,0])
    list_score_triads_markov_semi.append(d[1,0])

for s in list_size:
    d = np.load("scores/chromavae_nomarkovdoublesuper_scores_mixed_size%d.model.npy" % s) * 100
    list_score_majmin_uniform_super.append(d[2,0])
    list_score_majmin_uniform_semi.append(d[0,0])
    list_score_triads_uniform_super.append(d[3,0])
    list_score_triads_uniform_semi.append(d[1,0])

for s in list_size:
    d = np.load("scores/chromavae_noshift_scores_mixed_size%d.model.npy" % s) * 100
    list_score_majmin_super.append(d[2,0])
    list_score_triads_super.append(d[3,0])
    #list_score_majmin_super.append(d[2,0])
    #list_score_triads_super.append(d[3,0])

d_full = np.load("scores_full_chromavae_full_doublesuper.npy") * 100
list_score_majmin_markov_super.append(d_full[0,0])
list_score_triads_markov_super.append(d_full[1,0])
list_score_majmin_markov_semi.append(d_full[0,0])
list_score_triads_markov_semi.append(d_full[1,0])

d_full = np.load("scores_full_chromavae_nomarkovdoublesuper.npy") * 100
list_score_majmin_uniform_super.append(d_full[0,0])
list_score_triads_uniform_super.append(d_full[1,0])
list_score_majmin_uniform_semi.append(d_full[0,0])
list_score_triads_uniform_semi.append(d_full[1,0])


list_fullsemi_size = [250,500,700]
list_score_majmin_fullsemi = []
list_score_triads_fullsemi = []
for s in list_fullsemi_size:
    d = np.load("scores_fullsemi_%d.npy" % s) * 100
    list_score_majmin_fullsemi.append(d[0,0])
    list_score_triads_fullsemi.append(d[1,0])

for s in list_fullsemi_size:
    d = np.load("scores_fullsemi_chromavae_nomarkov_%d.npy"%s) * 100
    list_score_majmin_uniform_semi.append(d[0,0])
    list_score_triads_uniform_semi.append(d[1,0])

list_score_majmin_markov_semi.extend(list_score_majmin_fullsemi)
list_score_triads_markov_semi.extend(list_score_triads_fullsemi)

list_score_majmin_super.append(81.46)
list_score_triads_super.append(76.90)

xticks  = ["244+732","488+488","732+244","976+0","976+250","976+500","976+700"]

#plt.subplot(2,1,2)


font=fman.FontProperties(family="STIXGeneral",size=15)


plt.plot(list_score_majmin_markov_semi,color="blue",marker="o",label=" ")
plt.plot(list_score_majmin_markov_super,color="dodgerblue",marker="o",label=" ")
plt.plot(list_score_majmin_uniform_semi,color="darkgreen",marker="o",label=" ")
plt.plot(list_score_majmin_uniform_super,color="lawngreen",marker="o",label=" ")
plt.plot(list_score_majmin_super,color="tomato",marker="o",label=" ")

plt.plot(list_score_triads_markov_semi,color="blue",marker="o",linestyle="dashed",label="VAE-MR-SSL")
plt.plot(list_score_triads_markov_super,color="dodgerblue",marker="o",linestyle="dashed",label="VAE-MR-SL")
plt.plot(list_score_triads_uniform_semi,color="darkgreen",marker="o",linestyle="dashed",label="VAE-UN-SSL")
plt.plot(list_score_triads_uniform_super,color="lawngreen",marker="o",linestyle="dashed",label="VAE-UN-SL")
plt.plot(list_score_triads_super,color="tomato",marker="o",linestyle="dashed",label="ACE-SL")

plt.plot([3,3],[74,84],linestyle="dotted",color="gray")
plt.text(3.1,83.1,"External unsupervised data",fontname="STIXGeneral",fontsize=20,color="gray")
plt.text(0,82.5,"Majmin criterion",fontname="STIXGeneral",fontsize=20,color="black")
plt.text(0,77.5,"Triads criterion",fontname="STIXGeneral",fontsize=20,color="black")
for i in range(7):
    plt.text(i,list_score_majmin_markov_semi[i]+0.1,"%.2f" % list_score_majmin_markov_semi[i],fontname="STIXGeneral",fontsize=15,color="blue",ha="center",va="bottom")
    plt.text(i,list_score_triads_markov_semi[i]+(0.1 if i!=3 else -0.3),"%.2f" % list_score_triads_markov_semi[i],fontname="STIXGeneral",fontsize=15,color="blue",ha="center",va="bottom")

for i in range(4):
    plt.text(i,list_score_majmin_super[i]-0.1,"%.2f" % list_score_majmin_super[i],fontname="STIXGeneral",fontsize=15,color="tomato",ha="center",va="top")
    plt.text(i,list_score_triads_super[i]-0.1,"%.2f" % list_score_triads_super[i],fontname="STIXGeneral",fontsize=15,color="tomato",ha="center",va="top")

plt.ylim((74.5,83.5))

#plt.ylim((0.65,0.85))
plt.xticks([0,1,2,3,4,5,6],xticks,fontsize=20,fontname="STIXGeneral")
plt.yticks(fontsize=20,fontname="STIXGeneral")
plt.xlabel("Data size (num. of annotated songs + num. of non-annotated songs)",fontsize=20,fontname="STIXGeneral")
plt.ylabel("Chord estimation accuracy (%)",fontsize=20,fontname="STIXGeneral")
plt.legend(loc="center right",prop=font,ncol=2)
plt.grid(color="gainsboro",axis="y")
#plt.text(-1,79,"(a)",fontname="STIXGeneral",fontsize=20)