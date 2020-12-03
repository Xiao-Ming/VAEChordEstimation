#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 12:30:21 2020

@author: wu
"""
import matplotlib.pyplot as plt
import seaborn
import numpy as np
import pandas
import matplotlib.font_manager as fman


dat1 = np.load("scores_full_chromavae_full_doublesuper_all.npz")

dat1_03 = np.load("scores_full_chromavae_full_doublesuper_trans03_all.npz")
dat1_05 = np.load("scores_full_chromavae_full_doublesuper_trans05_all.npz")
dat1_07 = np.load("scores_full_chromavae_full_doublesuper_trans07_all.npz")

dat2 = np.load("scores_full_chromavae_nomarkovdoublesuper_all.npz")

dat3 = np.load("scores_full_chromavae_vanilla_all.npz")

dat4 = np.load("scores_fullsemi_chromavae_fullsemi_700_all.npz")

dat4_03 = np.load("scores_fullsemi_chromavae_fullsemi_03_700_all.npz")
dat4_05 = np.load("scores_fullsemi_chromavae_fullsemi_05_700_all.npz")
dat4_07 = np.load("scores_fullsemi_chromavae_fullsemi_07_700_all.npz")

dat5 = np.load("scores_fullsemi_chromavae_nomarkov_700_all.npz")

dat_gt = np.load("gtruth.npz")

transrate1 = dat1["transrate"]
transrate1_03 = dat1_03["transrate"]
transrate1_05 = dat1_05["transrate"]
transrate1_07 = dat1_07["transrate"]
transrate2 = dat2["transrate"]
transrate3 = dat3["transrate"]
transrate4 = dat4["transrate"]
transrate4_03 = dat4_03["transrate"]
transrate4_05 = dat4_05["transrate"]
transrate4_07 = dat4_07["transrate"]
transrate5 = dat5["transrate"]
transrate_gt = dat_gt["transrate"]

triad1 = dat1["majmin"] * 100
triad1_03 = dat1_03["majmin"] * 100
triad1_05 = dat1_05["majmin"] * 100
triad1_07 = dat1_07["majmin"] * 100
triad2 = dat2["majmin"] * 100
triad3 = dat3["majmin"] * 100
triad4 = dat4["majmin"] * 100
triad4_03 = dat4_03["majmin"] * 100
triad4_05 = dat4_05["majmin"] * 100
triad4_07 = dat4_07["majmin"] * 100
triad5 = dat5["majmin"] * 100

#transrate1 = np.random.rand(1000)
#transrate2 = np.random.rand(1000)
#transrate3 = np.random.rand(1000)
#transrate4 = np.random.rand(1000)
#transrate5 = np.random.rand(1000)
#transrate3 = dat3["transrate"]

dframe1 = pandas.DataFrame({"rate":transrate1,
                            "triads":triad1,
                           "method":"VAE-MR-SSL (976+0)",
                           "dataset":"976+0",
                           "transrate":"0.9"})

dframe1_03 = pandas.DataFrame({"rate":transrate1_03,
                            "triads":triad1_03,
                           "method":"VAE-MR-SSL (976+0)",
                           "dataset":"976+0",
                           "transrate":"0.3"})
dframe1_05 = pandas.DataFrame({"rate":transrate1_05,
                            "triads":triad1_05,
                           "method":"VAE-MR-SSL (976+0)",
                           "dataset":"976+0",
                           "transrate":"0.5"})
dframe1_07 = pandas.DataFrame({"rate":transrate1_07,
                            "triads":triad1_07,
                           "method":"VAE-MR-SSL (976+0)",
                           "dataset":"976+0",
                           "transrate":"0.7"})

dframe2 = pandas.DataFrame({"rate":transrate2,
                            "triads":triad2,
                            "method":"VAE-MR-SSL (976+0)",
                            "dataset":"976+0",
                            "transrate":"1/K"})
dframe3 = pandas.DataFrame({"rate":transrate3,
                            "triads":triad3,
                            "method":"ACE-SL",
                            "dataset":"976+0",
                            "transrate":"1/K"})

dframe4 = pandas.DataFrame({"rate":transrate4,
                            "triads":triad4,
                           "method":"VAE-MR-SSL (976+700)",
                           "dataset":"976+700",
                           "transrate":"0.9"})

dframe4_03 = pandas.DataFrame({"rate":transrate4_03,
                            "triads":triad4_03,
                           "method":"VAE-MR-SSL (976+700)",
                           "dataset":"976+700",
                           "transrate":"0.3"})
dframe4_05 = pandas.DataFrame({"rate":transrate4_05,
                            "triads":triad4_05,
                           "method":"VAE-MR-SSL (976+700)",
                           "dataset":"976+700",
                           "transrate":"0.5"})
dframe4_07 = pandas.DataFrame({"rate":transrate4_07,
                            "triads":triad4_07,
                           "method":"VAE-MR-SSL (976+700)",
                           "dataset":"976+700",
                           "transrate":"0.7"})

dframe5 = pandas.DataFrame({"rate":transrate5,
                            "triads":triad5,
                            "method":"VAE-MR-SSL (976+700)",
                            "dataset":"976+700",
                            "transrate":"1/K"})
dframe_gt = pandas.DataFrame({"rate":transrate_gt,
                              "triads":0,
                              "method":"Ground-truth",
                              "dataset": "Ground-truth",
                              "transrate":""})


dframe_cat_gt = pandas.concat([dframe1,dframe1_03,dframe1_05,dframe1_07,
                               dframe2,dframe3,dframe4,
                               dframe4_03,dframe4_05,dframe4_07,dframe5,
                               dframe_gt])
dframe = pandas.concat([dframe1,dframe1_03,dframe1_05,dframe1_07,
                               dframe2,dframe3,dframe4,
                               dframe4_03,dframe4_05,dframe4_07,dframe5])

font=fman.FontProperties(family="STIXGeneral",size=15)

seaborn.set(style="whitegrid")

plt.subplot(2,1,2)

seaborn.violinplot(x="transrate",y="rate",hue="method",
                order=["1/K","0.3","0.5","0.7","0.9",""],
                hue_order=["ACE-SL",
                           "VAE-MR-SSL (976+0)",
                           "VAE-MR-SSL (976+700)",
                           "Ground-truth",
                           ],
                palette=["lightsalmon","lightskyblue","dodgerblue","gainsboro"],
                data=dframe_cat_gt,showfliers=False,
                #width=0.95,
                scale="count",
                bw="silverman",
                cut=0,
                linewidth=.8
                #orient="h",
                )
#seaborn.boxplot(x="dataset",y="rate",hue="method",data=dframe_gt,showfliers=False)
plt.title("(b)",fontname="STIXGeneral",fontsize=20)
#plt.yticks([0, .8, 1.8, 2.8, 3.8, 4.8],fontname="STIXGeneral",fontsize=15)
plt.xticks(fontname="STIXGeneral",fontsize=15)
plt.yticks(fontname="STIXGeneral",fontsize=15)
plt.xlabel("Self-transition probability of Φ",fontname="STIXGeneral",fontsize=15)
plt.ylabel("Num. of frames per chord",fontname="STIXGeneral",fontsize=15)
#plt.ylim(-.2,5)
plt.ylim(0,66)
plt.xlim(-0.5,5.5)
#plt.legend(prop=font,loc="center left",bbox_to_anchor=(1.05, 0.5))
plt.legend(prop=font,loc="upper right")


plt.subplot(2,1,1)
seaborn.violinplot(x="transrate",y="triads",hue="method",data=dframe,showfliers=False,
                order=["1/K","0.3","0.5","0.7","0.9"],
                hue_order=["ACE-SL",
                            "VAE-MR-SSL (976+0)",
                           "VAE-MR-SSL (976+700)",
                           None
                           ],  
                #width=0.95,
                palette=["lightsalmon","lightskyblue","dodgerblue"],
                #orient="h"
                scale="count",
                bw="silverman",
                cut=0,
                linewidth=.8,
                )
plt.title("(a)",fontname="STIXGeneral",fontsize=20)
plt.ylim(1,142)
#plt.yticks([.0, 0.8, 1.8, 2.8, 3.8],fontname="STIXGeneral",fontsize=15)
plt.xticks(fontname="STIXGeneral",fontsize=15)
plt.yticks([20,40,60,80,100],fontname="STIXGeneral",fontsize=15)
plt.xlabel("Self-transition probability of Φ",fontname="STIXGeneral",fontsize=15)
plt.ylabel("Chord classification accuracy (%)",fontname="STIXGeneral",fontsize=15)
#plt.legend(prop=font,loc="center left", bbox_to_anchor=(1.05,0.5))

plt.xlim(-0.5,5.5)
plt.legend(prop=font,loc="upper right")

#plt.tight_layout()

