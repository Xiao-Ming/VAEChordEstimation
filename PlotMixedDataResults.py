#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 17:32:55 2019

@author: wu
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fman

list_size = [50,100,200,300,500]

list_score_majmin_super = []
list_score_majmin_semi = []
list_score_triads_super = []
list_score_triads_semi = []

for s in list_size:
    d = np.load("scores_size%d.model.npy" % s)
    list_score_majmin_super.append(d[2,0])
    list_score_majmin_semi.append(d[0,0])
    list_score_triads_super.append(d[3,0])
    list_score_triads_semi.append(d[1,0])
    
d_full = np.load("scores_full.npy")
list_score_majmin_super.append(d_full[0,0])
list_score_triads_super.append(d_full[1,0])
list_score_majmin_semi.append(d_full[0,0])
list_score_triads_semi.append(d_full[1,0])
"""
list_fullsemi_size = [500]
list_score_majmin_fullsemi = []
list_score_triads_fullsemi = []
for s in list_fullsemi_size:
    d = np.load("scores_fullsemi_%d.npy" % s)
    list_score_majmin_fullsemi.append(d[0,0])
    list_score_triads_fullsemi.append(d[1,0])
"""
"""
list_score_majmin_semi.extend(list_score_majmin_fullsemi)
list_score_triads_semi.extend(list_score_triads_fullsemi)
"""

xticks  = ["50+922","100+872","200+772","300+672","500+472","972+0"]

font=fman.FontProperties(family="IPAexGothic",size=20)


plt.plot(list_score_majmin_semi,color="dodgerblue",marker="o",label="majmin(半教師あり)")
plt.plot(list_score_majmin_super,color="tomato",marker="o",label="majmin(教師あり)")

plt.plot(list_score_triads_semi,color="dodgerblue",marker="o",linestyle="dashed",label="triads(半教師あり)")
plt.plot(list_score_triads_super,color="tomato",marker="o",linestyle="dashed",label="triads(教師あり)")

#plt.ylim((0.65,0.85))
plt.xticks([0,1,2,3,4,5],xticks,fontsize=20,fontname="STIXGeneral")
plt.yticks(fontsize=20,fontname="STIXGeneral")
plt.xlabel("学習データ曲数（アノテーションあり曲数＋アノテーションなし曲数）",fontsize=20,fontname="IPAexGothic")
plt.ylabel("平均認識精度",fontsize=20,fontname="IPAexGothic")
plt.legend(loc="lower right",prop=font)
plt.grid(color="gainsboro",axis="y")
plt.tight_layout()