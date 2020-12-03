#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:47:00 2019

@author: wu
"""


import numpy as np
from librosa.util import find_files
import os.path

q_map = {"":"maj",
         "m":"min",
         "M7":"maj7",
         "7":"7",
         "m7":"min7",
         "6":"maj6",
         "dim":"dim",
         "aug":"aug",
         "dim7":"dim7"
         }
note_map = {"C":0,
            "Cb":11,
            "C#":1,
            "Db":1,
            "D":2,
            "D#":3,
            "Eb":3,
            "E":4,
            "E#":5,
            "Fb":4,
            "F":5,
            "F#":6,
            "Gb":6,
            "G":7,
            "G#":8,
            "Ab":8,
            "A":9,
            "A#":10,
            "Bb":10,
            "B":11,
            "B#":0}

bass_note_symbol = ["0","b2","2","b3","3","4","b5","5","b6","6","b7","7"]

def convert2lab(symbol):
    if symbol=="N":
        return "N"
    sp = symbol.split("/")
    root = sp[0][0]
    idx_qual = 1
    if len(sp[0]) > 1:
        if sp[0][1]=="#" or sp[0][1]=="b":
            root += sp[0][1]
            idx_qual = 2
    qual = ""
    if len(sp[0])>idx_qual:
        qual = sp[0][idx_qual:]
    qual_map = q_map[qual]
    if len(sp)==1:
        return "%s:%s" % (root,qual_map)
    else:
        root_note = note_map[root]
        bass_note = note_map[sp[1]]
        bass_symb = bass_note_symbol[(bass_note-root_note)%12]
        return "%s:%s/%s" % (root,qual_map,bass_symb)




flist = find_files("RWC-MDB-P-2001-M",ext="txt")

for path in flist:
    print(path)
    f = open(path,"r")
    fname = path.split("/")[-1]
    line = f.readline()
    line = f.readline()
    lab = open(os.path.join("RWC-MDB-P-2001-M/labs",fname+".lab"),"w")
    while len(line.split()) > 4:
        sp = line.split()
        st = sp[0]
        ed = sp[1]
        symbol = sp[3]
        chord_lab = convert2lab(symbol)
        #print(chord_lab)
        lab.write("%s %s %s\n" % (st,ed,chord_lab))
        line = f.readline()
    lab.close()