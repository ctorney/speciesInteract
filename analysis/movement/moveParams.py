
import os
import csv
import math
import numpy as np
from datetime import datetime

from numpy import array, empty
from numpy.random import randint, rand
import numpy as np
import pandas as pd
from math import *
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

neighbours = np.load('neighbours.npy')
mvector = np.load('mvector.npy')
evector = np.load('evector.npy')
animals = np.load('animals.npy')
    


WILD=0
ZEB=1
#pick which species to focus on
ANIMAL=1
NEIGH=0
mvector = mvector[animals==ANIMAL]
evector = evector[animals==ANIMAL]
sin_ev = np.sin(evector)
cos_ev = np.cos(evector)
neighbours = neighbours[animals==ANIMAL]


def social_vector(il, de, ia):
        
    n_weights = ((neighbours[:,:,0]/il)*np.exp((1.0/de)*(1.0-np.power((neighbours[:,:,0]/il),de))))
    n_weights[(neighbours[:,:,1]<-ia)|(neighbours[:,:,1]>ia)]=0.0
    n_weights[neighbours[:,:,2]!=NEIGH] = 0.0
    
    xsv = np.sum(np.cos(neighbours[:,:,1])*n_weights,1)
    ysv = np.sum(np.sin(neighbours[:,:,1])*n_weights,1)
    
    lens = np.sqrt(xsv**2+ysv**2)
    print(np.max(lens))
    ysv[lens>1]=ysv[lens>1]/lens[lens>1]
    xsv[lens>1]=xsv[lens>1]/lens[lens>1]
    out = np.empty((len(mvector),2))

    out[:,0] = xsv
    out[:,1] = ysv
    
    return out


def desired_vector(al, be, sv):
    
    ally = be*sv[:,1]+(1.0-be)*(1.0-al)*sin_ev
    allx = be*sv[:,0]+(1.0-be)*(al*np.ones_like(mvector)+(1.0-al)*cos_ev)
    #dv = np.arctan2(np.sum(ypos,1), np.sum(xpos,1))
    dv = np.arctan2(ally,allx)
    
    return dv


def moves(social, dv):
    # this is the main function that calculates the log probability of all the moves based on the parameters that are passed in
    # and the assumed interaction function
    wcs = (1/(2*pi)) * (1-np.power(social,2))/(1+np.power(social,2)-2*social*np.cos((dv-mvector).transpose())) # weighted wrapped cauchy
    return np.sum(np.log(wcs))

il = 4.28
ia = 0.512
de = 0.0
rl = 0.88

be = 0.3642
al = 0.4134
    
social = 0.842
rholist = np.arange(0.1,50,1)
probs = np.zeros_like(rholist)

for i in range(len(rholist)):
    
    # this is the main function that calculates the log probability of all the moves based on the parameters that are passed in
    # and the assumed interaction function
    
    il   = rholist[i]
    de=50
    sv = social_vector(il,de,ia)
    dv = desired_vector(al,be,sv) # these are the headings (desired vector) without the autocorrelation; new heading = (eta)*(old heading) + (1-eta)*dv
    probs[i]=moves(social,dv)

plt.figure
plt.plot(rholist,probs)