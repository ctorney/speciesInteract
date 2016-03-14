
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


neighbours = np.load('neighbours1.npy')
mvector = np.load('mvector1.npy')
evector = np.load('evector1.npy')
animals = np.load('animals1.npy')
    


WILD=0
ZEB=1
#pick which species to focus on
ANIMAL=WILD

mvector = mvector[animals==ANIMAL]
evector = evector[animals==ANIMAL]
neighbours = neighbours[animals==ANIMAL]

il=18.5
ia = 0.2912
al = 0.0
rl = 0.88
discount = 0.5
ig=49.54


rholist = np.arange(0.0001,100,0.01)
probs = np.zeros_like(rholist)

for i in range(len(rholist)):

    # this is the main function that calculates the log probability of all the moves based on the parameters that are passed in
    # and the assumed interaction function
    be = 0.3642
    alpha = 0.4134
    
    social = 0.9159# rholist[i]#0.95
    ds=  1*rholist[i]
    dv = np.zeros_like(mvector) # these are the headings (desired vector) without the autocorrelation; new heading = (eta)*(old heading) + (1-eta)*dv
    n_weights = np.tanh(neighbours[:,:,0]*ig)*(0.5+0.5*np.tanh(ds*(il-neighbours[:,:,0])))
    n_weights[(neighbours[:,:,0]==0)|(neighbours[:,:,1]<-ia)|(neighbours[:,:,1]>ia)]=0.0
    n_weights[neighbours[:,:,2]!=ANIMAL] = 0.0#ds*n_weights[neighbours[:,:,2]!=ANIMAL]
    
    #n_weights = np.exp(-np.abs(neighbours[:,:,1])/ia)*np.exp(-neighbours[:,:,0]/il)*np.tanh(neighbours[:,:,0]/ig)
    #n_weights[(neighbours[:,:,0]==0)]=0.0
    
    xpos = np.cos(neighbours[:,:,1])*n_weights
    ypos = np.sin(neighbours[:,:,1])*n_weights
    
    #sv = np.arctan2(np.sum(ypos,1), np.sum(xpos,1))
#    ysv = np.sum(ypos,1)/np.sum(n_weights,1)
 #   xsv = np.sum(xpos,1)/np.sum(n_weights,1)
    ysv = np.sum(ypos,1)#/np.sum(n_weights,1)
    xsv = np.sum(xpos,1)#/np.sum(n_weights,1)
    lens = np.sqrt(xsv**2+ysv**2)
    ysv[lens>1]=ysv[lens>1]/lens[lens>1]
    xsv[lens>1]=xsv[lens>1]/lens[lens>1]
    
    #print(xsv[6],ysv[6])
    #ysv = np.sin(sv)
    #xsv = np.cos(sv)
    xsv[np.sum(n_weights,1)==0] = 0.0
    ysv[np.sum(n_weights,1)==0] = 0.0

    ally = be*ysv+(1.0-be)*(1.0-al)*np.sin(evector)
    allx = be*xsv+(1.0-be)*(al*np.ones_like(mvector)+(1.0-al)*np.cos(evector))
    #dv = np.arctan2(np.sum(ypos,1), np.sum(xpos,1))
    dv = np.arctan2(ally,allx)
   #print(dv[6])    
    wcs = (1/(2*pi)) * (1-np.power(social,2))/(1+np.power(social,2)-2*social*np.cos((dv-mvector).transpose())) # weighted wrapped cauchy    probs[i]= np.sum(np.log(wcs))
    probs[i]= np.sum(np.log(wcs))
    #print( np.sum(np.log(wcs)))
#    ds=  1.17#10*rholist[i]
#    dv = np.zeros_like(mvector) # these are the headings (desired vector) without the autocorrelation; new heading = (eta)*(old heading) + (1-eta)*dv
#    n_weights = np.tanh(neighbours[:,:,0]*ig)*(0.5+0.5*np.tanh(ds*(il-neighbours[:,:,0])))
#    n_weights[(neighbours[:,:,0]==0)|(neighbours[:,:,1]<-ia)|(neighbours[:,:,1]>ia)]=0.0
#    n_weights[neighbours[:,:,2]!=ANIMAL] = 0.0#ds*n_weights[neighbours[:,:,2]!=ANIMAL]
#    
#    #n_weights = np.exp(-np.abs(neighbours[:,:,1])/ia)*np.exp(-neighbours[:,:,0]/il)*np.tanh(neighbours[:,:,0]/ig)
#    #n_weights[(neighbours[:,:,0]==0)]=0.0
#    
#    xpos = np.cos(neighbours[:,:,1])*n_weights
#    ypos = np.sin(neighbours[:,:,1])*n_weights
#    
#    #sv = np.arctan2(np.sum(ypos,1), np.sum(xpos,1))
#    ysv = np.sum(ypos,1)#/np.sum(n_weights,1)
#    xsv = np.sum(xpos,1)#/np.sum(n_weights,1)
#    lens = np.sqrt(xsv**2+ysv**2)
#    ysv[lens>1]=ysv[lens>1]/lens[lens>1]
#    xsv[lens>1]=xsv[lens>1]/lens[lens>1]
#    print(xsv[6],ysv[6])
#    #ysv = np.sin(sv)
#    #xsv = np.cos(sv)
#    xsv[np.sum(n_weights,1)==0] = 0.0
#    ysv[np.sum(n_weights,1)==0] = 0.0
#    
#    ally = be*ysv+(1.0-be)*(1.0-al)*np.sin(evector)
#    allx = be*xsv+(1.0-be)*(al*np.ones_like(mvector)+(1.0-al)*np.cos(evector))
#    #dv = np.arctan2(np.sum(ypos,1), np.sum(xpos,1))
#    dv = np.arctan2(ally,allx)
#    print(dv[6])
#    wcs = (1/(2*pi)) * (1-np.power(social,2))/(1+np.power(social,2)-2*social*np.cos((dv[6]-mvector[6]).transpose())) # weighted wrapped cauchy    probs[i]= np.sum(np.log(wcs))
#    probs[i]= np.sum(np.log(wcs))
#    print( np.sum(np.log(wcs)))
#
#    break

plt.figure
plt.plot(rholist,probs)