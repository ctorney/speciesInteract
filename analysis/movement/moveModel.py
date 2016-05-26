
import os
import csv
import math

from datetime import datetime
from pymc import *
from numpy import array, empty
from numpy.random import randint, rand

import pandas as pd
from pymc.Matplot import plot as mcplot
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['decay_exponent','interaction_length','ignore_length','interaction_angle','rho_s','rho_m','rho_e','alpha','beta','mvector','social_vector','desired_vector']


interaction_length = Uniform('interaction_length', lower=0.5, upper=20.0)
ignore_length = Uniform('ignore_length', lower=0.5, upper=20.0)
#ignore_length = DiscreteUniform('ignore_length', lower=1, upper=3)#,value=1.0)
decay_exponent = Uniform('decay_exponent', lower=0.5, upper=10.0)#,value=1.0)
interaction_angle = Uniform('interaction_angle', lower=0, upper=pi)
rho_s = Uniform('rho_s',lower=0, upper=1)
rho_m = Uniform('rho_m',lower=0, upper=1)
rho_e = Uniform('rho_e',lower=0, upper=1)
alpha = Uniform('alpha',lower=0, upper=1)
beta = Uniform('beta',lower=0, upper=1)

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

@deterministic(plot=False)
def social_vector(il=interaction_length, ig=ignore_length, de=decay_exponent, ia=interaction_angle):
        
    #n_weights = ((neighbours[:,:,0]/il)*np.exp((1.0/de)*(1.0-(neighbours[:,:,0]/il)**de)))
    n_weights = np.tanh(neighbours[:,:,0]*ig)*(0.5+0.5*np.tanh(de*(il-neighbours[:,:,0])))
    

    n_weights[(neighbours[:,:,1]<-ia)|(neighbours[:,:,1]>ia)]=0.0
    n_weights[neighbours[:,:,2]!=NEIGH] = 0.0
    
    xsv = np.sum(np.cos(neighbours[:,:,1])*n_weights,1)
    ysv = np.sum(np.sin(neighbours[:,:,1])*n_weights,1)
    
    lens = np.sqrt(xsv**2+ysv**2)
    ysv[lens>1]=ysv[lens>1]/lens[lens>1]
    xsv[lens>1]=xsv[lens>1]/lens[lens>1]
    out = np.empty((len(mvector),2))

    out[:,0] = xsv
    out[:,1] = ysv
    
    return out

    

@stochastic(observed=True)
def moves(social=rho_s, rm=rho_m,re=rho_e,al=alpha, be=beta, sv=social_vector, value=mvector):
    # this is the main function that calculates the log probability of all the moves based on the parameters that are passed in
    # and the assumed interaction function
    svv = np.arctan2(sv[:,1],sv[:,0])
    lens = np.sqrt(sv[:,1]**2+sv[:,0]**2)
    als = al*lens
    socials=lens*social
    wcs = (1/(2*pi)) * (1-np.power(socials,2))/(1+np.power(socials,2)-2*socials*np.cos((svv-mvector).transpose())) # weighted wrapped cauchy
    wce = (1/(2*pi)) * (1-np.power(re,2))/(1+np.power(re,2)-2*re*np.cos((evector-mvector).transpose())) # weighted wrapped cauchy
    wcm = (1/(2*pi)) * (1-np.power(rm,2))/(1+np.power(rm,2)-2*rm*np.cos((-mvector).transpose())) # weighted wrapped cauchy
    wcc = als*wcs + (1.0-als)*(be*wce+(1.0-be)*wcm)
    return np.sum(np.log(wcc))



