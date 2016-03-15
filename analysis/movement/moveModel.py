
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

__all__ = ['decay_exponent','interaction_length','interaction_angle','rho','alpha','beta','mvector','social_vector','desired_vector']


interaction_length = Uniform('interaction_length', lower=0.5, upper=20.0)
#ignore_length = DiscreteUniform('ignore_length', lower=1, upper=3)#,value=1.0)
decay_exponent = Uniform('decay_exponent', lower=0.5, upper=10.0)#,value=1.0)
interaction_angle = Uniform('interaction_angle', lower=0, upper=pi)
rho = Uniform('rho',lower=0, upper=1)
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
def social_vector(il=interaction_length, de=decay_exponent, ia=interaction_angle):
        
    n_weights = ((neighbours[:,:,0]/il)*np.exp((1.0/de)*(1.0-(neighbours[:,:,0]/il)**de)))
    n_weights[(neighbours[:,:,1]<-ia)|(neighbours[:,:,1]>ia)]=0.0
    n_weights[neighbours[:,:,2]!=NEIGH] = 0.0
    
    xsv = np.sum(np.cos(neighbours[:,:,1])*n_weights,1)
    ysv = np.sum(np.sin(neighbours[:,:,1])*n_weights,1)
    
    lens = (xsv**2+ysv**2)
    ysv[lens>1]=ysv[lens>1]/lens[lens>1]
    xsv[lens>1]=xsv[lens>1]/lens[lens>1]
    out = np.empty((len(mvector),2))

    out[:,0] = xsv
    out[:,1] = ysv
    
    return out

@deterministic(plot=False)
def desired_vector(al=alpha, be=beta, sv=social_vector):
    
    ally = be*sv[:,1]+(1.0-be)*(1.0-al)*sin_ev
    allx = be*sv[:,0]+(1.0-be)*(al*np.ones_like(mvector)+(1.0-al)*cos_ev)
    #dv = np.arctan2(np.sum(ypos,1), np.sum(xpos,1))
    dv = np.arctan2(ally,allx)
    
    return dv

@stochastic(observed=True)
def moves(social=rho, dv=desired_vector, value=mvector):
    # this is the main function that calculates the log probability of all the moves based on the parameters that are passed in
    # and the assumed interaction function
    wcs = (1/(2*pi)) * (1-np.power(social,2))/(1+np.power(social,2)-2*social*np.cos((dv-mvector).transpose())) # weighted wrapped cauchy
    return np.sum(np.log(wcs))



