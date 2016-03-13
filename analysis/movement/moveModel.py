
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

__all__ = ['ignore_length','interaction_length','interaction_angle','rho','alpha','beta','discount','mvector']


interaction_length = Uniform('interaction_length', lower=0.5, upper=20.0)
#ignore_length = DiscreteUniform('ignore_length', lower=1, upper=3)#,value=1.0)
ignore_length = Uniform('ignore_length', lower=0.5, upper=10.0)#,value=1.0)
interaction_angle = Uniform('interaction_angle', lower=0, upper=pi)
rho = Uniform('rho',lower=0, upper=1)
alpha = Uniform('alpha',lower=0, upper=1)
beta = Uniform('beta',lower=0, upper=1)
#cross-species discount of social information
discount = Uniform('discount',lower=0.5, upper=10.0)

# rho is tanh(a dx) * exp(-b dx)
# the inflexion point is located at (1/2a)ln(2a/b + sqrt((2a/b)^2+1)

neighbours = np.load('neighbours1.npy')
mvector = np.load('mvector1.npy')
evector = np.load('evector1.npy')
animals = np.load('animals1.npy')
    


WILD=0
ZEB=1
#pick which species to focus on
ANIMAL=0
NEIGH=0
mvector = mvector[animals==ANIMAL]
evector = evector[animals==ANIMAL]
sin_ev = np.sin(evector)
cos_ev = np.cos(evector)
neighbours = neighbours[animals==ANIMAL]

@stochastic(observed=True)
def moves(il=interaction_length,ig=ignore_length, ia=interaction_angle, social=rho, al=alpha, be=beta, ds=discount, value=mvector):
    # this is the main function that calculates the log probability of all the moves based on the parameters that are passed in
    # and the assumed interaction function
    
    dv = np.zeros_like(mvector) # these are the headings (desired vector) without the autocorrelation; new heading = (eta)*(old heading) + (1-eta)*dv
    #lambdas[np.abs(mvector)>pi]=pi
    #dv[np.abs(mvector)>(1-al)*pi]=pi
    #dv[np.abs(mvector)<(1-al)*pi]=mvector[np.abs(mvector)<(1-al)*pi]/(1-al)
 #   dv =np.arctan2( (np.sin(mvector)-(1.0-beta)*(1.0-alpha)*np.sin(evector)), (np.cos(mvector)-(1.0-beta)*(1.0-alpha)*np.cos(evector)))
   # first calculate all the rhos
    n_weights = np.zeros_like(neighbours[:,:,0])
    #nonz=np.nonzero(neighbours[:,:,0])
#    n_weights = np.exp(-neighbours[:,:,0]/il)*np.tanh(neighbours[:,:,0]/ig)
    
#    powdist = np.power(neighbours[nonz][:,0]/il,ig)
    #powdist = (neighbours[nonz][:,0]/il)
    #n_weights = ((neighbours[:,:,0]/il)*np.exp((1.0/ig)*(1.0-(neighbours[:,:,0]/il)**ig)))#**ig
    #n_weights[nonz] = ((neighbours[nonz][:,0]/il)*np.exp((1.0-powdist)))#**ig
    n_weights = np.tanh(neighbours[:,:,0]*ig)*(0.5+0.5*np.tanh(ds*(il-neighbours[:,:,0])))
    n_weights[(neighbours[:,:,1]<-ia)|(neighbours[:,:,1]>ia)]=0.0
    #n_weights[(neighbours[:,:,0]==0)|(neighbours[:,:,1]<-ia)|(neighbours[:,:,1]>ia)]=0.0
    n_weights[neighbours[:,:,2]!=NEIGH] = 0.0#ds*n_weights[neighbours[:,:,2]!=ANIMAL]
    
    #n_weights = np.exp(-np.abs(neighbours[:,:,1])/ia)*np.exp(-neighbours[:,:,0]/il)*np.tanh(neighbours[:,:,0]/ig)
    #n_weights[(neighbours[:,:,0]==0)]=0.0
    
    xsv = np.sum(np.cos(neighbours[:,:,1])*n_weights,1)
    ysv = np.sum(np.sin(neighbours[:,:,1])*n_weights,1)
    
    #sv = np.arctan2(np.sum(ypos,1), np.sum(xpos,1))
#    ysv = np.sum(ypos,1)/np.sum(n_weights,1)
#    xsv = np.sum(xpos,1)/np.sum(n_weights,1)
    #ysv = np.sum(ypos,1)#/np.sum(n_weights,1)
    #xsv = np.sum(xpos,1)#/np.sum(n_weights,1)
    lens = (xsv**2+ysv**2)
    ysv[lens>1]=ysv[lens>1]/lens[lens>1]
    xsv[lens>1]=xsv[lens>1]/lens[lens>1]
    
    #ysv = np.sin(sv)
    #xsv = np.cos(sv)
 #   xsv[np.sum(n_weights,1)==0] = 0.0
  #  ysv[np.sum(n_weights,1)==0] = 0.0
    ally = be*ysv+(1.0-be)*(1.0-al)*sin_ev
    allx = be*xsv+(1.0-be)*(al*np.ones_like(mvector)+(1.0-al)*cos_ev)
    #dv = np.arctan2(np.sum(ypos,1), np.sum(xpos,1))
    dv = np.arctan2(ally,allx)
    # this isn't necessary here but if there are larger groups each neighbour has to be included and the total normalized
    #nc = np.sum(np.abs(rhos),1) # normalizing constant

    #wwc = ((rhos))*(1/(2*pi)) * (1-np.power(rhos,2))/(1+np.power(rhos,2)-2*rhos*np.cos((dv-neighbours[:,:,1].transpose()).transpose())) # weighted wrapped cauchy
    wcs = (1/(2*pi)) * (1-np.power(social,2))/(1+np.power(social,2)-2*social*np.cos((dv-mvector).transpose())) # weighted wrapped cauchy
 #   wcs[(np.sum(ypos,1)==0)&(np.sum(xpos,1)==0)] = 1/(2*pi)
 #   wce = (1/(2*pi)) * (1-np.power(be,2))/(1+np.power(be,2)-2*be*np.cos((dv-evector).transpose())) # weighted wrapped cauchy
    # sum along the individual axis to get the total compound cauchy
    #wwc = np.sum(wwc,1)/nc
    #wwc[np.isnan(wwc)]=1/(2*pi)
    
  #  wwc = (social/(be+social))*wcs + (be/(be+social))*wce
#    # next we want to split desired vector into social and environmental vector
#    sv = np.zeros_like(mvector)
#    # desired vector  = b*env_vector + (1-b)*social_vector
#    sv = (dv - be*evector)/(1-be)
#    sv[np.isnan(sv)]=pi
#    sv[np.abs(sv)>pi]=pi
#    
#    # first calculate all the rhos
#    rhos = np.zeros_like(neighbours[:,:,0])
#    rhos[(neighbours[:,:,0]>0)&(neighbours[:,:,0]<il)&(neighbours[:,:,1]>-ia)&(neighbours[:,:,1]<ia)]=social
#    
#    # this isn't necessary here but if there are larger groups each neighbour has to be included and the total normalized
#    nc = np.sum(np.abs(rhos),1) # normalizing constant
#
#    wwc = (np.abs(rhos))*(1/(2*pi)) * (1-np.power(rhos,2))/(1+np.power(rhos,2)-2*rhos*np.cos((sv-neighbours[:,:,1].transpose()).transpose())) # weighted wrapped cauchy
#    # sum along the individual axis to get the total compound cauchy
#    wwc = np.sum(wwc,1)/nc
#    wwc[np.isnan(wwc)]=1/(2*pi)
    return np.sum(np.log(wcs))
