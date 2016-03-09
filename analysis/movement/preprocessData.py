
import os
import csv
import math
import numpy as np
from datetime import datetime

from numpy import array, empty
from numpy.random import randint, rand
import numpy as np
import pandas as pd

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from math import *


HD = os.getenv('HOME')
DD = '/media/ctorney/SAMSUNG/'

DATADIR = DD + '/data/wildebeest/lacey-field-2015/'
CLIPDIR = DD + '/data/wildebeest/lacey-field-2015/wildzeb/'
CLIPLIST = HD + '/workspace/speciesInteract/clipListReduced.csv'

OUTDIR = HD + '/Dropbox/Wildebeest_collaboration/Data/w_z/'

df = pd.read_csv(CLIPLIST)
allDF = pd.DataFrame()

# wildebeest = 0
# zebra = 1

# data files store the positions at each frame - with a frame = 0.1 seconds


for index, row in df.iterrows():
    noext, ext = os.path.splitext(row.clipname)   
    posfilename = OUTDIR +  '/TRACKS_' + noext + '.csv'
    posDF = pd.read_csv(posfilename) 
    posDF['clip']=index
    posDF = posDF[posDF['frame']%20==0]
    dt = 20 # 20 frames is 2 seconds
    posDF['dtheta']=np.NaN
    posDF['env_heading']=np.NaN
    for index, row in posDF.iterrows():
        thisFrame =  row['frame']
        thisID = row['id']
       
        thisTheta = row['heading']
        # calculate the change in heading from this point to the next
        nextTime = posDF[(np.abs(posDF['frame']-(thisFrame+dt))<1e-6)&(posDF['id']==thisID)]
        if len(nextTime)==1:
            posDF.ix[index,'dtheta'] = nextTime.iloc[0]['heading'] -  thisTheta
            thisX = nextTime.iloc[0]['x']
            thisY = nextTime.iloc[0]['y']
            # calculate the average heading all the other caribou were taking at this point
            excThis = posDF[posDF['id']!=0]
            xp = excThis['x'].values
            yp = excThis['y'].values
            xdirs = excThis['dx'].values
            ydirs = excThis['dy'].values
            kappa = 1.0*1.0 ##check this 
            dists = (((xp - thisX)**2 + (yp - thisY)**2))
            weights = np.exp(-dists/kappa)
            xav = np.sum(weights*xdirs)/np.sum(weights)
            yav = np.sum(weights*ydirs)/np.sum(weights)
            posDF.ix[index,'env_heading']  = math.atan2(yav,xav)-  thisTheta
            

    
    allDF = allDF.append(posDF,ignore_index=True)
    
    
allDF = allDF[np.isfinite(allDF['dtheta'])]
allDF = allDF.reset_index(drop=True)
dsize = len(allDF)
maxN=0
for index, row in allDF.iterrows():
    thisFrame =  row['frame']
    thisID = row['id']
    thisClip = row['clip']
    window = allDF[(allDF.frame==thisFrame)&(allDF['clip']==thisClip)&(allDF['id']!=thisID)]
    if len(window)>maxN:
        maxN=len(window)#

neighbours = np.zeros((dsize,maxN,3)).astype(np.float32) # dist, angle

for index, row in allDF.iterrows():
    thisFrame =  row['frame']
    thisID = row['id']
    thisClip = row['clip']
    thisX = row['x']
    thisY = row['y']
    thisAngle = row['heading']
    window = allDF[(allDF.frame==thisFrame)&(allDF['clip']==thisClip)&(allDF['id']!=thisID)]
    ncount = 0

    for i2, w in window.iterrows():
        xj = w.x
        yj = w.y
        neighbours[index,ncount,0] = ((((thisX-xj)**2+(thisY-yj)**2))**0.5) 
        dx = xj - thisX
        dy = yj - thisY
        angle = math.atan2(dy,dx)
        angle = angle - thisAngle
        neighbours[index,ncount,1] = math.atan2(math.sin(angle), math.cos(angle))
        if w.animal=='z':
            neighbours[index,ncount,2] = 1.0
        ncount+=1



mvector = allDF['dtheta'].values
mvector[mvector<-pi]=mvector[mvector<-pi]+2*pi
mvector[mvector>pi]=mvector[mvector>pi]-2*pi

evector = allDF['env_heading'].values
evector[evector<-pi]=evector[evector<-pi]+2*pi
evector[evector>pi]=evector[evector>pi]-2*pi

animals = np.zeros_like(mvector)
animals[allDF['animal'].values=='z']=1.0

np.save('neighbours.npy', neighbours)
np.save('mvector.npy', mvector)
np.save('evector.npy', evector)
np.save('animals.npy',animals)

