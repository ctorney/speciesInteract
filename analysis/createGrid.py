import numpy as np
import pandas as pd
import os, re
import math
import time
from scipy import interpolate
from scipy import ndimage
import matplotlib.pyplot as plt
HD = os.getenv('HOME')
DD = '/media/ctorney/SAMSUNG/'

DATADIR = DD + '/data/wildebeest/lacey-field-2015/'
CLIPDIR = DD + '/data/wildebeest/lacey-field-2015/wildzeb/'
CLIPLIST = HD + '/workspace/speciesInteract/clipListReduced.csv'

OUTDIR = HD + '/Dropbox/Wildebeest_collaboration/Data/w_z/'

df = pd.read_csv(CLIPLIST)

for index, row in df.iterrows():
    noext, ext = os.path.splitext(row.clipname)   
    posfilename = OUTDIR +  '/TRACKS_' + noext + '.csv'
    gridfilename = OUTDIR + '/GRID_' + str(index) + '_' + noext + '.npy'
    gridPosfilename = OUTDIR + '/GRIDPOS_' + str(index) + '_' + noext + '.npy'
    posDF = pd.read_csv(posfilename) 
   
    posDF = posDF[posDF['frame']%10==0]
    #posDF['x']=posDF['x']#-min(posDF['x'])
    #posDF['y']=posDF['y']#-min(posDF['y'])
    xrange = max(posDF['x'])-min(posDF['x'])
    yrange = max(posDF['y'])-min(posDF['y'])
    minx = math.floor(min(posDF['x']))
    miny = math.floor(min(posDF['y']))
    nx = math.ceil(xrange/1)
    ny = math.ceil(yrange/1)
    grid = np.zeros((nx,ny,2))
    gridPos = np.zeros((nx,ny,2))
    xdirs = posDF['dx'].values
    ydirs = posDF['dy'].values
    xp = posDF['x'].values
    yp = posDF['y'].values
    kappa = 4
    for i in range(nx):
        for j in range(ny):
            gx = (i ) + minx
            gy = (j ) + miny
            dists = (((xp - gx)**2 + (yp - gy)**2))
            weights = np.exp(-dists/kappa)
            gridPos[i,j,0]= gx
            gridPos[i,j,1]= gy
            xav = np.sum(weights*xdirs)/np.sum(weights)
            yav = np.sum(weights*ydirs)/np.sum(weights)
            grid[i,j,0]=xav/math.sqrt(xav**2+yav**2)
            grid[i,j,1]=yav/math.sqrt(xav**2+yav**2)
    
    np.save(gridfilename, grid)
    np.save(gridPosfilename, gridPos)
    #plt.quiver(xp,yp,xdirs,ydirs,angles='xy', scale_units='xy', color='r', scale=1.0/32.0)
    
#    plt.axis([0,nx,0,ny])
#    plt.axis('equal')
#    l, = plt.quiver(gridPos[::3,::3,0],gridPos[::3,::3,1],grid[::3,::3,0],grid[::3,::3,1],angles='xy', scale_units='xy', scale=0.5)
#    l.axes.set_xlim(0,nx)
#    l.axes.set_ylim(0,ny)
#    break

    

