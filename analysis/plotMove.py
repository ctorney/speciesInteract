import numpy as np
import pandas as pd
import os, re
import math
import time
from scipy import interpolate
from scipy import ndimage
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib.animation as ani
HD = os.getenv('HOME')
DD = '/media/ctorney/SAMSUNG/'

DATADIR = DD + '/data/wildebeest/lacey-field-2015/'
CLIPDIR = DD + '/data/wildebeest/lacey-field-2015/wildzeb/'
CLIPLIST = HD + '/workspace/speciesInteract/clipListReduced.csv'

OUTDIR = HD + '/Dropbox/Wildebeest_collaboration/Data/w_z/'

df = pd.read_csv(CLIPLIST)

for index, row in df.iterrows():
    if index!=16:
        continue
    noext, ext = os.path.splitext(row.clipname)   
    posfilename = OUTDIR +  '/TRACKS_' + noext + '.csv'
    gridfilename = OUTDIR + '/GRID_' + str(index) + '_' + noext + '.npy'
    gridPosfilename = OUTDIR + '/GRIDPOS_' + str(index) + '_' + noext + '.npy'
    posDF = pd.read_csv(posfilename) 
    
    #posDF = posDF[posDF['frame']%10==0]
    
    xrange = max(posDF['x'])-min(posDF['x'])
    yrange = max(posDF['y'])-min(posDF['y'])
    minx = math.floor(min(posDF['x']))
    miny = math.floor(min(posDF['y']))
    nx = math.ceil(xrange/1)
    ny = math.ceil(yrange/1)
            
    grid = np.load(gridfilename)
    gridPos = np.load(gridPosfilename)
    #plt.quiver(xp,yp,xh,yh,angles='xy', scale_units='xy', color='r', scale=1.0/32.0)
    #plt.quiver(gridPos[:,:,0],gridPos[:,:,1],grid[:,:,0],grid[:,:,1],angles='xy', scale_units='xy', scale=1.0/32.0)



    
    fig = plt.figure()#figsize=(10, 10), dpi=5)
    
    
    
    totalFrames =500
    fc = 0
    #with writer.saving(fig, "move.mp4", totalFrames):# len(posDF.groupby('frame'))):


    for fnum, frame in posDF.groupby('frame'):
        
        wxp = frame[frame['animal']=='w']['x'].values
        wyp = frame[frame['animal']=='w']['y'].values
        wxh = frame[frame['animal']=='w']['dx'].values
        wyh = frame[frame['animal']=='w']['dy'].values
        zxp = frame[frame['animal']=='z']['x'].values
        zyp = frame[frame['animal']=='z']['y'].values
        zxh = frame[frame['animal']=='z']['dx'].values
        zyh = frame[frame['animal']=='z']['dy'].values        

        plt.clf()
        plt.axis('equal')
        l, = plt.plot(wxp,wyp, 'ro')
        plt.quiver(gridPos[::3,::3,0],gridPos[::3,::3,1],grid[::3,::3,0],grid[::3,::3,1],angles='xy', scale_units='xy', scale=0.5,headwidth=1)
        #plt.quiver(gridPos[:,:,0],gridPos[:,:,1],grid[:,:,0],grid[:,:,1],angles='xy', scale_units='xy', scale=1.0/32.0, headwidth=1)
        
        plt.quiver(wxp,wyp,wxh,wyh,angles='xy', scale_units='xy', color='r', scale=1.0/4.0, headwidth=1.5)
        if len(zxp):
            plt.plot(zxp,zyp, 'bo')
            plt.quiver(zxp,zyp,zxh,zyh,angles='xy', scale_units='xy', color='b', scale=1.0/4.0, headwidth=1.5)
            
    #plt.axis([0,4000, 2000,-2000])
        
        #l.axes.get_xaxis().set_visible(False)
        #l.axes.get_yaxis().set_visible(False)
        l.axes.set_xlim(0,nx)
        l.axes.set_ylim(0,ny)
        
        plt.savefig('frames/fig'+'{0:05d}'.format(fc)+'.png')
        fc=fc+1
    
            #writer.grab_frame()
    break

    


