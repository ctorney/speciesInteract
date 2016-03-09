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
    if index!=4:
        continue
    inputName = row.clipname
    wildeBL = row.bl_w
    noext, ext = os.path.splitext(inputName)
    trackname = CLIPDIR + '/FINAL_' + noext + '.csv'
    
    
    outname = OUTDIR +  '/TRACKS_' + noext + '.csv'

    posDF = pd.read_csv(trackname) 
    
    outTracks = pd.DataFrame(columns= ['frame','x','y','dx','dy','heading','vx','vy','ax','ay','id','animal'])
    
    #smoothing
    winLen = 10
    vwinLen = 30
    w = np.kaiser(winLen,1)
    w = w/w.sum()
    w2 = np.kaiser(vwinLen,1)
    w2 = w2/w2.sum()
    for cnum, cpos in posDF.groupby('id'):
        if len(cpos)<10:
            continue
        ft = cpos['frame'].values
        ani = cpos['animal'].values[0]
        xd = cpos['x'].values*(2.0/wildeBL)
        xd = np.r_[np.ones((winLen))*xd[0],xd,np.ones((winLen))*xd[-1]]
        xSmooth = np.convolve(w/w.sum(),xd,mode='same')[(winLen):-(winLen)]
        xv = np.diff(xSmooth)
        xv = np.r_[np.ones((vwinLen))*xv[0],xv,np.ones((vwinLen))*xv[-1]]
        xv = np.convolve(w2/w2.sum(),xv,mode='same')[(vwinLen):-(vwinLen-1)]
        xa = np.diff(xv)
        xa = np.r_[np.ones((vwinLen))*xa[0],xa,np.ones((vwinLen))*xa[-1]]
        xa = np.convolve(w2/w2.sum(),xa,mode='same')[(vwinLen):-(vwinLen-1)]
        yd = cpos['y'].values*(2.0/wildeBL)
        yd = np.r_[np.ones((winLen))*yd[0],yd,np.ones((winLen))*yd[-1]]
        ySmooth = np.convolve(w/w.sum(),yd,mode='same')[(winLen):-(winLen)]
        yv = np.diff(ySmooth)
        yv = np.r_[np.ones((vwinLen))*yv[0],yv,np.ones((vwinLen))*yv[-1]]
        yv = np.convolve(w2/w2.sum(),yv,mode='same')[(vwinLen):-(vwinLen-1)]
        ya = np.diff(yv)
        ya = np.r_[np.ones((vwinLen))*ya[0],ya,np.ones((vwinLen))*ya[-1]]
        ya = np.convolve(w2/w2.sum(),ya,mode='same')[(vwinLen):-(vwinLen-1)]
        #xSmooth = xSmooth[(winLen):-(winLen)]
        headings = np.zeros_like(xSmooth)
        dx = np.zeros_like(xSmooth)
        dy = np.zeros_like(xSmooth)
        for i in range(len(headings)):
            start = max(0,i-5)
            stop = min(i+5,len(headings))-1
            dx[i] = xSmooth[stop]-xSmooth[start]
            dy[i] = ySmooth[stop]-ySmooth[start]
        headings = np.arctan2(dy,dx)
        #headings[-1]=headings[-2] 
#        plot arrows for error checking
#        if cnum==100:
#            x=xSmooth[0:-1]
#            y=ySmooth[0:-1]
#            u = np.cos(headings)
#            v = np.sin(headings)
#            plt.quiver(x, y, u ,v) 
#            plt.axes().set_aspect('equal')
#            plt.show()
#            break

        newcpos = pd.DataFrame(np.column_stack((ft,xSmooth,ySmooth,dx,dy,headings,xv,yv,xa,ya)), columns= ['frame','x','y','dx','dy','heading','vx','vy','ax','ay'])
        newcpos['id']=cnum
        newcpos['animal']=ani
        outTracks = outTracks.append(newcpos,ignore_index=True )
        
       # for ind,posrow in dfFrame.iterrows():
       #     # get pixel coordinates
       #     xx=posrow['x_px']
       #     yy=posrow['y_px']
            
       #     [mx,my,_] = np.dot(full_warp, np.array([[xx],[yy],[1]]))
       #     posDF.set_value(ind,'x',mx)
       #     posDF.set_value(ind,'y',my)
        
    #    break


    
    
    outTracks.to_csv(outname, index=False)

