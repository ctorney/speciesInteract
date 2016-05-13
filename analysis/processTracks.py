import numpy as np
import pandas as pd
import os, re
import math
import time
from scipy import interpolate
from scipy import ndimage
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
HD = os.getenv('HOME')
DD = '/media/ctorney/SAMSUNG/'

DATADIR = DD + '/data/wildebeest/lacey-field-2015/'
CLIPDIR = DD + '/data/wildebeest/lacey-field-2015/wildzeb/'
CLIPLIST = HD + '/workspace/speciesInteract/clipListReduced.csv'
OUTDIR = HD + '/Dropbox/Wildebeest_collaboration/Data/w_z/'


# converted to 10 fps


# transitions for 2d movement with positions, velocity and acceleration
transition_matrix = [[1,0,1,0,0.5,0], [0,1,0,1,0,0.5], [0,0,1,0,1,0], [0,0,0,1,0,1], [0,0,0,0,1,0], [0,0,0,0,0,1]]

# observe only positions
observation_matrix = [[1,0,0,0,0,0], [0,1,0,0,0,0]]

# low noise on transitions
transition_covariance = np.eye(6)*1e-3
observation_covariance_m = np.eye(2)*2

kf = KalmanFilter(transition_matrices = transition_matrix, observation_matrices = observation_matrix, transition_covariance=transition_covariance,observation_covariance=observation_covariance_m)






df = pd.read_csv(CLIPLIST)
for index, row in df.iterrows():
    inputName = row.clipname
    wildeBL = row.bl_w
    noext, ext = os.path.splitext(inputName)
    trackname = CLIPDIR + '/FINAL_' + noext + '.csv'
    
    
    outname = OUTDIR +  '/TRACKS_' + noext + '.csv'

    posDF = pd.read_csv(trackname) 
    
    outTracks = pd.DataFrame(columns= ['frame','x','y','dx','dy','heading','vx','vy','ax','ay','id','animal'])
    px_to_m = (2.0/wildeBL) # approximate metres by wildebeest body length of 2 metres
    
    for cnum, cpos in posDF.groupby('id'):
        ft = np.arange(cpos['frame'].values[0],cpos['frame'].values[-1]+1)
        #obs = np.vstack((cpos['x'].values, cpos['y'].values)).T
        obs = np.zeros((len(ft),2))
        obs = np.ma.array(obs, mask=np.zeros_like(obs))
        for f in range(len(ft)):
            if len(cpos[cpos['frame']==ft[f]].x.values)>0:
                obs[f][0]=cpos[cpos['frame']==ft[f]].x.values[0]*px_to_m
                obs[f][1]=cpos[cpos['frame']==ft[f]].y.values[0]*px_to_m
            else:
                obs[f]=np.ma.masked

        kf.initial_state_mean=[cpos['x'].values[0]*px_to_m,cpos['y'].values[0]*px_to_m,0,0,0,0]
        sse = kf.smooth(obs)[0]

        ani = cpos['animal'].values[0]

        xSmooth = sse[:,0]
        ySmooth = sse[:,1]
        xv = sse[:,2]/0.1
        yv = sse[:,3]/0.1
        xa = sse[:,4]/0.01
        ya = sse[:,5]/0.01
        headings = np.zeros_like(xSmooth)
        dx = np.zeros_like(xSmooth)
        dy = np.zeros_like(xSmooth)
        for i in range(len(headings)):
            start = max(0,i-5)
            stop = min(i+5,len(headings))-1
            dx[i] = xSmooth[stop]-xSmooth[start]
            dy[i] = ySmooth[stop]-ySmooth[start]
        headings = np.arctan2(yv,xv)
        
        
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

