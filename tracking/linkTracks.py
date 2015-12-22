

import cv2
import numpy as np
import pandas as pd
import os
import re
import math
import time
from scipy import interpolate

import trackpy.predict

HD = os.getenv('HOME')
DD = '/media/ctorney/SAMSUNG/'

DATADIR = DD + '/data/wildebeest/lacey-field-2015/'
CLIPDIR = DD + '/data/wildebeest/lacey-field-2015/wildzeb/'
CLIPLIST = HD + '/workspace/speciesInteract/clipList.csv'



df = pd.read_csv(CLIPLIST)

for index, row in df.iterrows():
    if index<21:
        continue
    
    # pandas file for export of positions
    dfPos = pd.DataFrame(columns= ['x', 'y', 'frame'])  
    
    
    inputName = row.clipname
    noext, ext = os.path.splitext(inputName)
    posfilename = CLIPDIR + '/' + noext + '.csv'
    trackname = CLIPDIR + '/TRACKS_' + noext + '.csv'
    
    
    toLink = pd.read_csv(posfilename,index_col=0)
    pred = trackpy.predict.NearestVelocityPredict()




    f_iter = (frame for fnum, frame in toLink.groupby('frame'))
    t = pd.concat(pred.link_df_iter(f_iter, 5.00, memory=20))
    outTracks = pd.DataFrame(columns= ['frame','x','y','id'])

    minFrames = 10
    p_id = 0
    for cnum, cpos in t.groupby('particle'):
        # delete tracks that are too short
        frameLen = max(cpos['frame'])-min(cpos['frame'])
        if frameLen<minFrames:
            continue
        
        # interpolate to smooth and fill in any missing frames        
        frameTimes = np.arange(min(cpos['frame'])+1,max(cpos['frame']),1)  
        posData = cpos[['x','y']].values
        timeData= cpos[['frame']].values
        tData = interpolate.interp1d(timeData[:,0], posData.T)(frameTimes)
        tData=tData.T

    
        
        newcpos = pd.DataFrame(np.column_stack((frameTimes,tData)), columns= ['frame','x','y'])  
        newcpos['id']=p_id
        p_id+=1
        outTracks = outTracks.append(newcpos,ignore_index=True )



        

    outTracks.to_csv(trackname, index=True)
    


