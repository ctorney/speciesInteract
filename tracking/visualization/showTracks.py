

import cv2
import numpy as np
import os,sys
import re
import math 
import pandas as pd
import random

HD = os.getenv('HOME')
DD = '/media/ctorney/SAMSUNG/'

DATADIR = DD + '/data/wildebeest/lacey-field-2015/'
CLIPDIR = DD + '/data/wildebeest/lacey-field-2015/wildzeb/'
CLIPLIST = HD + '/workspace/speciesInteract/clipList.csv'
df = pd.read_csv(CLIPLIST)

show_index = 2
outputMovie=True

for index, row in df.iterrows():
    if index!=show_index:
        continue


    inputName = row.clipname
    noext, ext = os.path.splitext(inputName)

    tfilename = CLIPDIR + '/FINAL_' + noext + '.csv'
    posfilename = CLIPDIR + '/' + noext + '.csv'

    
    linkedDF = pd.read_csv(tfilename) 
    posDF = pd.read_csv(posfilename) 

    

    
    
    
    cap = cv2.VideoCapture(CLIPDIR + inputName)
    
    
    
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    fCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    
    
    S = (1920,1080)
    
    
    if outputMovie:
        out = cv2.VideoWriter('tmp'+str(random.randint(0,10000))+ '.avi', cv2.VideoWriter_fourcc('M','J','P','G'), cap.get(cv2.CAP_PROP_FPS), S, True)
    
    for tt in range(fCount):

        _, frame = cap.read()
        
            
        thisFrame = linkedDF.ix[linkedDF['frame']==(tt)]

        
        # draw detected objects and display
        sz=10
        
        for i, trrow in thisFrame.iterrows():
            cv2.putText(frame ,str(trrow['animal'])+str(int(trrow['id'])) ,((int(trrow['x'])+12, int(trrow['y'])+12)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,255,2)
            cv2.rectangle(frame, ((int( trrow['x'])-sz, int( trrow['y'])-sz)),((int( trrow['x'])+sz, int( trrow['y'])+sz)),(0,0,0),2)
        
            
        thisFrame = posDF.ix[posDF['frame']==(tt)]

        
        # draw detected objects and display
        sz=6
        
        #for i, trrow in thisFrame.iterrows():
            #cv2.putText(frame ,str(int(trrow['id'])) ,((int(trrow['x'])+12, int(trrow['y'])+12)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,255,2)
            #cv2.rectangle(frame, ((int( trrow['x'])-sz, int( trrow['y'])-sz)),((int( trrow['x'])+sz, int( trrow['y'])+sz)),(0,0,0),2)
                
        
        if outputMovie:
            out.write(frame)
            
        cv2.imshow('frame',frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    
    cv2.destroyAllWindows()
    cap.release()
    out.release()

