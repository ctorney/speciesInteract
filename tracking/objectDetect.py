

import cv2
import numpy as np
import pandas as pd
import os
import re
import math
import time

HD = os.getenv('HOME')

DD = '/media/ctorney/SAMSUNG/'

DATADIR = DD + '/data/wildebeest/lacey-field-2015/'
CLIPDIR = DD + '/data/wildebeest/lacey-field-2015/wildzeb/'
CLIPLIST = HD + '/workspace/speciesInteract/clipList.csv'

params = cv2.SimpleBlobDetector_Params()
params.filterByColor = False;
params.filterByArea= 1
params.filterByCircularity= 0
params.filterByInertia= 0
params.filterByConvexity= 0
params.minArea = 5
blobdetector = cv2.SimpleBlobDetector_create(params)

outputMovie=0

df = pd.read_csv(CLIPLIST)

for index, row in df.iterrows():
    if index<4:
        continue
    # pandas file for export of positions
    dfPos = pd.DataFrame(columns= ['x', 'y', 'frame'])  
    
    
    inputName = row.clipname
    noext, ext = os.path.splitext(inputName)
    posfilename = CLIPDIR + '/' + noext + '.csv'
    
    cap = cv2.VideoCapture(CLIPDIR + inputName)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    fCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    S = (1920,1080)

    if outputMovie:
        outputName = 'tmp.avi'
        out = cv2.VideoWriter(CLIPDIR + outputName, cv2.VideoWriter_fourcc('M','J','P','G'), fps, S, True)

    pMOG2 = cv2.createBackgroundSubtractorMOG2(detectShadows=1)


    # use the first ten seconds to create the background
    for tt in range(100):
        _, frame = cap.read()
        fgmask = pMOG2.apply(frame)
    cv2.imwrite(CLIPDIR + noext + '.png',frame)
  
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    maxB = 200
    for tt in range(fCount):
        # Capture frame-by-frame
        _, frame = cap.read()
        fgmask = pMOG2.apply(frame)
        _,fgmask = cv2.threshold(fgmask,227,255,cv2.THRESH_BINARY)
        imOut = cv2.cvtColor(fgmask,cv2.COLOR_GRAY2BGR)
        blobs= blobdetector.detect(imOut)
        # draw detected objects and display
        sz=6
        thisFrame = pd.DataFrame(columns= ['x', 'y', 'frame'])
        # get altitude of frame
        ind = 0
        for b in blobs:
            
            if outputMovie:
                cv2.rectangle(frame, ((int(b.pt[0])-sz, int(b.pt[1])-sz)),((int(b.pt[0])+sz, int(b.pt[1])+sz)),(0,0,0),2)
            thisFrame.set_value(ind, 'x', b.pt[0])
            thisFrame.set_value(ind, 'y', b.pt[1])
            thisFrame.set_value(ind, 'frame', tt)
            ind +=1
            if ind>maxB:
                break
        dfPos = pd.concat([dfPos,thisFrame])
        if outputMovie:
            out.write(frame)
        

    cap.release()
    if outputMovie:
        out.release()
    dfPos.to_csv(posfilename)
    


