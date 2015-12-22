

import cv2
import numpy as np
import pandas as pd
import os, sys
import math
from sklearn.ensemble import AdaBoostClassifier

import pickle
sys.path.append('./training/.')
from circularHOGExtractor import circularHOGExtractor
ch = circularHOGExtractor(4,2,4) 

fhgClass = pickle.load( open( "./boostClassifier.p", "rb" ) )
def checkIsAnimal(x,y,frame,z_sz):
    nx = 1920
    ny = 1080
    # work out size of box if box if 32x32 at 100m
# use a zebra sized box but rescale later
    grabSize = math.ceil(0.75*z_sz)
    tmpImg =  cv2.cvtColor(frame[max(0,y-grabSize):min(ny,y+grabSize), max(0,x-grabSize):min(nx,x+grabSize)].copy(), cv2.COLOR_BGR2GRAY)

            
    if tmpImg.size == 4*grabSize*grabSize:# and tmpImg[tmpImg==0].size<10 :
        res = fhgClass.predict(ch.extract(cv2.resize(tmpImg,(32,32))))
        return res[0] # 0 for not an animal, 1 for wildebeest, 2 for zebra

    return 0
HD = os.getenv('HOME')

DD = '/media/ctorney/SAMSUNG/'

DATADIR = DD + '/data/wildebeest/lacey-field-2015/'
CLIPDIR = DD + '/data/wildebeest/lacey-field-2015/wildzeb/'
CLIPLIST = HD + '/workspace/speciesInteract/clipList.csv'

outputMovie=0

df = pd.read_csv(CLIPLIST)

for index, row in df.iterrows():
    if index==20:
        continue
    if index<9:
        continue
    inputName = row.clipname
    zebra_sz = row.bl_z
    noext, ext = os.path.splitext(inputName)
    trackName = CLIPDIR + '/TRACKS_' + noext + '.csv'
    outName = CLIPDIR + '/FINAL_' + noext + '.csv'
    linkedDF = pd.read_csv(trackName,index_col=0) 
    numPars = int(linkedDF['id'].max()+1)    
    
    wildCount = np.zeros(numPars)
    zebCount = np.zeros(numPars)
    noneCount = np.zeros(numPars)
    cap = cv2.VideoCapture(CLIPDIR + inputName)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    fCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    S = (1920,1080)




    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    for tt in range(fCount):
        # Capture frame-by-frame
        _, frame = cap.read()
        thisFrame = linkedDF.ix[linkedDF['frame']==tt]
        for i, row in thisFrame.iterrows():
    
            ix = int(row['x'])
            iy = int(row['y'])
            pid = int(row['id'])



            aniClass = checkIsAnimal(ix,iy,frame,zebra_sz)
            
            if aniClass==0:
                noneCount[pid]+=1
            if aniClass==1:
                wildCount[pid]+=1
            if aniClass==2:
                zebCount[pid]+=1

        
    cap.release()
    for i in range(numPars):
        if (noneCount[i]>wildCount[i])&(noneCount[i]>zebCount[i]):
            linkedDF.loc[linkedDF['id']==i,'animal']='n'
        if (wildCount[i]>noneCount[i])&(wildCount[i]>zebCount[i]):
            linkedDF.loc[linkedDF['id']==i,'animal']='w'
        if (zebCount[i]>noneCount[i])&(zebCount[i]>wildCount[i]):
            linkedDF.loc[linkedDF['id']==i,'animal']='z'
    linkedDF.to_csv(outName)
    


