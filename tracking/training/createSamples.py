

import cv2
import numpy as np
import os,sys
import math as m
import pandas as pd

HD = os.getenv('HOME')

DATADIR = HD + '/data/wildebeest/lacey-field-2015/'
CLIPDIR = HD + '/data/wildebeest/lacey-field-2015/wildzeb/'
CLIPLIST = HD + '/workspace/speciesInteract/clipList.csv'

df = pd.read_csv(CLIPLIST)
for index, row in df.iterrows():

    if index!=2:
        continue
    
    inputName = row.clipname
    zebra_sz = row.bl_z
    noext, ext = os.path.splitext(inputName)

    trackName = CLIPDIR + '/TRACKS_' + noext + '.csv'

        

    # name the images after the track file name
    path, fileonly = os.path.split(trackName)
    noext, ext = os.path.splitext(fileonly)
    
    
    linkedDF = pd.read_csv(trackName) 
    
    nx = 1920
    ny = 1080
    
    
    cap = cv2.VideoCapture(CLIPDIR + inputName)
    
    numPars = int(linkedDF['id'].max()+1)
    
    zorw = np.zeros(shape=(0,2),dtype=int)
    box_dim = 128    
   
    
    sz=16
    frName = inputName + ' wildebeest (w) or zebra (z) or neither (n)?'
    cv2.destroyAllWindows()
    cv2.namedWindow(frName, flags =  cv2.WINDOW_NORMAL)
    escaped = False
    for i in range(numPars):
        thisPar = linkedDF[linkedDF['id']==i]
        if escaped == True:
            break
       
        
#        print(thisPar.count()[0])
        for _, row in thisPar.iterrows():
    
            ix = int(row['x'])
            iy = int(row['y'])
            fNum = int(row['frame'])
            

            
            cap.set(cv2.CAP_PROP_POS_FRAMES,fNum)
            _, frame = cap.read()
            
            
            cv2.rectangle(frame, ((int( row['x'])-sz, int( row['y'])-sz)),((int( row['x'])+sz, int( row['y'])+sz)),(0,0,0),1)
            tmpImg = frame[max(0,iy-box_dim/2):min(ny,iy+box_dim/2), max(0,ix-box_dim/2):min(nx,ix+box_dim/2)]
            
            cv2.imshow(frName,tmpImg)
            k = cv2.waitKey(1000) -  0x100000
            
            
            if k==ord('z'):
                zorw = np.vstack((zorw, [i,2]))
                break            
            if k==ord('w'):
                zorw = np.vstack((zorw, [i,1]))
                break
            if k==ord('n'):
                zorw = np.vstack((zorw, [i,0]))
                break
            if k==27:    # Esc key to stop
                escaped=True
                break 
            
    cv2.destroyAllWindows()
    for animal in zorw:
        thisPar = linkedDF[linkedDF['id']==animal[0]]
    
        for index2, row2 in thisPar.iterrows():
            if (index2%5)!=0:
                continue
            ix = int(row2['x'])
            iy = int(row2['y'])
            fNum = int(row2['frame'])
            
            
            cap.set(cv2.CAP_PROP_POS_FRAMES,fNum)
            _, frame = cap.read()
    
            # use a zebra sized box but rescale later
            grabSize = m.ceil(0.75*zebra_sz)
            tmpImg =  cv2.cvtColor(frame[max(0,iy-grabSize):min(ny,iy+grabSize), max(0,ix-grabSize):min(nx,ix+grabSize)].copy(), cv2.COLOR_BGR2GRAY)

            
            if tmpImg.size == 4*grabSize*grabSize:# and tmpImg[tmpImg==0].size<10 :
                if animal[1]==2:
                    cv2.imwrite('./zebra/' + noext + '_' + str(animal[0]) + '_' + str(fNum) + '.png',cv2.resize(tmpImg,(32,32)))
                if animal[1]==1:
                    cv2.imwrite('./wildebeest/' + noext + '_' + str(animal[0]) + '_' + str(fNum) + '.png',cv2.resize(tmpImg,(32,32)))
                if animal[1]==0:
                    cv2.imwrite('./no/' + noext + '_' + str(animal[0]) + '_' + str(fNum) + '.png',cv2.resize(tmpImg,(32,32)))
                    break
            
    
    break
    cap.release()



