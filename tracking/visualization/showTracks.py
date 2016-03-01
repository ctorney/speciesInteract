
import operator
import cv2
import numpy as np
import os,sys
import re
import math 
import pandas as pd
import random

def cvDrawDottedLine(img, pt1, pt2, color, thickness, lengthDash, lengthGap):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    
    start = 0
    while start<dist:
        stop=min(dist,start+lengthDash)
        x1=int((pt1[0]*(1-start/dist)+pt2[0]*start/dist))
        y1=int((pt1[1]*(1-start/dist)+pt2[1]*start/dist))
        x2=int((pt1[0]*(1-stop/dist)+pt2[0]*stop/dist))
        y2=int((pt1[1]*(1-stop/dist)+pt2[1]*stop/dist))
        cv2.line(img,(x1,y1),(x2,y2),color,thickness)
        start += (lengthDash+lengthGap)
            
def cvDrawDottedRect(img, x, y, color):

    hwidth = 20
    corner = 4
    cross = 6

#    // corners 
    
    p1=np.array((x-hwidth,y-hwidth))
    p2=np.array((x+hwidth,y-hwidth))
    p3=np.array((x+hwidth,y+hwidth))
    p4=np.array((x-hwidth,y+hwidth))
    #// draw box
    cvDrawDottedLine(img, p1, p2, color, 2, 2, 4)
    cvDrawDottedLine(img, p2, p3, color, 2, 2, 4)
    cvDrawDottedLine(img, p3, p4, color, 2, 2, 4)
    cvDrawDottedLine(img, p4, p1, color, 2, 2, 4)
    #// draw corners
    x_off=np.array((corner, 0))
    y_off=np.array((0, corner))
    cvDrawDottedLine(img, p1, p1 + x_off, color, 3, 10, 10)
    cvDrawDottedLine(img, p1, p1 + y_off, color, 3, 10, 10)
    cvDrawDottedLine(img, p2, p2 - x_off, color, 3, 10, 10)
    cvDrawDottedLine(img, p2, p2 + y_off, color, 3, 10, 10)
    cvDrawDottedLine(img, p3, p3 - x_off, color, 3, 10, 10)
    cvDrawDottedLine(img, p3, p3 - y_off, color, 3, 10, 10)
    cvDrawDottedLine(img, p4, p4 + x_off, color, 3, 10, 10)
    cvDrawDottedLine(img, p4, p4 - y_off, color, 3, 10, 10)
    #// draw cross
    x_coff=np.array((cross, 0))
    y_coff=np.array((0, cross))
    p1 = np.array((x,y-hwidth))
    p2 = np.array((x,y+hwidth))
    p3 = np.array((x-hwidth,y))
    p4 = np.array((x+hwidth,y))
    cvDrawDottedLine(img, p1 + y_coff, p1, color, 1, 10, 10)
    cvDrawDottedLine(img, p2, p2 - y_coff, color, 1, 10, 10)
    cvDrawDottedLine(img, p3 + x_coff, p3, color, 1, 10, 10)
    cvDrawDottedLine(img, p4 - x_coff, p4, color, 1, 10, 10)




HD = os.getenv('HOME')
DD = '/media/ctorney/SAMSUNG/'

DATADIR = DD + '/data/wildebeest/lacey-field-2015/'
CLIPDIR = DD + '/data/wildebeest/lacey-field-2015/wildzeb/'
CLIPLIST = HD + '/workspace/speciesInteract/clipList.csv'
df = pd.read_csv(CLIPLIST)

show_index = 22
outputMovie=0

for index, row in df.iterrows():
    if index!=show_index:
        continue


    inputName = row.clipname
    noext, ext = os.path.splitext(inputName)
    tfilename = CLIPDIR + '/FINAL_' + noext + '.csv'
    #tfilename = CLIPDIR + '/TRACKS_' + noext + '.csv'
    posfilename = CLIPDIR + '/' + noext + '.csv'
    #tfilename= CLIPDIR + '/' + noext + '.csv'
    
    linkedDF = pd.read_csv(tfilename) 
    posDF = pd.read_csv(posfilename) 

    

    
    
    
    cap = cv2.VideoCapture(CLIPDIR + inputName)
    
    
    
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    fCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    
    
    S = (1920,1080)
    
    sf=0.5
    if outputMovie:
        out = cv2.VideoWriter('tmp'+str(random.randint(0,10000))+ '.avi', cv2.VideoWriter_fourcc('M','J','P','G'), cap.get(cv2.CAP_PROP_FPS), S, True)
    running = True
    cframe = 0
    while cframe<fCount:
    #for tt in range(fCount):

        print(cframe)
        thisFrame = linkedDF.ix[linkedDF['frame']==(cframe)]
        if running:
            _, frame = cap.read()
            cframe+=1
        
            
        

        
        # draw detected objects and display
        sz=20
        
        for i, trrow in thisFrame.iterrows():
            if str((trrow['animal']))=='w':
                if not(running):
                    cv2.putText(frame ,str(int(trrow['id'])) ,((int(trrow['x'])-5, int(trrow['y'])-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,0,0),2)
               # cv2.rectangle(frame, ((int( trrow['x'])-sz, int( trrow['y'])-sz)),((int( trrow['x'])+sz, int( trrow['y'])+sz)),(0,0,0),2)
                cv2.circle(frame, ((int( trrow['x']), int( trrow['y']))),3,(225,50,50),-1)
                #cv2.putText(frame ,str(int(trrow['id'])) ,((int(trrow['x'])-5, int(trrow['y'])-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,0,0),2)
    #            #cvDrawDottedRect(frame, int( trrow['x']), int( trrow['y']),(225,50,50))
            if str((trrow['animal']))=='z':
           #     if not(running):
           #         cv2.putText(frame ,str(int(trrow['id'])) ,((int(trrow['x'])-5, int(trrow['y'])-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,0,0),2)
            #cv2.rectangle(frame, ((int( trrow['x'])-sz, int( trrow['y'])-sz)),((int( trrow['x'])+sz, int( trrow['y'])+sz)),(0,0,0),2)
                cv2.circle(frame, ((int( trrow['x']), int( trrow['y']))),3,(34,34,200),-1)
                #cvDrawDottedRect(frame, int( trrow['x']), int( trrow['y']),(34,34,200))
            
  #      thisFrame = posDF.ix[posDF['frame']==(tt)]

        
        # draw detected objects and display
        sz=6
        
        #for i, trrow in thisFrame.iterrows():
            #cv2.putText(frame ,str(int(trrow['id'])) ,((int(trrow['x'])+12, int(trrow['y'])+12)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,255,2)
            #cv2.rectangle(frame, ((int( trrow['x'])-sz, int( trrow['y'])-sz)),((int( trrow['x'])+sz, int( trrow['y'])+sz)),(0,0,0),2)
                
        
        if outputMovie:
            out.write(frame)
        cv2image = cv2.resize(frame,(0,0), fx=sf, fy=sf)
        cv2.imshow('frame',cv2image)
        k = cv2.waitKey(1) & 0xff
        if k==ord('p'):
            running = not(running)
        if k == 27:
            break
    
    cv2.destroyAllWindows()
    cap.release()
    if outputMovie:
        out.release()

