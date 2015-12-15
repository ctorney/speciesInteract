
import cv2
import numpy as np
import pandas as pd
import os
import re
import math
import time

HD = os.getenv('HOME')


DATADIR = HD + '/data/wildebeest/lacey-field-2015/'
CLIPDIR = HD + '/data/wildebeest/lacey-field-2015/wildzeb/'
CLIPLIST = HD + '/workspace/speciesInteract/clipList.csv'


df = pd.read_csv(CLIPLIST)
df['clipname']=''

warp_mode = cv2.MOTION_EUCLIDEAN
warp_mode = cv2.MOTION_HOMOGRAPHY
number_of_iterations = 20

# Specify the threshold of the increment
# in the correlation coefficient between two iterations
termination_eps = -1e-8;

# Define termination criteria
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

for index, row in df.iterrows():
    if index!=2:
        continue
    

    h,m,s = re.split(':',row.start)
    timeStart = int(h)*3600+int(m)*60+int(s)
    h,m,s = re.split(':',row.stop)
    timeStop = int(h)*3600+int(m)*60+int(s)

    inputName = DATADIR + row.folder + '/' + row.filename
    outputName = time.strftime("%Y%m%d", time.strptime(row.date,"%d-%b-%Y")) + '-' + str(index) + '.avi'

    df.loc[index,'clipname'] = outputName
    
    
    print('Movie ' + row.folder + '/' + row.filename + ' from ' + str(timeStart) + ' to ' + str(timeStop) + ' out to ' + outputName)
#   if index<6: continue
#    if index>10: continue

    cap = cv2.VideoCapture(inputName)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    
    fStart = timeStart*fps
    fStop = timeStop*fps

    cap.set(cv2.CAP_PROP_POS_FRAMES,fStart)
    S = (1920,1080)
    
    # reduce to ten frames a second
    ds = math.ceil(fps/10)
    out = cv2.VideoWriter(CLIPDIR + outputName, cv2.VideoWriter_fourcc('M','J','P','G'), fps/ds, S, True)

  
    im1_gray = np.array([])
    first = np.array([])
    #cap.set(cv2.CAP_PROP_POS_FRAMES,480)
    warp_matrix = np.eye(3, 3, dtype=np.float32) 
    full_warp = np.eye(3, 3, dtype=np.float32)
    for tt in range(fStart,fStop):
        # Capture frame-by-frame
        _, frame = cap.read()
        if (tt%ds!=0):
            continue
        if frame.shape!=S:
            frame = cv2.resize(frame, S) #in case size has been altered by fecking windows
        if not(im1_gray.size):
            im1_gray = cv2.equalizeHist(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
#im1_gray *= 1.5#cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

            first = frame.copy()
        
        im2_gray =  cv2.equalizeHist(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
        

        try:
            (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)
# (cc, warp_matrix) = cv2.findTransformECC(im2_gray,im1_gray,warp_matrix, warp_mode, criteria)    
        except cv2.error as e:
            im1_gray = cv2.equalizeHist(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
            first = frame.copy()
            print("missed frame")
#im1_gray =im2_gray.copy()
        # keep track of all the transformations to this point
        
#full_warp = np.dot(full_warp,np.vstack((warp_matrix,[0,0,1])))
         #full_warp = np.dot(full_warp, np.vstack((warp_matrix,[0,0,1])))
         # full_warp = np.dot(full_warp, warp_matrix)
#    full_warp = np.dot(warp_matrix,full_warp)
        im2_aligned = np.empty_like(frame)
        np.copyto(im2_aligned, first)
        # transform the frame - the 5s are to cut out the border that the shitty windows conversion created for no reason
        im2_aligned = cv2.warpPerspective(frame[5:-5,5:-5,:], warp_matrix, (S[0],S[1]), dst=im2_aligned, flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP  , borderMode=cv2.BORDER_TRANSPARENT)
         # im2_aligned = cv2.warpPerspective(frame[5:-5,5:-5,:], full_warp, (S[0],S[1]), dst=im2_aligned, flags=cv2.INTER_NEAREST  , borderMode=cv2.BORDER_TRANSPARENT)
#im2_aligned = cv2.warpAffine(frame[5:-5,5:-5,:], warp_matrix, (S[0],S[1]), dst=im2_aligned, flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP , borderMode=cv2.BORDER_TRANSPARENT)
#        im2_aligned = cv2.warpAffine(frame[5:-5,5:-5,:], full_warp[0:2,:], (S[0],S[1]), dst=im2_aligned, flags=cv2.INTER_NEAREST  , borderMode=cv2.BORDER_TRANSPARENT)
        out.write(im2_aligned)
        

    cap.release()
    out.release()
    
#df.to_csv(CLIPLIST,index=False)


