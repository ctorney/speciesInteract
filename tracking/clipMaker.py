
import cv2
import numpy as np
import pandas as pd
import os
import re
import time

HD = os.getenv('HOME')


DATADIR = HD + '/data/wildebeest/lacey-field-2015/'
CLIPDIR = HD + '/data/wildebeest/lacey-field-2015/wildzeb/'
CLIPLIST = HD + '/workspace/speciesInteract/clipList.csv'


df = pd.read_csv(CLIPLIST)
df['clipname']=''

for index, row in df.iterrows():

    if index<11:
        continue
    h,m,s = re.split(':',row.start)
    timeStart = int(h)*3600+int(m)*60+int(s)
    h,m,s = re.split(':',row.stop)
    timeStop = int(h)*3600+int(m)*60+int(s)

    inputName = DATADIR + row.folder + '/' + row.filename
    outputName = time.strftime("%Y%m%d", time.strptime(row.date,"%d-%b-%Y")) + '-' + str(index) + '.avi'

    df.loc[index,'clipname'] = outputName
    
    print('Movie ' + row.folder + '/' + row.filename + ' from ' + str(timeStart) + ' to ' + str(timeStop) + ' out to ' + outputName)
    cap = cv2.VideoCapture(inputName)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    
    fStart = timeStart*fps
    fStop = timeStop*fps

    cap.set(cv2.CAP_PROP_POS_FRAMES,fStart)
    S = (1920,1080)
    
    ds = 6
    out = cv2.VideoWriter(CLIPDIR + outputName, cv2.VideoWriter_fourcc('M','J','P','G'), fps/ds, S, True)

  
    global allTransforms
    for tt in range(fStop-fStart):
        # Capture frame-by-frame
        _, frame = cap.read()
        if frame is None:
            break
        if (tt%ds==0):
            outFrame = cv2.resize(frame, S) #in case size has been altered by fecking windows
            out.write(outFrame)

    cap.release()
    out.release()
    
df.to_csv(CLIPLIST,index=False)


