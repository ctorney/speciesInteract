
import cv2
import numpy as np
import os,sys
import math as m
params = cv2.SimpleBlobDetector_Params()
params.filterByColor = False;
params.filterByArea= 1
params.filterByCircularity= 0
params.filterByInertia= 0
params.filterByConvexity= 0
params.minArea = 5
blobdetector = cv2.SimpleBlobDetector_create(params)

def getBackground(filename, outputfilename, start, frames):
    cap = cv2.VideoCapture(filename)
    cap.set(cv2.CAP_PROP_POS_FRAMES,start)


    S = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(outputfilename, cv2.VideoWriter_fourcc('M','J','P','G'), cap.get(cv2.CAP_PROP_FPS), (1920,1080), True)


#  pMOG2 = cv2.createBackgroundSubtractorMOG2()
    pMOG2 = cv2.createBackgroundSubtractorMOG2(detectShadows=0)

#   avIm = np.array([])
    for tt in range(frames):
    # Capture frame-by-frame
        _, frame = cap.read()
        img = pMOG2.apply(frame)
        ret,fgMaskMOG2 = cv2.threshold(img,227,255,cv2.THRESH_BINARY)
        print tt
        
        if tt==0:
            lastIm = fgMaskMOG2.copy()

#        combIm = cv2.bitwise_and(lastIm,fgMaskMOG2)
        combIm = cv2.addWeighted(lastIm,0.5,fgMaskMOG2,0.5,0)
        lastIm = fgMaskMOG2.copy()

#       blobs= blobdetector.detect(combIm)
#            avIm = np.zeros_like(fgMaskMOG2)
        sz=6
        blank = np.zeros_like(frame)
#       for b in blobs:
#           cv2.rectangle(frame, ((int(b.pt[0])-sz, int(b.pt[1])-sz)),((int(b.pt[0])+sz, int(b.pt[1])+sz)),(0,0,0),2)
#            cv2.circle(blank, ((int(b.pt[0])-sz, int(b.pt[1])-sz)),3,(255,255,255),-2)
#       if t    t>100:
#   color_img =
        imOut = np.zeros_like(combIm)

        imOut[:,335:1585]=combIm[:,0:1250]#=combIm[:,1250:]/10
        out.write( cv2.cvtColor(imOut, cv2.COLOR_GRAY2BGR))

        if tt==60:
            cv2.imwrite('first.png', imOut );
#        cv2.imshow('framea',combIm)
#        k = cv2.waitKey(30) & 0xff
#    
#        if k == 27:
#            break
    cv2.imwrite('last.png', imOut );

    imBk = pMOG2.getBackgroundImage()
    cv2.imwrite('bkGround.png', imBk );
    
if __name__ == '__main__':
    FULLNAME = sys.argv[1]
    frameStart = int(sys.argv[2])
    frameLength = int(sys.argv[3])
    path, filename = os.path.split(FULLNAME)
    noext, ext = os.path.splitext(filename)
    outputname = noext + '_FR' + str(frameStart) + '.avi' 
    getBackground(FULLNAME, outputname, frameStart, frameLength)
    print outputname


