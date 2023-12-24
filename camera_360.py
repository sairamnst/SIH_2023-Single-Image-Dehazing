import os
from ultralytics import YOLO
import numpy as np
import cv2
import time
from cv2.ximgproc import guidedFilter as gf
import keyboard
from playsound import playsound

global X1,X2
X1=106#img is left lower is 320,24(x,y)
X2=213#so dividing by three x1 is 106 and x2 is 213

def left():
    print("left")
    playsound('left.mp3')

def right():
    print("right")
    playsound('right.mp3')

def rescale(frame,scale=0.5):
    width =int(frame.shape[1] *scale)
    height = int(frame.shape[0] *scale)
    dimension=(width,height)
    return cv2.resize(frame,dimension,interpolation=cv2.INTER_AREA)




def dehaze(im):
    start=time.time()
    im = im.astype(np.double) / 255.0
    l, b, h = im.shape
    radius = 7

    # GETTING DARK CHANNEL
    dark = np.min(im, axis=2)
    dark = np.minimum.reduce([np.roll(dark, i, axis=(0, 1)) for i in range(-radius, radius+1)], axis=0)

    # GETTING LIGHT
    num = int(l * b * 0.001)
    flat_dark = dark.flatten()
    threshold = np.partition(flat_dark, -num)[-num] # using kth minimum element algorithm rather than sorting algorithm so as time changes from O(n^2) to O(n.logn)
    tmp = im[dark >= threshold]
    light = np.mean(tmp, axis=0) #estimation of airlight by finding pixels of high intensity.

    # GIVING AIR LIGHT
    airlight = 1.0 - 0.7 * (im.min(axis=2) / light.max()) #the 0.7 value in this light is to introduce a little haze so as to not completely remove all the haze and retain some naturality of the dehazed image. You can also give the value of 0.7 as 0.95 for daytime images to get a more visually appeasing video.

    # GUIDED FILTER : We use a filter for an image because the result of the above operations caused pixelation of images thereby losing the edges of the images. Initially we used a bilateral filter but its time is O(n^2). So we rather used a guided filter in our code which runs on O(1)
    img = (im * 255).astype(np.uint8)[:, :, (2, 1, 0)]
    guided = gf(img, (airlight * 255).astype(np.uint8), 60, 0.001) / 255.0

    t0 = 0.1
    guided[guided < t0] = t0
    t = guided[:, :, np.newaxis].repeat(3, axis=2)
    t = np.clip(t, 0.1, 255)
    #RESTORING THE ORIGINAL TRANSMISSION OF THE VIDEO
    light = np.clip(light, 0, 255)
    dst = (im - light) / t + light
    dst = np.clip(dst * 255, 0, 255)
    end=time.time()
    print('Total Time:',str(end-start),'s')
    return cv2.cvtColor(dst[:, :, (2, 1, 0)].astype(np.uint8), cv2.COLOR_BGR2RGB)



model_path = os.path.join('.','best.pt')
model = YOLO(model_path)  
c=0
threshold = 0.75

def human(frame):
    # c1=0
    results = model(frame)[0]
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            print("x1,y1,x2,y2"+str(x1)+" "+str(y1)+" "+str(x2)+" "+str(y2))
            if(x1<X1 and x2<X1):
                left()
            if(x1>X2 and x2>X2):
                right()

            frame=cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
                # frame=cv2.putText(frame,'HUMAN', (int(x1), int(y1 - 10)),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0), 3, cv2.LINE_AA)  
                # frame=cv2.putText(frame, str(score*100)[0:3], (int(x1), int(y1 - 10+50)),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                # c1+=1
                # global c
                # c=max(c,c1)  
                # cv2.putText(frame,'COUNT:'+str(c1),(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,), 3, cv2.LINE_AA)
    cv2.imshow('',cv2.resize(frame[:,X1:X2],(750,600)))
    #cv2.imshow('',frame)
                

def heat(frame):
    results = model(frame)[0]
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            frame=cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
            img[int(y1):int(y2),int(x1):int(x2)]+=1
            img_norm=(img-img.min())/(img.max()-img)
            img_norm=img_norm.astype('uint8')
            img_norm=cv2.GaussianBlur(img_norm,(5,5),0)
            heatmap=cv2.applyColorMap(img_norm,cv2.COLORMAP_JET)
            final=cv2.addWeighted(heatmap,0.5,frame,0.5,0)
            cv2.imshow('',cv2.resize(heatmap,(750,600)))
            #cv2.imshow("",heatmap)


# vid='tst1.mp4'
vid = 0
cap=cv2.VideoCapture(vid)
ret,frame=cap.read()
frame=rescale(frame)
h,w=frame.shape[0],frame.shape[1]
img=np.ones([int(h),int(w)],dtype='uint32')
o=0
while True:
    deh=dehaze(frame)
    # cv2.imshow('Original',frame)
    cv2.imshow('Dehazed',cv2.resize(deh,(750,600)))
    #cv2.imshow('Dehazed',deh)
    if cv2.waitKey(1)==ord("p") or o==1:
        human(deh)
        o=1
    if cv2.waitKey(1)==ord("h") or o==2:
        heat(deh)
        o=2
    ret,frame=cap.read()
    if cv2.waitKey(1)==ord ("q"):
        break
    if not ret:
        break
    frame=rescale(frame)
    
