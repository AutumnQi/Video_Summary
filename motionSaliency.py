import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
import os
# from dataEntropy import calcEntropy
from scipy import stats

fgbg=cv2.createBackgroundSubtractorMOG2()
kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

stasaliency = cv2.saliency.StaticSaliencyFineGrained_create()

def motionSaliency(cap,f):
    lines=f.readlines()

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(length)
    saliency = None
    result=[]

    if cap.isOpened():
        for m in range(length):
            ret, frame = cap.read()
            frame=imutils.resize(frame,width=500)

            objSal=eval(lines[m])["objSal"]


            # if saliency is None:
            #     saliency = cv2.saliency.MotionSaliencyBinWangApr2014_create()
            #     saliency.setImagesize(frame.shape[1], frame.shape[0])
            #     saliency.init()
            #
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # (success, saliencyMap) = saliency.computeSaliency(gray)
            # saliencyMap = (saliencyMap * 255).astype("uint8")
            #
            #
            # motSal=np.sum(saliencyMap) / (saliencyMap.shape[0] * saliencyMap.shape[1] * 255)

            # cv2.imshow("motionMap", saliencyMap)
            staSal = staticSaliency(frame)
            fgSal = foreground(frame)

            # if cv2.waitKey(0) and 0xFF:
            #     # cv2.putText(frame,str(staSal*motSal),(100,100),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),2)
            #     # print(staSal*motSal)
            #     fgSal = foreground(frame)
            #     cv2.imshow("motionMap", saliencyMap)
            #     staSal = staticSaliency(frame)
            print(staSal,fgSal,objSal,'%.5f' %  float(staSal * objSal * fgSal))
            # cv2.imshow("res", frame)
            sal=round(float(staSal * objSal * fgSal),2)
            result.append(sal)
    result[0]=0
    return result


def staticSaliency(frame):

    (success, saliencyMap) = stasaliency.computeSaliency(frame)
    threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # cv2.imshow("staticMap", saliencyMap)
    # cv2.imshow("Thresh", threshMap)
    return np.sum(threshMap) / (threshMap.shape[0] * threshMap.shape[1] * 255)

def foreground(frame):
    fgmask=fgbg.apply(frame)
    fgmask=cv2.morphologyEx(fgmask,cv2.MORPH_OPEN,kernel)
    fgmask = cv2.threshold(fgmask.astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # cv2.imshow("fgmask",fgmask)
    return np.sum(fgmask) / (fgmask.shape[0] * fgmask.shape[1] * 255)


def findPeak(list,peak):
    res = 0
    for i in range(1,len(list)-1):
        if list[i]>list[i-1] and list[i]>list[i+1]:
            res+=1
            peak.append(i)
    return res

def findValley(list,valley):
    res = 0
    for i in range(1,len(list)-1):
        if list[i]<list[i-1] and list[i]<list[i+1]:
            res+=1
            valley.append(i)
    return res

video_path="./video/"
video_name="test"

f=open('./test_output/'+video_name+'.json','r')
cap = cv2.VideoCapture(video_path+video_name+".mp4")
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
res=motionSaliency(cap,f)


peak=[]
valley=[]
mean=np.mean(res,axis=0)
std=np.std(res,axis=0)
cv=std/mean
peak_num=findPeak(res,peak)
valley_num=findValley(res,valley)
peak_y=[0]*peak_num
# entropy=calcEntropy(res)
# print(entropy(res))
# print(np.mean(res,axis=0))
# print(nxindPeak(res,peak),peak)

plt.plot(list(range(length)), res , color='red',label='color')
plt.plot(peak,peak_y,"bo")
plt.title(video_name)
plt.hlines(mean,0,len(res),colors='green',label=mean)
plt.text(0,0.45,'mean='+str(round(mean,5))+' std='+str(round(std,5))+' cv='+str(round(cv,5))
         +' peak='+str(peak_num)+' valley='+str(valley_num))
plt.ylim((0,0.50))
plt.show()

cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
