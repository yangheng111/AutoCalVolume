import os
import numpy as np
import time
import math
import cv2

import torch
import torch.nn.functional as F

from Net import Net
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
def calHist(image):
    b, g, r = cv2.split(image)

    b_hist= cv2.calcHist([b], [0], None, [256], [0.0,255.0]) 
    g_hist= cv2.calcHist([g], [0], None, [256], [0.0,255.0]) 
    r_hist= cv2.calcHist([r], [0], None, [256], [0.0,255.0]) 

    norm_b_hist = (b_hist-min(b_hist))/(max(b_hist)-min(b_hist))
    norm_g_hist = (g_hist-min(g_hist))/(max(g_hist)-min(g_hist))
    norm_r_hist = (r_hist-min(r_hist))/(max(r_hist)-min(r_hist))

    hist = np.concatenate((norm_b_hist, norm_g_hist,norm_r_hist), axis=0)

    return hist

def calEnergy(image):
    b,g,r = cv2.split(image)
    ress = []
    for a in [b,g,r]:
        tmp = []
        for i in range(256):
            tmp.append(0)
        val = 0
        k = 0
        res = 0
        img = np.array(a)
        for i in range(len(img)):
            for j in range(len(img[i])):
                val = img[i][j]
                tmp[val] = float(tmp[val] + 1)
                k =  float(k + 1)
        for i in range(len(tmp)):
            tmp[i] = float(tmp[i] / k)
        for i in range(len(tmp)):
            if(tmp[i] == 0):
                res = res
            else:
                res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
        ress.append(res)

    return ress

if __name__ == "__main__":
    net = Net(n_feature=4,n_hidden=1000,n_output=1).to(device)
    net.load_state_dict(torch.load('./model_epoch_190.pth'))

    lines= open('./test_feat.txt','r').readlines()
    error =0
    num_r = 0
    num_r_14 = 0
    # f= open('./file.txt','w')
    for line in lines:
        name = line.split()[0]
        ratio = line.split()[1]
        quality = float(line.split()[-1])

        imgPath = name
        # img = cv2.imread(imgPath)

        # energy = calEnergy(img)
        # print([ratio]+energy)
        # feat = [ratio]
        feat = [float(line.split()[i+1]) for i in range(4)]
        # hist = calHist(img)
        # feat = np.insert(calHist(img),0,[ratio],0)
        x = torch.from_numpy(np.array(feat,dtype=float).reshape(1,-1)).float().to(device)
        
        prediction=net(x)
        prediction = int(prediction.data.cpu().numpy()[0][0]*100)
        # if prediction > 100:
        #     prediction =100
        # elif prediction < 0:
        #     prediction = 0

        if (prediction - int(quality)) >= -10 and (prediction-int(quality))<=0:
            num_r = num_r+1

        elif (prediction - int(quality))>0:
            if abs(prediction - int(quality))/prediction <= 0.75:
                num_r_14 +=1
            else:
                error += 1
                print('imgPath :',imgPath)
                print('error:{},right:{},right1/4:{},all:{},predict:{},label:{}'.format(error,num_r,num_r_14,len(lines),prediction,quality))
        else:
            if abs(prediction-int(quality))/(100-prediction) <= 0.75:
                num_r_14 += 1
            else:
                error +=1
                print('imgPath :',imgPath)
                print('error:{},right:{},right1/4:{},all:{},predict:{},label:{}'.format(error,num_r,num_r_14,len(lines),prediction,quality))

    print('error:{},right:{},right1/4:{},all:{}'.format(error,num_r,num_r_14,len(lines)))
            