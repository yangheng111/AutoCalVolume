import os
import sys
import math
import numpy as np
import torch
from torch.utils.data import Dataset

import cv2

def calHist(image):
    b, g, r = cv2.split(image)
    b_hist= cv2.calcHist([b], [0], None, [256], [0.0,255.0]) 
    g_hist= cv2.calcHist([g], [0], None, [256], [0.0,255.0]) 
    r_hist= cv2.calcHist([r], [0], None, [256], [0.0,255.0]) 
    # print("ori:",b_hist)
    # print(max(b_hist),min(b_hist))
    norm_b_hist = (b_hist-min(b_hist))/(max(b_hist)-min(b_hist))
    norm_g_hist = (g_hist-min(g_hist))/(max(g_hist)-min(g_hist))
    norm_r_hist = (r_hist-min(r_hist))/(max(r_hist)-min(r_hist))
    # print("dat",norm_b_hist)
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


class Dataset(Dataset):
    def __init__(self,anntxt):
        ann = open(anntxt,'r')
        self.lines = ann.readlines()
        # self.f = open('test_feat.txt','w')
    def __getitem__(self, index):
        line = self.lines[index]

        imgPath = line.split()[0]
        ratio = float(line.split()[1])
        quality = float(line.split()[-1])/100

        # img = cv2.imread(imgPath)
        # energy = calEnergy(img)
        # print([ratio]+energy)
        # feat = [ratio]+energy
        feat = [float(line.split()[i+1]) for i in range(4)]
        # s = imgPath + ' '+str(ratio)+' '+str(energy[0])+' '+str(energy[1])+' '+str(energy[2])+' '+str(quality*100)+'\n'
        # print(s)
        # self.f.write(s)
        # self.f.flush()
        # feat = np.insert(calHist(img),0,[ratio],0)
        hist = torch.from_numpy(np.array(feat,dtype=float).reshape(1,-1)).float()
        # print(hist)
        label = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(np.array(quality,dtype=float)).float(),dim=0),dim=0)

        return hist, label


    def __len__(self):
        return len(self.lines)

if __name__ == "__main__":
    trainDataSet = Dataset('./test.txt')
    trainDataLoader = torch.utils.data.DataLoader(trainDataSet,batch_size=1,shuffle=True,num_workers=32,
                pin_memory=False,drop_last=True)

    for i,(x,y) in enumerate(trainDataLoader):
        print(i)