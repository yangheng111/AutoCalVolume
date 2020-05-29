import os
import numpy as np
import time
import math
import json

import cv2
import torch

import requests

from Net import Net
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class LimitVolume():
    def __init__(self,):
        #初始网络的内部结构
        super(LimitVolume,self).__init__()
        self.net = self.loadModel('./model_epoch_190.pth')

    def loadModel(self, path):
        net = Net(n_feature=4,n_hidden=1000,n_output=1).to(device)
        net.load_state_dict(torch.load(path))
        print('网络初始化完成！')
        return net

    def resizeEnergyInput(self, image):
        h,w,c = image.shape
        if h < 224 or w < 224:
            return image 
        scale = float(224/max(h,w))
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return image

    def calEnergy(self, image):
        image = self.resizeEnergyInput(image)
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

    def DLpred(self,image,ori_volume,dst_volume):
        energy = self.calEnergy(image)
        ratio = dst_volume/ori_volume
        feat = [ratio]+energy
        x = torch.from_numpy(np.array(feat,dtype=float).reshape(1,-1)).float().to(device)
        
        prediction=self.net(x)
        prediction = int(prediction.data.cpu().numpy()[0][0]*100)
        print('DL predict:',prediction)

        return prediction

    def APITest(self,imgType,pred):
        data = {
            'url': '',
            'type': 1,
            'quality': 0,
            'limit_size': 0
        }
        data['url'] = 'https://cdn.geekdigging.com/opencv/opencv_header.png'
        data['type'] = imgType
        data['quality'] = pred

        headers = {'Content-Type': 'application/json'}
        URL = 'http://172.16.2.30/v1.0/compress/assign'
        res = requests.request("post",url=URL, headers=headers, data=json.dumps(data))
        print('压缩相应结果:',res.text)
        resJson = json.loads(res.text)

        return resJson["dest_size"]

    def inference(self,image,imgType,ori_volume,limit_volume):
        # first DL pred
        DL_pred = self.DLpred(image,ori_volume,limit_volume)
        # second api test
        temp_Volume = self.APITest(imgType,DL_pred)

        # compare
        minQ = 0
        maxQ = 100
        while not (temp_Volume <= limit_volume and temp_Volume/limit_volume >= 0.8):
            if temp_Volume > limit_volume:
                maxQ = DL_pred
                DL_pred -= 0.5*(maxQ-minQ)
            else:
                minQ = DL_pred
                DL_pred += 0.5*(maxQ-minQ)
            print(minQ,' ',maxQ)
            temp_Volume_1 = self.APITest(imgType,DL_pred)
            if temp_Volume == temp_Volume_1 :
                break
            temp_Volume = temp_Volume_1

        if temp_Volume > limit_volume :
            print('无法压缩到指定体积！')
            return -1
        else:
            print('最接近压缩质量和体积:',DL_pred,' ',temp_Volume)
            return 1