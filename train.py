import os
import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler

from Net import Net
from dataset import Dataset
import cv2

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
def train(DataLoader,Net,Loss,Optimizer,scheduler):
    print('------      启动训练      ------')
    for epoch in range(200):
        Net.train()
        scheduler.step()
        for i,(x,y) in enumerate(DataLoader):
            # Variable是将tensor封装了下，用于自动求导使用
            x, y = x.to(device), y.to(device)

            Optimizer.zero_grad()  #清除上一梯度

            prediction=net(x)
            loss=Loss(prediction,y)
            # print(loss)
            
            loss.backward() #反向传播计算梯度
            optimizer.step()  #应用梯度
        
            if i%10 == 0:
                print("Train Epoch:{} iter:{} mse loss:{}".format(epoch,i,loss))

        if epoch%10 ==0:
            print("Saving")
            savepath = './model_epoch_'+str(epoch)+'.pth'
            torch.save(Net.state_dict(), savepath)
        
 
if __name__=='__main__':

    trainDataSet = Dataset('./feat.txt')
    trainDataLoader = torch.utils.data.DataLoader(trainDataSet,batch_size=32,shuffle=True,num_workers=8,
                pin_memory=False,drop_last=True)
    
    print('------      搭建网络      ------')
    net = Net(n_feature=4,n_hidden=1000,n_output=1).to(device)

    # net.load_state_dict(torch.load('./model_epoch_100.pth'))
    print('网络结构为：',net)
    # loss_func=F.mse_loss
    loss_func=F.smooth_l1_loss
    optimizer=torch.optim.SGD(net.parameters(),lr=0.01, momentum=0.9)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20,50,70,90], gamma=0.1)

    train(trainDataLoader,net,loss_func,optimizer,scheduler)
