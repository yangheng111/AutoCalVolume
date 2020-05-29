# import the necessary packages
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
from skimage.measure import compare_mse
import argparse
import imutils
import cv2
import shutil
import os
Clines = open('./test.txt','r').readlines()
Dllines = open('./file.txt','r').readlines()

for i in range(len(Clines)):
    cline = Clines[i]
    name = cline.split()[0].split('/')[-1]
    cname = cline.split()[-1] + '___'+name
    dlline = Dllines[i]
    dlname = dlline.split()[-1]+'___'+dlline.split()[1]

    path = cline.split()[0]
    cpath = os.path.join('./test',cname)
    dlpath = os.path.join('./test',dlname)
 
    # load the two input images
    imageA = cv2.imread(cpath)
    imageB = cv2.imread(dlpath)
    
    if imageA is None or imageB is None:
        print("{}--{} is None".format(cname,dlname))
        continue
    # convert the images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    #反映人眼主观感受。一般取值范围：0-1.值越大，视频质量越好
    ssim= compare_ssim(grayA, grayB)
    if ssim < 0.99 and int(dlline.split()[-1])<int(cline.split()[-1]):
        print("{}--{}--SSIM: {}".format(cname,dlname,ssim))
        shutil.copyfile(path,os.path.join('ssimerro',name))
        shutil.copyfile(cpath,os.path.join('ssimerro','c___'+cname))
        shutil.copyfile(dlpath,os.path.join('ssimerro','dl___'+dlname))

    # #PSNR越高，压缩后失真越小。
    # psnr = compare_psnr(imageA,imageB)
    # print("PSNR: {}".format(psnr))

    # #误差，越小越好
    # mse = compare_mse(imageA,imageB)
    # print("MSE: {}".format(mse))