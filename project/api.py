import io
import numpy as np

from PIL import Image
import cv2

import requests

from utils import LimitVolume


# URL到图片，获取Mat图像和图像原始体积
def UrlToMat(url):
    image = requests.get(url).content
    image_b = io.BytesIO(image).read()
    volume = len(image_b)
    # print("{} byte\n{} kb\n{} Mb".format(volume, volume / 1e3, volume / 1e6))
    image = Image.open(io.BytesIO(image))
    image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)   

    return image,volume

if __name__ == "__main__":
    imgUrl = 'https://cdn.geekdigging.com/opencv/opencv_header.png'
    # https://st-gdx.dancf.com/gaodingx/0/uxms/design/20200417-144252-6102.jpg
    imgType = 1
    limit_volume = 20000

    #api
    Limit_Volume = LimitVolume()
    image,ori_volume = UrlToMat(imgUrl)
    status = Limit_Volume.inference(image,imgType,ori_volume,limit_volume)




