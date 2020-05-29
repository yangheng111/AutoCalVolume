import os
import random
import io


imgFolds = ['graph_20200110']
foldq = ['20','30','40','50','60','70','80','auto']


def calCompressRatio(oriPath,dstPath):
    ori = open(oriimgPath,'rb').read()
    ori_b = io.BytesIO(ori).read()
    ori_ram = len(ori_b)

    dst = open(dstPath,'rb').read()
    dst_b = io.BytesIO(dst).read()
    dst_ram = len(dst_b)

    return float(dst_ram)/float(ori_ram)

ftrian = open('train.txt','w')
ftest = open('test.txt','w')

for imgfold in imgFolds:
    for q in foldq:
        dstfold = imgfold+'_'+q
        # print(dstfold)
        files = os.listdir(dstfold)
        val = random.sample(files,int(0.2*(len(files))))

        for name in files:
            oriname = name.split('___')[-1]
            quality = name.split('___')[0]

            if quality == '0':
                continue

            oriimgPath = os.path.join(imgfold,oriname)
            dstimgPath = os.path.join(dstfold,name)

            ratio = calCompressRatio(oriimgPath,dstimgPath)

            s = oriimgPath + ' ' + str(ratio) + ' ' + quality + '\n'
            print(s)

            if name in val:
                ftest.write(s)
            else:
                ftrian.write(s)

ftrian.close()
ftest.close()



