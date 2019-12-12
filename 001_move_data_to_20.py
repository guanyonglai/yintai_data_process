# -*- coding: utf-8 -*-
import json
from time import time
import os
import os.path as osp
import shutil


def main():
    srcdir='/data/nfs/data1001'
    dstdir='/data/nfs/Reid/cxyt-2019-10-01'
    date='2019-10-01'
    hour='10'
    cams=os.listdir(srcdir)
    cams=[c for c in cams if c.startswith('8')]
    print('len cams:',len(cams))
    
    for indc,cam in enumerate(cams):
        if indc>0:break
        print(indc,cam)
        srcimgdir=osp.join(srcdir,cam,date,hour)
        dstimgdir=osp.join(dstdir,cam,date,hour)
        if not os.path.exists(srcimgdir):continue
        #if not os.path.exists(dstimgdir):os.makedirs(dstimgdir)
        shutil.copytree(srcimgdir,dstimgdir)
        

if __name__ == '__main__':
    main()
    
    
    