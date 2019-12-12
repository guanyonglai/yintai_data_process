#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import glob
import json
import argparse
import sys
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import torch
from torch import nn
from scipy.spatial.distance import cdist
from multiprocessing import JoinableQueue, Process, Manager
import threading
import os.path as osp
from torchvision import models
from torch.nn import functional as F
from scipy.spatial.distance import cdist
import pickle
from time import time
import threadpool
import math
from torch.utils import data
from torch.utils.data import DataLoader
import copy
import datetime


class Data_test(data.Dataset):
    def __init__(self, args, dataset_dir, transforms=None):
        # dataset_dir='/home/guanyonglai/data/pedestrain_20190607/trajectory/2019-06-07/'
        # guanlaoban
        self.dataset_dir=dataset_dir
        if os.path.exists(self.dataset_dir + '/Thumbs.db'):
            os.remove(self.dataset_dir + '/Thumbs.db')
        self.imgdirs=[] # all imgs in one cam
        cam=self.dataset_dir.split('/')[-1] # pids_dicts
        # if cam=='':cam=self.dataset_dir.split('/')[-2]
        with open(osp.join(args.pids_dicts,cam+'.txt')) as f:
            pids=f.readlines()
        pids = [p.strip() for p in pids]
        # pids=os.listdir(self.dataset_dir)
        for pid in pids:
            piddir=osp.join(self.dataset_dir, pid)
            if not os.path.exists(piddir):continue
            imgdir=os.listdir(piddir)
            imgdir=[osp.join(self.dataset_dir, pid, d) for d in imgdir]
            # imgdir=[i for i in imgdir if osp.getsize(i) > 0]
            self.imgdirs+=imgdir

        self.transform = transforms

    def __getitem__(self, index):
        imgdir = self.imgdirs[index]
        img = Image.open(imgdir).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, imgdir

    def __len__(self):
        return len(self.imgdirs)



class Data(object):
    def __init__(self,args, datadir, transforms=None):
        self.dataset_dir = datadir
        self.data_set = Data_test(args, self.dataset_dir, transforms)
        self.data_loader = DataLoader(dataset=self.data_set, batch_size=args.batch_size, num_workers=args.workers)


class ResNetReID(nn.Module):

    def __init__(self, num_classes=0, pretrained=True, num_features=2048, dropout=0.1):
        super(ResNetReID, self).__init__()

        self.num_features = num_features
        self.dropout = dropout
        self.num_classes = num_classes
        self.pretrained = pretrained

        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)

        self.base = models.resnet50(pretrained=pretrained)
        out_planes = self.base.fc.in_features

        self.feat = nn.Linear(out_planes, self.num_features, bias=False)
        self.feat_bn = nn.BatchNorm1d(self.num_features)
        self.relu = nn.ReLU(inplace=True)
        self.classifier_x2 = nn.Linear(self.num_features, self.num_classes)
        if not self.pretrained:
            self.reset_params()

    def forward(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        x1 = F.avg_pool2d(x, x.size()[2:])
        x1 = x1.view(x1.size(0), -1)
        x2 = self.feat(x1)
        x2 = self.feat_bn(x2)
        x2 = self.relu(x2)
        x2 = self.drop(x2)
        x2 = self.classifier_x2(x2)
        return x1, x2

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


class MGN_g(nn.Module):
    """
    same as github implement 这个是第三方改过的，和论文不一样，和我原来看的是一样的，，
    """
    def __init__(self, num_classes):
        super().__init__()
        res50 = models.resnet50(pretrained=True)
        res50.maxpool.ceil_mode = True
        self.backbone = nn.Sequential(
            res50.conv1,
            res50.bn1,
            res50.relu,
            res50.maxpool,
            res50.layer1,
            res50.layer2,
            res50.layer3[0]
            )
        res_conv4 = nn.Sequential(*res50.layer3[1:])
        res_conv5_g = res50.layer4
        # no down sampling in res_conv5 for part-2 and part-3
        res_conv5_p = copy.deepcopy(res50.layer4)
        res_conv5_p[0].conv2.stride = 1
        res_conv5_p[0].downsample[0].stride = 1

        self.res_layer45_G = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_conv5_g))
        self.res_layer45_P2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_conv5_p))
        self.res_layer45_P3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_conv5_p))

        self.maxpool_G_g = nn.MaxPool2d((12, 4))
        self.maxpool_P2_g = nn.MaxPool2d((24, 8))
        self.maxpool_P3_g = nn.MaxPool2d((24, 8))
        self.maxpool_P2_p = nn.MaxPool2d((12, 8))
        self.maxpool_P3_p = nn.MaxPool2d((8, 8))
        # self.maxpool_P2_p = nn.MaxPool2d((3, 8)) # 2019-7-30 09:27:18  be 8 part:12345 45678
        # self.maxpool_P3_p = nn.MaxPool2d((6, 8)) # 2019-7-30 09:27:18  be 4 part:12 23 34

        reduction = nn.Sequential(nn.Conv2d(2048, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU())
        self._init_reduction(reduction)
        self.reduction_G = copy.deepcopy(reduction)
        self.reduction_P2_g = copy.deepcopy(reduction)
        self.reduction_P2_p0 = copy.deepcopy(reduction)
        self.reduction_P2_p1 = copy.deepcopy(reduction)
        self.reduction_P3_g = copy.deepcopy(reduction)
        self.reduction_P3_p0 = copy.deepcopy(reduction)
        self.reduction_P3_p1 = copy.deepcopy(reduction)
        self.reduction_P3_p2 = copy.deepcopy(reduction)

        self.class_G = nn.Linear(256, num_classes) # 三个井号的地方不一样
        self.class_P2_g = nn.Linear(256, num_classes) #
        self.class_P2_p0 = nn.Linear(256, num_classes)
        self.class_P2_p1 = nn.Linear(256, num_classes)
        self.class_P3_g = nn.Linear(256, num_classes) #
        self.class_P3_p0 = nn.Linear(256, num_classes)
        self.class_P3_p1 = nn.Linear(256, num_classes)
        self.class_P3_p2 = nn.Linear(256, num_classes)

        self._init_fc(self.class_G)
        self._init_fc(self.class_P2_g)
        self._init_fc(self.class_P2_p0)

        self._init_fc(self.class_P2_p1)
        self._init_fc(self.class_P3_g)
        self._init_fc(self.class_P3_p0)
        self._init_fc(self.class_P3_p1)
        self._init_fc(self.class_P3_p2)

    def forward(self, x):
        x = self.backbone(x)
        p_G = self.res_layer45_G(x)
        p_P2 = self.res_layer45_P2(x)
        p_P3 = self.res_layer45_P3(x)

        z_G_g =  self.maxpool_G_g(p_G)  # for class loss
        z_P2_g = self.maxpool_P2_g(p_P2)  # for class loss
        z_P2_p = self.maxpool_P2_p(p_P2)
        z_P3_g = self.maxpool_P3_g(p_P3)  # for class loss
        z_P3_p = self.maxpool_P3_p(p_P3)

        f_G_g = self.reduction_G(z_G_g).squeeze(3).squeeze(2)  # for triple loss
        f_P2_g = self.reduction_P2_g(z_P2_g).squeeze(3).squeeze(2)  # for triple loss
        f_P2_p0 = self.reduction_P2_p0(z_P2_p[:, :, :1, :]).squeeze(3).squeeze(2)
        f_P2_p1 = self.reduction_P2_p1(z_P2_p[:, :, 1:2, :]).squeeze(3).squeeze(2)

        f_P3_g = self.reduction_P3_g(z_P3_g).squeeze(3).squeeze(2)  # for triple loss
        f_P3_p0 = self.reduction_P3_p0(z_P3_p[:, :, :1, :]).squeeze(3).squeeze(2)
        f_P3_p1 = self.reduction_P3_p1(z_P3_p[:, :, 1:2, :]).squeeze(3).squeeze(2)
        f_P3_p2 = self.reduction_P3_p2(z_P3_p[:, :, 2:3, :]).squeeze(3).squeeze(2)

        feature = torch.cat([f_G_g, f_P2_g, f_P2_p0, f_P2_p1, f_P3_g, f_P3_p0, f_P3_p1, f_P3_p2], dim=1)
        return feature
        
    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        #nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        #nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

class ModelForward():
    def __init__(self,args):
        #cudnn.benchmark = True
        #dataset = TransferData(args)
        self.model = MGN_g(num_classes=8395)
        model = nn.DataParallel(self.model).cuda()
        checkpoint = torch.load(args.resume)#load_checkpoint(osp.join(osp.dirname(osp.abspath(__file__)), 'logs_office', 'final30.pth'))
        #model.module.load_state_dict(checkpoint['state_dict'], strict=False)
        model.module.load_state_dict(checkpoint, strict=False)
        criterion = []
        optimizer = []
        #self.trainer = TransferTrainer(args, model, criterion, optimizer, dataset)

    def evaluate(self):
        print("Test with the original model trained on target domain (direct transfer):")
        t1 = time.time()
        self.trainer.evaluate()  # TODO: 35.17349624633789s
        print("TIME IS:", time.time() - t1)
        
def getfeat(inputs,model,transformer):
    # ff = torch.FloatTensor()
    # image = Image.open(imgfile).convert('RGB')
    # inputs = transformer(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model.model(inputs.cuda())
    ff = outputs.data.cpu()
    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
    ff = ff.div(fnorm.expand_as(ff))
    # features = torch.cat((features, ff), 0)
    return ff

def write_pid_feat(args):
    t1=time()
    model = ModelForward(args)
    model.model.eval()
    # transformer_list = [
    #     transforms.Resize(size=(380, 124), interpolation=3),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ]
    transformer_list = [
        transforms.Resize(size=(380, 124), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    transformer = transforms.Compose(transformer_list)
    print('load model use time:',time()-t1)

    wd = args.dataset_dir
    cams=os.listdir(wd)
    # cams=cams[:120]
    index=0
    for cam in cams:
        # if cam != '84E0F421A1478004':continue 
        index+=1
        #if index<76:continue
        print('processing',index,cam,'---------------')
 
        if not os.path.exists(args.dst_dir):
            os.makedirs(args.dst_dir)
        datadir=osp.join(args.dataset_dir,cam)
        dataset = Data(args,datadir,transformer)
        cam_feats_dicts={}
        cam_feats_dicts[cam]={}

        for batch, (inputs, imgdirs) in enumerate(dataset.data_loader):
            if batch%20==0:print(f'  batch:{batch}/{len(dataset.data_loader)}',cam)
            feats = getfeat(inputs, model, transformer)
            
            feats = feats.numpy()
            for feat, img_name in zip(feats, imgdirs):
                feat = feat.astype(np.float16)
                pid=img_name.split('/')[-2]
                name = img_name.split('/')[-1].split('.')[0]
                if pid not in cam_feats_dicts[cam].keys():cam_feats_dicts[cam][pid]={}
                cam_feats_dicts[cam][pid][name]=feat

        with open(osp.join(args.dst_dir, cam+'.txt'),'wb') as f:
            pickle.dump(cam_feats_dicts,f)


def main(args):
    t1=time()
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    print('now!',now)
    write_pid_feat(args)
    t2=time()
    print('time use:',t2-t1)
    
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,5,4,3,2,0'
    parser = argparse.ArgumentParser(description='merge argument', add_help=False)
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--resume', type=str, metavar='PATH',
                        default = '/home/guanyonglai/gyl/goods_project/MGN_s/experiment/TrainInfo_1206_1645-28×4+90.84/800_mgntmp_best.pth')
                        # default = '/home/guanyonglai/gyl/goods_project/MGN_tmp/experiment/TrainInfo_1003_0946/epoch450_best.pth')
    parser.add_argument('--dataset_dir', type=str,
                        default = '/home/guanyonglai/data21/Reid/pedestrain_20191103/trajectory/2019-11-03/')
    parser.add_argument('--dst_dir', type=str,
                        default='/home/guanyonglai/data21/Reid/pedestrain_20191103/trajectory/2019-11-03-feats-dicts-mgn')
    parser.add_argument('--pids_dicts', type=str,
                        default='/home/guanyonglai/data21/Reid/pedestrain_20191103/trajectory/2019-11-03-pids-dicts/')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    main(args)


