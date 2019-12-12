import sys
import torch
import cv2
import argparse
import numpy as np
from utils import nms
import glob
import tinyyolo
from time import time
import os 
import os.path as osp
import shutil


class Detector:
    def __init__(self, resolution, confidence=0.5, nms_thresh=0.45, anchors=None):
        self.resolution = np.array([int(i) for i in resolution.split('x')])
        self.model = tinyyolo.TinyTinyYolov3()
        checkpoint = (torch.load('tinyyolo/personhead.pt'))
        load_dict = dict()
        for key in checkpoint['weights'].keys():
            new_key = key.replace('module.layers', 'layers')
            load_dict[new_key] = checkpoint['weights'][key]
        self.model.load_state_dict(load_dict, strict=True)
        self.model.cuda()
        self.model.eval()
        if anchors is not None:
            self.anchors = anchors
        else:
            self.anchors = [[(20,20), (50,100), (80,223)], [(194,193), (143,336), (316,356)]]#[[(194,193), (143,336), (316,356)], [(20,20), (50,100), (80,223)]]#[[(81, 82), (135, 169), (344, 319)], [(10, 14), (23, 27), (37, 58)]] #

        self.classes = ['person', 'head']
        # self.num_class = len(self.classes)
        # self.classes = ["bicycle", "motorbike"]
        self.num_class = len(self.classes)
        self.confidence = confidence
        self.nms_thresh = nms_thresh

    def detect(self, image):
        img_ori = image
        h, w = img_ori.shape[:2]
        img = torch.tensor(self.img_prepare(img_ori)[np.newaxis,:]).float().cuda()
        scaling_factor = min(self.resolution[1] * 1. / img_ori.shape[0], self.resolution[0] * 1. / img_ori.shape[1])
        with torch.no_grad():
            output = self.model(img)
        rects_prepare = self._rect_prepare(output)
        rects_post = self._get_rects(rects_prepare, scaling_factor, w, h)

        return rects_post


    def img_prepare(self, img):
        img_w, img_h = img.shape[1], img.shape[0]
        w, h = self.resolution
        new_w = int(img_w * min(w * 1. / img_w, h * 1. / img_h))
        new_h = int(img_h * min(w * 1. / img_w, h * 1. / img_h))
        resized_image = cv2.resize(img, (new_w, new_h),interpolation=cv2.INTER_CUBIC)

        canvas = np.full((h, w, 3), 127)

        canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

        return canvas.transpose([2, 0, 1]) / 255.0

    def _rect_prepare(self, output):
        prediction = None
        for index, value in enumerate(output):
            # anchor sizes are borrowed from YOLOv3 config file
            if index == 1:
            # if key == 'conv19':
            #     continue
                anchors = self.anchors[0]
            # elif index == 1:
            # elif key == 'conv17':
            #     continue
                # anchors = self.anchors[1]
            elif index == 0:
            # elif key == 'conv15':
            #     continue
                anchors = self.anchors[1]
            if prediction is None:
                prediction = self._predict_transform(value.cpu().numpy(), anchors)
            else:
                prediction = np.concatenate(
                    [prediction, self._predict_transform(value.cpu().numpy(), anchors)],
                    axis=1)
        conf_mask = np.expand_dims((prediction[:, :, 4] > self.confidence), axis=2)
        prediction = prediction * conf_mask
        prediction = prediction[np.nonzero(prediction[:, :, 4])]

        img_result = dict()
        for cls in self.classes:
            img_result[cls] = []
        max_conf_cls = np.argmax(prediction[:, 5:5 + self.num_class], 1)
        for rect, i, prob in zip(prediction[:, :4], max_conf_cls, prediction[:, 4]):
            img_result[self.classes[i]].append([rect[0], rect[1], rect[2], rect[3], prob])
        return img_result

    def _predict_transform(self, prediction, anchors):
        batch_size = prediction.shape[0]
        stride = self.resolution[1] // prediction.shape[2]
        grid_size = self.resolution // stride
        bbox_attrs = 5 + self.num_class
        num_anchors = len(anchors)

        prediction = np.reshape(prediction, (batch_size, bbox_attrs * num_anchors, grid_size[1] * grid_size[0]))
        prediction = np.swapaxes(prediction, 1, 2)
        prediction = np.reshape(prediction, (batch_size, grid_size[1] * grid_size[0] * num_anchors, bbox_attrs))
        anchors = [(a[0] * 1. / stride, a[1] * 1. / stride) for a in anchors]

        # Sigmoid the  centre_X, centre_Y. and object confidencce
        prediction[:, :, 0] = 1 / (1 + np.exp(-prediction[:, :, 0]))
        prediction[:, :, 1] = 1 / (1 + np.exp(-prediction[:, :, 1]))
        prediction[:, :, 4] = 1 / (1 + np.exp(-prediction[:, :, 4]))

        # Add the center offsets
        # grid = np.arange(grid_size)
        a, b = np.meshgrid(np.arange(grid_size[0]), np.arange(grid_size[1]))

        x_offset = a.reshape(-1, 1)
        y_offset = b.reshape(-1, 1)

        x_y_offset = np.concatenate((x_offset, y_offset), 1)
        x_y_offset = np.tile(x_y_offset, (1, num_anchors))
        x_y_offset = np.expand_dims(x_y_offset.reshape(-1, 2), axis=0)

        prediction[:, :, :2] += x_y_offset

        # log space transform height, width and box corner point x-y
        anchors = np.tile(anchors, (grid_size[0] * grid_size[1], 1))
        anchors = np.expand_dims(anchors, axis=0)

        prediction[:, :, 2:4] = np.exp(prediction[:, :, 2:4]) * anchors
        prediction[:, :, 5: 5 + self.num_class] = 1 / (1 + np.exp(-prediction[:, :, 5: 5 + self.num_class]))
        # prediction[:, :, 5: 5 + self.num_class] = softmax(prediction[:, :, 5: 5 + self.num_class])
        prediction[:, :, :4] *= stride

        box_corner = np.zeros(prediction.shape)
        box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
        box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
        box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
        box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
        prediction[:, :, :4] = box_corner[:, :, :4]

        return prediction

    def _get_rects(self, rects_pre, scaling_factor, w, h):
        rects_post = dict()
        for label, pieces in rects_pre.items():
            if len(pieces) == 0:
                rects_post[label] = np.zeros((0,5))
                continue
            pieces = np.array(pieces, dtype=np.float32)
            keeps = nms(pieces, self.nms_thresh, force_cpu=True)
            results = np.array([pieces[i] for i in keeps])
            results[:, :4] = self._coor_change(results[:, :4], scaling_factor, w, h)
            rects_post[label] = results
        return rects_post

    def _coor_change(self, rects, scaling_factor, w, h):
        rects[:, 0] -= (self.resolution[0] - scaling_factor * w) / 2.
        rects[:, 2] -= (self.resolution[0] - scaling_factor * w) / 2.
        rects[:, 1] -= (self.resolution[1] - scaling_factor * h) / 2.
        rects[:, 3] -= (self.resolution[1] - scaling_factor * h) / 2.

        rects[:, 0] = np.clip((rects[:, 0] / scaling_factor).astype(int), a_min=0, a_max=w)
        rects[:, 2] = np.clip((rects[:, 2] / scaling_factor).astype(int), a_min=0, a_max=w)
        rects[:, 1] = np.clip((rects[:, 1] / scaling_factor).astype(int), a_min=0, a_max=h)
        rects[:, 3] = np.clip((rects[:, 3] / scaling_factor).astype(int), a_min=0, a_max=h)
        return rects


    def draw(self, img, results, out=None):
        '''
        draw detection results
        '''
        color_dict = {'person': (255, 255, 0),
                      'head': (0, 255, 255)
                      }
        for label, pieces in results.items():
            color = color_dict[label]
            for piece in pieces:
                text = "{}:{:.2f}".format(label, piece[-1])
                pt1 = [int(piece[0]), int(piece[1])]
                pt2 = [int(piece[2]), int(piece[3])]
                cv2.rectangle(img, tuple(pt1), tuple(pt2), color, 3)
                t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                pt2 = pt1[0] + t_size[0] + 3, pt1[1] + t_size[1] + 4
                cv2.rectangle(img, tuple(pt1), tuple(pt2), color, -1)
                cv2.putText(img, text, (pt1[0], t_size[1] + 4 + pt1[1]), cv2.FONT_HERSHEY_PLAIN,
                            cv2.FONT_HERSHEY_PLAIN, 1, 1, 2)
        return img


    def save_model(self,name):
        self.model.save(name + '.caffemodel')

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    t0=time()
    parser = argparse.ArgumentParser('YOLOv3')
    parser.add_argument('--date', default='2019-11-03', type=str)
    parser.add_argument('--image', type=str, default='images/*.jpg')
    parser.add_argument('--resolution', type=str, default='416x416')
    args = parser.parse_args()
    print('load detector...')
    detector = Detector(args.resolution)
    print('load detector OK! use time:',round(time()-t0))
    
    def drop_rubbish_body(wd,cam):
        t1=time()
        srcdir=osp.join(wd,args.date,cam)
        dstdir=osp.join(wd,args.date+'-rubbish',cam)
        if not os.path.exists(dstdir):os.makedirs(dstdir)
        
        rubbn_score=0
        rubbn_size=0
        empty_pid=0
        imgindex=0
        t0=time()
        
        pids=os.listdir(srcdir)
        print('list pids use time:',time()-t0)
        for pid in pids:
            piddir=osp.join(srcdir,pid)
            imgnames=os.listdir(piddir)
            for imgname in imgnames:
                imgindex+=1
                imgdir=osp.join(piddir,imgname)
                if cam=='84E0F420A4EF8045' and imgindex<8000:
                    continue
                if imgindex==200 or imgindex%1000==0:
                    utime=round(time()-t0)
                    print(imgindex,imgdir,'  ',utime)
                    t0=time()
                    
                img_roi = cv2.imread(imgdir)
                
                # delete the too wide images
                imgh,imgw,imgc=img_roi.shape
                hwrate=imgh/imgw
                #if 1<=hwrate<=4.5:continue
                if hwrate<1 or hwrate>4.5:
                    #rubbn+=1
                    rubbn_size+=1
                    namejpg=imgdir.split('/')[-1]
                    name=namejpg.split('.')[0]
                    pid=imgdir.split('/')[-2]
                    dst_piddir=osp.join(dstdir,pid)

                    #try:
                    #print('img too wide, or too heigh!!')
                    dstimgdir1=osp.join(dst_piddir,namejpg)
                    if os.path.exists(dstimgdir1):
                        print('hw path already exists:',dstimgdir1)
                        os.remove(imgdir)
                        if len(os.listdir(piddir))==0:
                            os.rmdir(piddir)
                        continue
                    if not os.path.exists(dst_piddir):os.makedirs(dst_piddir)
                    shutil.move(imgdir,dst_piddir)
                    if len(os.listdir(piddir))==0:
                        empty_pid+=1
                        os.rmdir(piddir)# delete empty folder
                    #except:
                    #    print(imgindex,'move img error!!')
                    
                    continue
                
                results = detector.detect(img_roi)
                body_results=results['person']
                
                if body_results.shape[0] == 0:
                    top1_score=0
                else:
                    top1_score=body_results[0][4]
                
                if top1_score<0.7:
                    rubbn_score+=1
                    namejpg=imgdir.split('/')[-1]
                    name=namejpg.split('.')[0]
                    pid=imgdir.split('/')[-2]
                    dst_piddir=osp.join(dstdir,pid)
              
                    #try:
                    dstimgdir2=osp.join(dst_piddir,namejpg)
                    if os.path.exists(dstimgdir2):
                        print('sco path already exists:',dstimgdir2)
                        os.remove(imgdir)
                        if len(os.listdir(piddir))==0:
                            os.rmdir(piddir)
                        continue
                    if not os.path.exists(dst_piddir):os.makedirs(dst_piddir)
                    #print('imgdir',imgdir)
                    #print('dst_piddir',dst_piddir)
                    shutil.move(imgdir,dst_piddir)
                    if len(os.listdir(piddir))==0:
                        empty_pid+=1
                        os.rmdir(piddir)# delete empty folder
                    #except:
                    #    print(imgindex,'move img error!!')
                    
        return rubbn_score,rubbn_size,empty_pid
        
    
    def main():
        t0=time()
        rubb_nums=0
        rubbn_score_nums=0
        rubbn_size_nums=0
        empty_pids=0
        wd='/home/guanyonglai/data21/Reid/pedestrain_20191103/trajectory'
        wdcams='/home/guanyonglai/data21/Reid/pedestrain_20191103/trajectory/2019-11-03'
        #wdtmp='/home/guanyonglai/gyl/camstmp.txt'
        camsx=os.listdir(wdcams)
        cams=[cam.split('.')[0] for cam in camsx]
        #cams=cams[300:]
        #with open(wdtmp) as ff:
        #    cams=ff.readlines()
        #cams=[c.strip() for c in cams]
        #cams=['84E0F421284B8045','84E0F42128508045','84E0F42128528045','84E0F42128558045']
        #cams=['84E0F421A0B78004']
        for indcam,cam in enumerate(cams):
            if not (indcam>=0 and indcam<10):continue
            print(indcam,cam)
            t1=time()
            rubbn_score,rubbn_size,empty_pid=drop_rubbish_body(wd,cam)
            rubb_num=rubbn_score+rubbn_size
            rubb_nums+=rubb_num
            rubbn_score_nums+=rubbn_score
            rubbn_size_nums+=rubbn_size
            empty_pids+=empty_pid
            uset1=round(time()-t1)
            uset0=round(time()-t0)
            print('use time:{}/{} -- empty_pids:{}/{} -- rubbish num:{}/{} rubbscore:{}/{} rubbsize:{}/{}'.format(uset1,uset0,
                   empty_pid,empty_pids,rubb_num,rubb_nums,rubbn_score,rubbn_score_nums,rubbn_size,rubbn_size_nums))
        print('all use time:',time()-t0)
        print('rubb_nums:',rubb_nums)
        print('empty_pids:',empty_pids)
    
    main()