import os
import sys
import json
import glob
import cv2
import threading
import argparse
from time import time as ti
import threadpool 
import datetime

t0=ti()
cam_index=0
def extract_traj(device_id):
    global cam_index,t0
    cam_index+=1
    save_idx = 0
    cut_time='16'
    date_list = ['2019-11-03']
    src_path = '/home/guanyonglai/data21/Reid/cxyt-2019-11-03'
    save_path = '/home/guanyonglai/data21/Reid/pedestrain_20191103/trajectory'
    t1=ti()
    
    for date in date_list:
        json_files = glob.glob(os.path.join(src_path, device_id, date,cut_time ,'*.json'))
        for idx, json_file in enumerate(json_files):
            if cut_time != json_file.split('/')[-1].split('_')[1].split('-')[0]:
                continue
            try:
                # time = os.path.splitext(os.path.basename(json_file))[0]
                if idx % 500 == 0:
                    print('processing :***{} {} {} {}/{}  cut_time:{}'.format(cam_index, device_id, date, idx, len(json_files),cut_time))
                info = json.load(open(json_file))
                img = cv2.imread(os.path.splitext(json_file)[0] + '.jpeg')
                for person in info['realTimeInfo']:
                    if person['body']['flag']==0:continue
                    height=person['imgSize'][0]
                    width=person['imgSize'][1]
                    xmin = int(person['body']['box'][0])
                    ymin = int(person['body']['box'][1])
                    xmax = int(person['body']['box'][2])
                    ymax = int(person['body']['box'][3])
                    
                    w=xmax-xmin
                    h=ymax-ymin
                    xmin = xmin-int(w/2.4) # 3.5 h w rate: h/w=2.16
                    ymin = ymin-int(h/30) # 2019-8-1 17:01:23
                    xmax = xmax+int(w/2.4) # 2.8 h w rate: h/w=1.99
                    ymax = ymax+int(h/25)
        
                    if xmin<0:xmin=0
                    if ymin<0:ymin=0
                    if xmax>width:xmax=width
                    if ymax>height:ymax=height
                    
                    person_id = person['personId']
                    person_path = os.path.join(save_path, date, device_id, str(person_id))
                    if not os.path.exists(person_path):
                        os.makedirs(person_path)
                    file_name = os.path.join(person_path, 
                        device_id + '_' + json_file.split('/')[-1].replace('.json','_')+str(person_id)+'.jpg')
                    if cv2.imwrite(file_name, img[ymin:ymax, xmin:xmax, :]):
                        # print(file_name)
                        save_idx += 1
            except KeyboardInterrupt as e:
                print(e)
                return
            except BaseException as e:
                print(type(e), e)
                print('save img error : ', json_file)
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    print('thread finish : {}   all use time : {}   now : {}'.format(device_id,round(ti()-t0,2),now))


def main():
    src_path = '/home/guanyonglai/data21/Reid/cxyt-2019-11-03'
    device_lst = os.listdir(src_path)
    device_lst = [d for d in device_lst if d.startswith('8')]
    device_lst = device_lst[45:]
    #device_lst=device_lst[0:1]

    start_time = time.time()
    pool = threadpool.ThreadPool(2) 
    requests = threadpool.makeRequests(extract_traj, device_lst) 
    [pool.putRequest(req) for req in requests] 
    pool.wait() 
        

if __name__ == '__main__':
    import time
    print('start time : ', time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime()))
    main()  # TODO: pass into arg in this way, shuold be list
    print('end time : ', time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime()))
    
    
    
