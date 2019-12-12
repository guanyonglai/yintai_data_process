import os
import sys
import json
import glob
import pickle
import numpy as np
import os.path as osp
from scipy.spatial.distance import cdist
import shutil
import copy
import cv2
from time import time


def pid_find_keys(pid, dict1_):
    '''
    pid is a value in dict1, in this function, we should
    find all key whole include the value 'pid',exp:
       dict1:
           key1:[pid1,pid8...]
           key2:[pid8,pid2...]
           key3:[pid4,pid5...]
        if our input is pid8,the output is [key1,key2],
        because both key1 and key2 include pid8
    '''
    same_keys=[]
    for k_, pids_ in dict1_.items():
        if pid in pids_:
            same_keys.append(k_)
    return same_keys

def merge_intersection_of_dict(dict1, order=False):
    '''
    the dict1 has many repeated values, we should merge the
    keys who has the same value,exp:
       dict1:
           key1:[pid1,pid8...]
           key2:[pid8,pid2...]
           key3:[pid4,pid5...]
           key4:[pid2,pid6...]
       if our input is dict1,the output is
       merged_dict:
           key1:[pid1,pid2,pid6,pid8...] #无意中做了这个和牛逼的操作
           key3:[pid4,pid5...]  
    order: order=True use to keep the original order
           order=False means the key of dict grow from 0
    '''    
    invalid_keys=[]
    merged_dict={}
    ind=-1
    for k,pids in dict1.items():
        if k in invalid_keys:
            continue
        else:
            ind+=1
            if ind % 1 == 0: print('======',ind, k)
            merged=dict1[k] # 但是不用深拷贝会改动原文件，(无意中做了一个很牛逼的操作)
#            merged=copy.deepcopy(dict1[k]) # 这里用深拷贝达不到需要的输出
            kk=[k]
            invalid_keys.append(k)
            for indp,pid in enumerate(pids):
                if indp%100==0:print(indp)
                same_keys=pid_find_keys(pid,dict1)
                if len(same_keys)==0:
                    continue
                else:
                    for keys in same_keys:
                        if keys in kk:
                            continue
                        else:
                            kk.append(keys)
                            merged+=dict1[keys]
                            invalid_keys.append(keys)
            merged=list(set(merged))
            if order==True:
                merged_dict.update({k:merged})
            else:
                merged_dict.update({str(ind):merged})
    return merged_dict

def merge_intersection_of_key(dict1, order=False):
    """
    其实这里和merge_intersection_of_dict函数的功能类似，只不过这里是
    把value有交集的key结合起来，上面是把value结合起来
       dict1:
           key1:[pid1,pid8...]
           key2:[pid8,pid2...]
           key3:[pid4,pid5...]
           key4:[pid2,pid6...]
       if our input is dict1,the output is
       merged_dict:
           0:[key1,key2,key4...] #无意中做了这个和牛逼的操作
           1:[key3...]  
    order: order=True use to keep the original order
    """
    newdict = {}
    megdict = merge_intersection_of_dict(dict1)
    ind = -1
    for k_, vmeg in megdict.items():
        ind += 1
        megk = []
        for k, v in dict1.items():
            if set(vmeg).intersection(set(v)):
                megk.append(k)
        if order==True:
            newdict.update({k:megk})
        else:
            newdict.update({str(ind):megk})
    return newdict
    
         
# ================================================a new start================================================
            
def sort_original_label_file(srcdir):
    """
    01
    对高朋给的原始文件按顺序排好，
    因为我把几份原始标注文件挪到一块了，所以ID会有重复，现在重新排ID 2019-12-2 14:11:47
    """
    dstdir = 'Z:/guanyonglai/yintai/new_annotation1001/1001_10_annotation_sorted.json'
    with open(srcdir) as f:
        orifile = json.load(f)
    orifile = orifile['RECORDS']
    output = []
    indxID = 50000 # indxID作为我们重新排序用的ID,从indxID开始计数
    for adict in orifile:
        indxID += 1
        new_adict = {}
        for key in adict.keys():
            if key == 'data_id':
                new_adict.update({key: str(indxID)})
            else:
                new_adict.update({key: adict[key]})
        output.append(new_adict)
#    with open(dstdir, 'w') as f:
#        json.dump(output, f, indent = 0)
    return output
        
def tidy_original_label_file(srcdir):
    """
    02
    对原始标注文件进行整理，统一轨迹长度，统一把路径改为相对路径
    2019-12-2 15:56:55
    """
    dstdir = 'Z:/guanyonglai/yintai/new_annotation1001/ZZ_Label_Results_1001_14/1001_14_annotation_tidy.json'
    print('sort_original_label_file ...')
    orifile = sort_original_label_file(srcdir)
    print('ok!')
    output = []
    
    for adict in orifile:
        new_adict = {}
        
        for key in adict.keys():
            if key == 'same_folder':
                same_folder = eval(adict[key])
                for traj in same_folder: assert len(traj.split('/'))==2
                new_adict.update({key: str(same_folder)}) #保持原有格式
                
            elif key == 'need_delete_picture':
                ndp = adict[key]
                new_ndp = []
                if ndp=='null' or ndp==None or ndp=='[]' or len(str(ndp)) < 5:
                    new_ndp = ndp
                else:
                    ndp = eval(ndp)
                    for p in ndp:
                        lp = len(p.split('/'))
                        if lp == 3:
                            pass
                        elif lp == 11:
                            p = '/'.join(p.split('/')[-3:])
                        else:
                            raise ValueError(f'len of path is not 3 or 11:{p}, is {lp}')
                        new_ndp.append(p)
                new_adict.update({key: str(new_ndp)})
                
            elif key == 'need_delete_folder':
                ndfo = adict[key]
                new_ndfo = []
                if ndfo=='null' or ndfo==None or ndfo=='[]' or len(str(ndfo)) < 5:
                    new_ndfo = ndfo
                else:
                    ndfo = eval(ndfo)
                    for fo in ndfo:
                        lfo = len(fo.split('/'))
                        if lfo == 2:
                            pass
                        elif lfo == 10:
                            fo = '/'.join(fo.split('/')[-2:])
                        else:
                            raise ValueError(f'len of path is not 2 or 10:{p}, is {lfo}')
                        new_ndfo.append(fo)
                new_adict.update({key: str(new_ndfo)})
                
            else:new_adict.update({key: adict[key]})
        output.append(new_adict)
        
    with open(dstdir, 'w') as f:
        json.dump(output, f, indent = 0)
    return output
    
        
        
def process_original_file(srcdir):
    """
    03
    对于最开始的标注文件，先不着急进行任何合并，先对原文件中相同轨迹、需要删除的轨迹、需要删除的图片进行整理，
    保持原有顺序不变，整理出我们需要的轨迹和图片信息
    2019-12-2 15:36:27
    """
    t0 = time()
    dstdir = 'Z:/guanyonglai/yintai/new_annotation1001/1001_10_annotation_prossed.json'
    my_img_dir='Y:/data/Reid/pedestrain_20191001_10/trajectory/2019-10-01' # 存储图片的路径
    with open(srcdir) as f:
        orifile = json.load(f)
    # orifile = tidy_original_label_file(srcdir)
    output = []
    
    for ind, adict in enumerate(orifile):
        if ind > 100: break
        if ind % 10 == 0: print('processing', ind, '--all_use_time:', round(time() - t0, 2))
        new_adict = {}
        data_id = adict['data_id']
        same_folder = adict['same_folder']
        need_delete_picture = adict['need_delete_picture'] # 需要删除的脏图
        ndfo = adict['need_delete_folder'] #标错的需要删除的轨迹
        same_folder = eval(same_folder)
        need_delete_picture = eval(need_delete_picture)
        
        # 清理轨迹
        if ndfo!='null' and ndfo!=None and ndfo!='[]' and len(str(ndfo)) > 5:
            ndfo = eval(ndfo)
            same_folder_set = list(set(same_folder) - set(ndfo))
            same_folder_set.sort(key = same_folder.index)
            same_folder = same_folder_set
            
        # 清理图片
        ori_imgs = []
        for traj in same_folder:
            trajdir = osp.join(my_img_dir, traj)
            imgs = os.listdir(trajdir)
            imgdirs = [osp.join(traj, img).replace('\\', '/') for img in imgs]
            ori_imgs += imgdirs

        if len(ori_imgs) > 0 and len(need_delete_picture)> 0:
            assert len(ori_imgs[0].split('/')) == len(need_delete_picture[0].split('/')) # 确保格式一致做set才有意义
        flt_imgs = list(set(ori_imgs) - set(need_delete_picture))
        flt_imgs.sort(key = ori_imgs.index)
            
        new_adict.update({'data_id': data_id})
        new_adict.update({'same_folder': same_folder})
        new_adict.update({'valid_imgs': flt_imgs})
        output.append(new_adict)
        
    with open(dstdir, 'w') as f:
        json.dump(output, f, indent = 0)

def merge_webpage(srcdir):
    '''
    04
    标注的时候是同一个ID分为几个页面标注的，而结果是每个页面都存一个标注结果，所以我们要把一个ID的多个页面的结果合并。
    合并依据：同一ID分多个页面标注时，页面左侧都是该ID的图片，即same_folder的第一个轨迹.
    如合并前的轨迹：
           id1:[traj1,traj8...]
           id2:[traj1,trajd2...]
           id3:[traj4,traj2...] # 第二个轨迹一样都是traj2，不关注不合并，只关注第一个否一样，
    则输出轨迹为：
           id1:[traj1,traj2,traj8...]
           id3:[traj4,traj2...]
    这次是二期的标注，和一期不同，我们只关注key对应轨迹的第一个pid是否相同，不管后面的是否相同，
    '''    
    dstdir = 'Z:/guanyonglai/yintai/new_annotation1001/ZZ_Label_Results_1001_14/meged_ids_14.json'
    with open(srcdir) as f:
        orifile = json.load(f)
    same_trajs_dict = {} # 提取原文件中的相同轨迹信息
    for adict in orifile:
        key = adict['data_id']
        trajs = adict['same_folder']
        same_trajs_dict.update({key: trajs})
    
    onepids = [] # 每个key只保留第一个pid
    onepids_dict = {}
    merged_ids = [] # 把第一个轨迹相同的ID合并起来
    ind = 0

    for k, trajs in same_trajs_dict.items():
        trajs = eval(trajs)
        if len(trajs) < 1: continue
        onepids.append(trajs[0])
        onepids_dict.update({k: trajs[0]})
    onepids_set = list(set(onepids))
    onepids_set.sort(key = onepids.index)
    print(f'ori labeled pages:{len(onepids)}   ori labeled ID:{len(onepids_set)}')
    
    for pid in onepids_set:
        ind += 1
        sameids = pid_find_keys(pid, onepids_dict)
        merged_ids.append(sameids)
    with open(dstdir, 'w') as f:
        json.dump(merged_ids, f, indent = 0)
    return merged_ids

def make_traj_and_img_dict(srcdir):
    """
    05
    srcdir 是004处理后的文件，
    通过我们合并后的ID，追溯到原文件，整理好按ID分类的轨迹、去除脏图后的图片
    2019-12-2 15:16:52
    """
    megid_dir = 'Z:/guanyonglai/yintai/new_annotation1001/ZZ_Label_Results_1001_14/meged_ids_14.json'
    dstdir_traj = 'Z:/guanyonglai/yintai/new_annotation1001/ZZ_Label_Results_1001_14/meged_trajs.json'
    dstdir_imgs = 'Z:/guanyonglai/yintai/new_annotation1001/ZZ_Label_Results_1001_14/meged_imgs.json'
    with open(srcdir) as f:
        orifile = json.load(f)
    with open(megid_dir) as f:
        merged_ids = json.load(f)
#    merged_ids = merge_webpage(srcdir) #这里srcdir不要用004处理后的，要用最原始的文件，否则需要改代码
    print('len merged_ids:', len(merged_ids))
    all_meg_trajs = {} # 存放整理好的按ID分类的轨迹
    all_meg_imgs = {} # 存放整理好的去除脏图后轨迹的图片
    
    for ind, megid in enumerate(merged_ids):
        if ind % 100 == 0:print(ind, megid)
        meg_trajs = []
        meg_imgs = []
        prikey = megid[0]
        for key in megid:
            for adict in orifile:
                if adict['data_id'] == key:
                    meg_trajs += adict['same_folder']
                    meg_imgs += adict['valid_imgs']
        set_meg_trajs = list(set(meg_trajs))
        set_meg_trajs.sort(key = meg_trajs.index)
        set_meg_imgs = list(set(meg_imgs))
        set_meg_imgs.sort(key = meg_imgs.index)
        if len(set_meg_trajs) <=1 or len(set_meg_imgs) <= 1: continue
        all_meg_trajs.update({prikey: set_meg_trajs})
        all_meg_imgs.update({prikey: set_meg_imgs})
        
    with open(dstdir_traj, 'w') as f:
        json.dump(all_meg_trajs, f, indent = 0)
    with open(dstdir_imgs, 'w') as f:
        json.dump(all_meg_imgs, f, indent = 0)
        
        
def deep_merge_logic():
    """
    对于多个ID同时包含同一个轨迹的情况，我们已经让兼职同学去检查了是否真的需要合并两个ID，
    兼职同学把两个需要合并的ID写在一行，这里读取标注文件，把一行的ID进行imgs和trajs的合并，
    这里还只是逻辑层面的合并，并没有进行算法比较相似度的合并，
    2019-12-4 13:55:06
    """
    trajdir = 'Z:/guanyonglai/yintai/new_annotation1001/ZZ_Label_Results_1001_14/meged_imgs.json'
    deepmegdir = 'Z:/guanyonglai/yintai/new_annotation1001/ZZZ_deep_merged/deep_merge_info_14.txt'
    dstwd = 'Z:/guanyonglai/yintai/new_annotation1001/ZZZ_deep_merged/deep_merge_imgs_14.json'

    with open(deepmegdir) as f:
        dmg_file = f.readlines()
    dmg_file = [df.strip().split() for df in dmg_file]
    print(f'len of deep meged files:{len(dmg_file)}')
    
    with open(trajdir) as f:
        trajs = json.load(f)
    all_megid_num = 0
    nokey = []
    all_meg_file = {}
    for ind, dfs in enumerate(dmg_file):
        # dfs 是一行需要结合的ID
        if ind % 10 == 0: print(ind, dfs)
        all_megid_num += len(dfs)
        meg_file = []
        prikey = dfs[0].lstrip('0')
        for d in dfs:
            d = d.lstrip('0')
            if d not in trajs.keys():
                nokey.append(d)
                continue
            meg_file += trajs[d]
        meg_file_set = list(set(meg_file))
        meg_file_set.sort(key = meg_file.index)
        all_meg_file.update({prikey:meg_file_set})
    
    print('no keys', nokey)
    print('no keys num', len(nokey))
    print('all_megid_num:', all_megid_num)
    if len(nokey) != 0:
        raise ValueError(f'find no keys in dict: {nokey}')
    with open(dstwd, 'w') as f:
        json.dump(all_meg_file, f, indent = 0)   


def gen_meg_imgs_to_check():
    """
    通过上面的函数check_similar_ids的结果生成我们需要标注的合成图
    """    
    my_img_dir='Y:/data/Reid/pedestrain_20191001_14/trajectory/2019-10-01'
    srcdir = 'Z:/guanyonglai/yintai/new_annotation1001/ZZZZZ_1208/all_sameID_dict_filt_hash2_14.json'
    topkdir = 'Z:/guanyonglai/yintai/new_annotation1001/ZZZZZ_1208/all_sameID_dict_filt_hash2_topk_similar_new_14.json'
    dstwd = 'Z:/guanyonglai/yintai/new_annotation1001/ZZZZZ_1208/Check_IDs_1001_14/Check_IDs_011'
    
    with open(srcdir) as f:
        hash2 = json.load(f)
    with open(topkdir) as f:
        topkfile = json.load(f)
    if not osp.exists(dstwd): os.makedirs(dstwd)
    
    ind = 0
    for id_qdir, id_gdirs in topkfile.items():
        imgdirs = []
        idtxt = []
        ind += 1
        if ind <= 10080 or ind > 15000: continue
#        if ind > 10: break
        qid, qdir = id_qdir.split('+')
        idtxt.append(qid)
        idtxt.append(qid)
        qimg2 = hash2[qid]
        imgdirs += qimg2
        for id_gdir in id_gdirs:
            gid, gdir = id_gdir.split('+')
            idtxt.append(gid)
            idtxt.append(gid)
            gimg2 = hash2[gid]
            imgdirs += gimg2
        imgdirs = [osp.join(my_img_dir, imgdir).replace('\\', '/') for imgdir in imgdirs]
        
        suffend = '_' + '_'.join(qdir.split('_')[-2:])
        
        name = qid.zfill(6) + suffend + '_similar'
        print(ind, len(imgdirs), len(idtxt))
        UnitImg = UnitShowImg(imgdirs, idtxt)
        linecolor = (29,145,255)
        linethickness = 5
        cv2.line(UnitImg, (0, 300), (1212, 300), linecolor, linethickness, 8)
        cv2.line(UnitImg, (404, 0), (404, 600), linecolor, linethickness, 8)
        cv2.line(UnitImg, (808, 0), (808, 600), linecolor, linethickness, 8)
        cv2.imwrite(osp.join(dstwd, name + '.jpg'), UnitImg)
        
    
def Find_similar_imgs():
    '''
       每个ID取几张图片，查找该图片的最相似topk图片，看最相似的k张图片和查询图片是否为同一人，
       以此决定是否真的需要合并（由兼职同学进行标注确认）
    '''
    wd='/home/guanyonglai/data8/yintai/new_annotation1001/ZZZZZ_1208'
    featwd='/home/guanyonglai/data/Reid/pedestrain_20191001_14/trajectory/feats_dicts_256_mgn.txt'
    allID_filedir=osp.join(wd,'all_sameID_dict_filt_hash2_14.json')
    dstdir=osp.join(wd,'all_sameID_dict_filt_hash2_topk_similar_14.json')
    All_Dict={}
    all_imgs_dd=[]
    topk=5 # for each img,find topk most similar imgs
    all_imgs_dd_feats=[] # dd: drop_duplicate
    t0=time()
    
    # load the feats dict
    t0=time()
    print('loading feats_dict ...')
    with open(featwd,'rb') as fd:
        feats_dict=pickle.load(fd)
    print('load feats dict use time:',time()-t0)

    with open(allID_filedir) as f:
        all_sameID_dict_dd=json.load(f)

    for idk,imgs in all_sameID_dict_dd.items():
        all_imgs_dd+=imgs
    print('all_imgs_dd:',len(all_imgs_dd))
    t1=time()            
       
    # key and value reverse
    all_sameID_dict_dd_reverse={}
    for personid,imgs in all_sameID_dict_dd.items():
        for img in imgs:
            all_sameID_dict_dd_reverse.update({img:personid})
            
    # extract all feats
    for ind,img in enumerate(all_imgs_dd):
        if ind%10000==0:print(ind,'extracting feat...')
        cam,pid,namejpg=img.split('/')
        name=namejpg.split('.')[0]
        feature=feats_dict[cam][pid][name]
        all_imgs_dd_feats.append(feature)
    print('len all_imgs_dd_feats:',len(all_imgs_dd_feats))
    all_imgs_dd_feats=torch.Tensor(all_imgs_dd_feats)
    
    t2=time()
    print('read feats use time:',t2-t1)

    # computer each img`s dist in all_imgs_dd_feats with each other
    pindex=0
    similar_img_dict={} # for each img in all_sameID_dict_dd, we should find it`s most similar imgs
    for personid,imgs in all_sameID_dict_dd.items():
        pindex+=1
        #if pindex>9375:break
        if pindex%10==0:print(f'index:{pindex}  personid:{[personid]}, time:{round((time()-t2),1)}, {imgs[0]}')
        has_pids=['/'.join(i.split('/')[:2]) for i in imgs]
        has_pids=list(set(has_pids)) # a string`s traj are in has_pids
        
        for ind, imgdir in enumerate(imgs):
#            if ind>0:break
            duplicated_pids=has_pids[:]
            cam,pid,namejpg=imgdir.split('/')
            name=namejpg.split('.')[0]
            feat1=feats_dict[cam][pid][name]
            topk_similiar_imgs=[]
            simi_indx=0 # for couent, we only take topk valid most similiar imgs, (vaild means not in same personid)
            rep_indx=0
            feat1=[feat1]
            feat1=torch.Tensor(feat1)
            dist=my_cdist(feat1,all_imgs_dd_feats)
            dist=dist[0]
            dist = np.array(dist)
            rank_ID = np.argsort(dist)
#            dist=dist[rank_ID]
            all_imgs_dd_ = np.array(all_imgs_dd)
            all_imgs_dd_ = all_imgs_dd_[rank_ID]
            
            # find the valid similar imgs 2019-9-7 19:50:40
            for ind_si, simi_img in enumerate(all_imgs_dd_):
                cam1,pid1,namejpg1 = simi_img.split('/')
                key_cp=cam1+'/'+pid1
                if key_cp in duplicated_pids:
                    rep_indx+=1
                    continue # simi_img and imgdir has same personid
                else:
                    simi_indx+=1
                    if simi_indx>topk:break
                    topk_similiar_imgs.append(simi_img)
                    # find: simi_img -> corresponding personid -> the pids(trajs) in corresponding personid
                    cor_personid=all_sameID_dict_dd_reverse[simi_img]
                    cor_pids=all_sameID_dict_dd[cor_personid]
                    cor_pids=['/'.join(i.split('/')[:2]) for i in cor_pids] # one personid has several trajs
                    cor_pids=list(set(cor_pids))
                    duplicated_pids+=cor_pids # has_pids: for make no topk similar imgs from different personid
                    duplicated_pids=list(set(duplicated_pids))
            similar_img_dict.update({imgdir:topk_similiar_imgs})
            
            if pindex%20==0:
                # set checkpoint, Prevent code interruption
                with open(osp.join(wd,'Find_similar_imgs_no_no_duplicated2_backup.json'),'w') as fwtmp:
                    json.dump(similar_img_dict,fwtmp,indent=True)

    with open(dstdir,'w') as f:
        json.dump(similar_img_dict,f,indent=True)
    print('Find_similar_imgs all use time:',time()-t0)
    
    
if __name__ == '__main__':
    '''
    本工程五个主要函数的作用：
       01 -> 02 -> 03 \
                          ===> 05        
                     04 /
    01,02,03 针对单独文件进行整理，04整理出合并后的key，再由03和04的结果通过05整理出合并后的文件 2019-12-3 09:33:42
    '''
    srcdir = 'Z:/guanyonglai/yintai/new_annotation1001/1001_10_annotation_sorted.json'
    make_traj_and_img_dict(srcdir)
    