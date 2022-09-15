from email.policy import default
from glob import glob
from multiprocessing import connection

from tqdm import tqdm
import pdb
import argparse
import torch
import cv2
import numpy as np
import os

import sys
sys.path.append("..")
from src.face_detect.src.retina_face_extractor import RetinaFaceExtract
from src.face_detect.src.config.config import init_config as RetinaFaceExtract_cfg
from src.dlib_face import ldmk_68_detecter

def face_bigger(height, width, detections, scale=1.7, minsize=None):
    x1 = detections[0]
    y1 = detections[1]
    x2 = detections[2]
    y2 = detections[3]
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    k_left = size_bb // 2 if (int(center_x - size_bb // 2) > 0) else center_x
    k_right = size_bb // 2 if (int(center_x + size_bb // 2) < width) else (width - center_x)
    k_top = size_bb // 2 if (int(center_y - size_bb // 2) > 0) else center_y
    k_bottom = size_bb // 2 if (int(center_y + size_bb // 2) < height) else (height - center_y)

    size_fn = min(k_left, k_right, k_top, k_bottom)
    return [center_x - size_fn, center_y - size_fn, center_x + size_fn, center_y + size_fn]

def facecrop(model_det,video_path,save_path,save_face,save_bbox,num_frames=10,scale = 1.7):

    capture = cv2.VideoCapture(video_path)

    face_list = []

    # 获取需要的帧id
    frame_count =int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idxs = np.linspace(0,frame_count-1,num_frames,endpoint=True,dtype=np.int64)

    # 创建bbox地址
    bbox_dir = save_path + 'bbox_'+str(num_frames)+'_'+str(scale)+'/'
    os.makedirs(bbox_dir,exist_ok=True)
    bbox_dict_path = bbox_dir+os.path.basename(video_path).replace('.mp4','.npy')

    if  os.path.isfile(bbox_dict_path):
        bbox_dict = np.load(bbox_dir+os.path.basename(video_path).replace('.mp4','.npy'),allow_pickle=True).item()
    else:
        bbox_dict = {}
    
    for frame_idx in range(frame_count):
        # read frame
        ret, frame_old = capture.read() # frame_old: np(480,640,3)
        if not ret:
            tqdm.write('Frame read {} Error! : {}'.format(frame_idx,os.path.basename(video_path)))
            break
        if frame_idx not in frame_idxs :
            continue
        # 如果需要生成新的bbox覆盖掉旧的请注释这一行
        if frame_idx in bbox_dict:
            continue

        height, width = frame_old.shape[:-1] # 除去最后一维 通道数
        frame = cv2.cvtColor(frame_old,cv2.COLOR_BGR2RGB) # 通道交换
        # extract face
        faces, ori_coordinates, coordinates, bbox, landms, scores_all = model_det.inference(frame)
        try:
            if len(faces)==0:
                tqdm.write('No faces in {}:{}'.format(frame_idx,os.path.basename(video_path)))
                continue
            size_max = -1
            x0,y0,x1,y1 = -1,-1,-1,-1
            for face_idx in range(len(faces)):
                # xx0,yy0,xx1,yy1=faces[face_idx]['bbox'] 
                xx0,yy0,xx1,yy1=bbox[face_idx]
                face_s=(xx1-xx0)*(yy1-yy0)
                if face_s > size_max:
                    size_max,x0,y0,x1,y1 = face_s,int(xx0),int(yy0),int(xx1),int(yy1)
            x0,y0,x1,y1 = face_bigger(height,width,[x0,y0,x1,y1],scale=scale)
            face = frame_old[y0:y1,x0:x1][:]
            bbox_dict[frame_idx] = (y0,y1,x0,x1)
        except Exception as e:
            print(f'error in {frame_idx}:{video_path}')
            print(e)
            continue
        
        # 保存frame
        # frame_dir=save_path+'frames_'+str(num_frames)+'_'+str(scale)+'/'+os.path.basename(video_path).replace('.mp4','/') 
        frame_dir=save_path+'frames_'+str(num_frames)+'/'+os.path.basename(video_path).replace('.mp4','/') 
        os.makedirs(frame_dir,exist_ok=True) # 递归创建目录，exist_ok=True创建的目录不报错
        frame_path=frame_dir+str(frame_idx).zfill(3)+'.png' # 多建一层文件夹存帧图片，zfill是指定长度，左补0
        if not os.path.isfile(frame_path):
            cv2.imwrite(frame_path,frame_old)

        # 保存face
        if save_face:
            face_dir=save_path+'faces_'+str(num_frames)+'_'+str(scale)+'/'+os.path.basename(video_path).replace('.mp4','/') 
            os.makedirs(face_dir,exist_ok=True)
            face_path = face_dir+str(frame_idx).zfill(3)+'.png' 
            if not os.path.isfile(face_path):
                cv2.imwrite(face_path,face)
        
    # 保存bbox
    # print(bbox_dict)
    if save_bbox:
        np.save(bbox_dict_path,bbox_dict)
        capture.release()
        return

def generate_frame(video_path,save_path,num_frames=10,scale=1.7):
    capture = cv2.VideoCapture(video_path)
    frame_count =int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idxs = np.linspace(0,frame_count-1,num_frames,endpoint=True,dtype=np.int)
    for frame_idx in range(frame_count):
        # read frame
        ret, frame_old = capture.read() # frame_old: np(480,640,3)
        if not ret:
            tqdm.write('Frame read {} Error! : {}'.format(frame_idx,os.path.basename(video_path)))
            break
        if frame_idx not in frame_idxs:
            continue
        frame_dir=save_path+'frames_'+str(num_frames)+'/'+os.path.basename(video_path).replace('.mp4','/') 
        os.makedirs(frame_dir,exist_ok=True) # 递归创建目录，exist_ok=True创建的目录不报错
        frame_path=frame_dir+str(frame_idx).zfill(3)+'.png' # 多建一层文件夹存帧图片，zfill是指定长度，左补0
        if not os.path.isfile(frame_path):
            cv2.imwrite(frame_path,frame_old)
    capture.release()
    return


def crop_face_from_bbox(target_path,num_frames=10,scale=1.7):
    # 可能需要按情况修改一下地址
    target_frame_dir = target_path + 'frames_'+str(num_frames)+'_'+str(scale)+'/'
    ori_bbox_dir = target_path + 'bbox_'+str(num_frames)+'_'+str(scale)+'/'
    target_face_dir = target_path + 'faces_'+str(num_frames)+'_'+str(scale)+'/'
    os.makedirs(target_face_dir,exist_ok=True)
    target_video_dirs = sorted(glob(target_frame_dir+'*/'))
    for target_video_dir in target_video_dirs:
        splits = target_video_dir.split('/')
        ori_bbox_path = ori_bbox_dir+splits[-2]+'.npy'
        bbox_dict = np.load(ori_bbox_path,allow_pickle=True).item()
        target_frame_list = sorted(glob(target_video_dir+'*.png'))
        target_face_dir_new = target_face_dir+splits[-2]+'/'
        os.makedirs(target_face_dir_new,exist_ok=True)
        for target_frame_path in target_frame_list:
            frame_idx = os.path.basename(target_frame_path).replace('.png','')
            if int(frame_idx) not in bbox_dict.keys():
                continue
            x0,x1,y0,y1 = bbox_dict[int(frame_idx)]
            target_img = cv2.imread(target_frame_path)
            target_face = target_img[x0:x1,y0:y1][:]
            target_face_path = target_face_dir_new+os.path.basename(target_frame_path)
            if not os.path.isfile(target_face_path):
                cv2.imwrite(target_face_path,target_face)

def get_path(args):
    if args.dataset=='Original':
        dataset_path = '/raid/lpy/data/FaceForensics++/original_sequences/youtube/{}/'.format(args.comp)
    elif args.dataset=='Deepfakes':
        dataset_path = '/raid/lpy/data/FaceForensics++/Deepfakes/{}/'.format(args.comp)
    elif args.dataset=='FaceShifter':
        dataset_path = '/raid/lpy/data/FaceForensics++/FaceShifter/{}/'.format(args.comp)
    elif args.dataset=='FaceSwap':
        dataset_path = '/raid/lpy/data/FaceForensics++/FaceSwap/{}/'.format(args.comp)
    elif args.dataset=='NeuralTextures':
        dataset_path = '/raid/lpy/data/FaceForensics++/NeuralTextures/{}/'.format(args.comp)
    elif args.dataset=='Face2Face':
        dataset_path = '/raid/lpy/data/FaceForensics++/Face2Face/{}/'.format(args.comp)
    else:
        raise NotImplementedError
    return dataset_path

if __name__ == '__main__':
    ####
    # 1.需要修改get_path()函数里的dataset地址
    # 2.修改args
    # 3.修改203行的dataset
    ####
    parser=argparse.ArgumentParser()
    # parser.add_argument('-d',dest='dataset',default='Original',choices=['Original','Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures','FaceShifter']) 
    # 需要在下面自己修改dataset
    parser.add_argument('--comp',choices=['raw','c23','c40','tmp'],default='c23')
    parser.add_argument('--num_frames',type=int,default=2)
    parser.add_argument('--scale',type=float,default=1.7)
    parser.add_argument('--gpu',type=int,default=0)

    parser.add_argument('--face',type=bool,default=False,help='Do you need crop faces?')
    parser.add_argument('--bbox',type=bool,default=True,help='Do you need save bbox?')

    args=parser.parse_args()
    device=torch.device('cuda')
    gpu_id = args.gpu
    
    model_det = RetinaFaceExtract(RetinaFaceExtract_cfg(), gpu_id=gpu_id)
    # 在这里修改需要的dataset
    #for args.dataset in ['Original','Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures','FaceShifter']:
    for args.dataset in ['Original']:
        dataset_path = get_path(args)
        video_path_list=sorted(glob(dataset_path+'videos/'+'*.mp4'))
        n_sample = len(video_path_list)
        print("There are {} videos in {}".format(n_sample,args.dataset))
        for i in tqdm(range(n_sample)):
            # 直接提取帧，但是不能跳过无人脸的，优点是速度快
            if not args.face and not args.bbox: 
                generate_frame(
                    video_path_list[i],
                    save_path=dataset_path,
                    num_frames=args.num_frames,
                    scale=args.scale
                )
            else:
                # 有提取人脸的步骤，不需要frame、bbox和face可以在里面注释
                facecrop(
                    model_det,
                    video_path_list[i],
                    save_path=dataset_path,
                    save_face=args.face,
                    save_bbox=args.bbox,
                    num_frames=args.num_frames,
                    scale=args.scale)
        