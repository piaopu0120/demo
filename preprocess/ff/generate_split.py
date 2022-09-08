import json
import os
from glob import glob
import random
import readline
def plain_json():
    f_train = open('/raid/lpy/data/FaceForensics++/splits/train.json',"r")
    f_val = open('/raid/lpy/data/FaceForensics++/splits/val.json',"r")
    f_test = open('/raid/lpy/data/FaceForensics++/splits/test.json',"r")
    train_json = json.load(f_train)
    val_json = json.load(f_val)
    test_json =json.load(f_test)
    train_name = []
    val_name = []
    test_name = []
    for x,y in train_json:
        train_name.append(x)
        train_name.append(y)
    for x,y in val_json:
        val_name.append(x)
        val_name.append(y)
    for x,y in test_json:
        test_name.append(x)
        test_name.append(y)
    return train_name,val_name,test_name

def write_annotation(path,data_list,label):
    f = open(path,"w")
    for data in data_list:
        f.write(data+" "+label+'\n')
    f.close()

def generate_annotations(train_name,val_name,test_name):
    category = ['Deepfakes','Face2Face','FaceSwap','NeuralTextures','FaceShifter']
    label = '0'
    n_frames = 50
    train_list = []
    val_list = []
    test_list = []
    
    # for cate in category:
    if True:
    #     path = '/raid/lpy/data/FaceForensics++/'+cate+'/c23/faces_64/'
        path = '/raid/lpy/data/FaceForensics++/original_sequences/youtube/c23/faces_dcl/'
        for _,dirs,__ in os.walk(path):
            for dir in dirs:
                video_path = path+dir
                tmp = glob(video_path+'/*.png')
                tmp = sorted(random.sample(tmp,min(n_frames,len(tmp))))
                u = dir
                # u,v = dir.split('_')
                if u in train_name:
                    train_list.extend(tmp)
                elif u in val_name:
                    val_list.extend(tmp)
                elif u in test_name:
                    test_list.extend(tmp)
                else:
                    print('why?? '+dir)
            break
        print(len(test_list))
    random.shuffle(train_list)
    train_path = '/raid/lpy/data/FaceForensics++/annotations/ff_baseline/train_real.txt'
    val_path = '/raid/lpy/data/FaceForensics++/annotations/ff_baseline/val_real.txt'
    test_path = '/raid/lpy/data/FaceForensics++/annotations/ff_baseline/test_real.txt'
    write_annotation(train_path,train_list,label)
    write_annotation(val_path,val_list,label)
    write_annotation(test_path,test_list,label)

def shuffle():
    path = '/raid/lpy/data/FaceForensics++/annotations/ff_baseline/val.txt'
    with open(path) as f:
        tmp = f.readlines()
    random.shuffle(tmp)
    new_val_file_path = "/raid/lpy/data/FaceForensics++/annotations/ff_baseline/val_shuffle.txt"
    f_val = open(new_val_file_path,"w")
    f_val.writelines(tmp)
    f_val.close()

 

if __name__ == '__main__':
    # train_name,val_name,test_name = plain_json()
    # generate_annotations(train_name,val_name,test_name)
    shuffle()
