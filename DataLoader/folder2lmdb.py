# coding: utf-8
import os
import os.path as osp
import os, sys
from PIL import Image
import six
import string

import lmdb
import pickle
#import umsgpack
import tqdm
import pyarrow as pa
from os.path import basename
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets


from PIL import Image
from torch.utils.data import Dataset
 
#集成Dataset类
class MyDataset(Dataset):
    def __init__(self, txt_path):
        """
        tex_path : txt文本路径，该文本包含了图像的路径信息，以及标签信息
        """
        fh = open(txt_path, 'r')  #读取文件
        imgs = []  #用来存储路径与标签
        #一行一行的读取
        for line in fh:
            line = line.rstrip()  #这一行就是图像的路径，以及标签  
            
            words = line.split()
            label = int(words[-1])
            img_path = words[0]#line[0:len(line)-len(words[-1])-1]
            imgs.append((img_path, label))  #路径和标签添加到列表中
            self.imgs = imgs                        
    
    def __getitem__(self, index):
        fn, label = self.imgs[index]   #通过index索引返回一个图像路径fn 与 标签label
        img = raw_reader(fn)  #把图像转成RGB
        return img, label, fn              #这就返回一个样本
    
    def __len__(self):
        return len(self.imgs)          #返回长度，index就会自动的指导读取多少



class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


def read_txt(fname):
    map = {}
    with open(fname) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    for line in content:
        idx,label,imgpath = line.split(" ")#img, idx = line.split(" ")
        map[int(idx)]=[int(label),imgpath]
    return map


class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length = loads_pyarrow(txn.get(b'__len__'))
            # self.keys = umsgpack.unpackb(txn.get(b'__keys__'))
            self.keys = loads_pyarrow(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform
        map_path = db_path[:-5] + "_images_idx.txt"
        self.img2idx = read_txt(map_path)
        
    def __getitem__(self, index):
        img, label = None, None
        label = self.img2idx[index]
        env = self.env
        with env.begin(write=False) as txn:
            #print("key", self.keys[index].decode("ascii"))
            key=self.keys[index].decode("ascii")
            byteflow = txn.get(self.keys[index])

        unpacked = loads_pyarrow(byteflow)

        imgbuf = unpacked
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        
        img = Image.open(buf).convert('RGB')
        # img.save("img.jpg")
        if self.transform is not None:
            img = self.transform(img)
        im2arr = np.array(img)
        # print(im2arr.shape)

        return im2arr,label

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


def folder2lmdb(dpath, name="dataset",txt_path="list.txt", write_frequency=5000, num_workers=4):
    all_imgpath = []
    all_idxs = []
    print("Loading dataset from %s" % txt_path)
    dataset = MyDataset(txt_path)
    data_loader = DataLoader(dataset, num_workers=num_workers, collate_fn=lambda x: x)

    lmdb_path = os.path.join(dpath, "%s.lmdb" % name)
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)
    
    labels = []
    for idx, data in enumerate(data_loader):
        # print(type(data), data)
        # image, label = data[0]
        image, label, imgpath = data[0]
        print(type(image),label,imgpath)
        # print(image.shape)
        #imgpath = basename(imgpath)
        all_imgpath.append(imgpath)
        all_idxs.append(idx)
        labels.append([idx,label])
        txn.put(u'{}'.format(idx).encode('ascii'), dumps_pyarrow(image))
        # txn.put(u'{}'.format(imgpath).encode('ascii'), dumps_pyarrow(image))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()

    fout = open(dpath + "/" + name + "_images_idx.txt", "w")
    for img, idx in zip(all_imgpath, labels):
        fout.write("{} {} {}\n".format(idx[0],idx[1],img))
    fout.close()

if __name__ == '__main__':
    dpath = './lmdb'
    name = 'dfc_train'
    txt_path = '/data/xiaozihao/data/datalist/v1.3.1/train.txt'
    folder2lmdb(dpath, name, txt_path, write_frequency=5000)


'''
if __name__ == '__main__':
    transform_train = albumentations.Compose([
            RandomResizedCrop(224, 224),
            ShiftScaleRotate(p=0.3, scale_limit=0.25, border_mode=1, rotate_limit=25),
            HorizontalFlip(p=0.2),
            RandomBrightnessContrast(p=0.3, brightness_limit=0.25, contrast_limit=0.5),
            MotionBlur(p=.2),
            GaussNoise(p=.2),
            JpegCompression(p=.2, quality_lower=50),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],),
            ToTensor()
            ])

# Data loading 
    train_dataset = ImageFolderLMDB(db_path=args.data_dir, \
        transform=transform_train)
'''