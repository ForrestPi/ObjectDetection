'''

split a subcnn train val txt from image_0

'''
import os
import glob
import sys
import random

d = sys.argv[1]
print('split from: {}'.format(d))

all_files = glob.glob(os.path.join(d, '*.jpg'))
random.shuffle(all_files)

train_files = all_files[0: int(len(all_files) * 0.8)]
val_files = all_files[int(len(all_files) * 0.8):]

train_save_f = os.path.join('ImageSets_subcnn', os.path.basename(d), 'train.txt')
val_save_f = os.path.join('ImageSets_subcnn', os.path.basename(d), 'val.txt')

os.makedirs(os.path.dirname(train_save_f), exist_ok=True)

open(train_save_f, 'w').writelines([os.path.basename(i).split('.')[0] + '\n' for i in train_files])
open(val_save_f, 'w').writelines([os.path.basename(i).split('.')[0] + '\n' for i in val_files])