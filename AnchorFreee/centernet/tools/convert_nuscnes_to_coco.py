"""
Added by MANA AI authors.

We try to training centernet on nuScenes data
first we convert nuScenes to KITTI, and then converts
that data to coco format

NOTE: this scripts ONLY solves front camera by default since
every camera has different intrinsic

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import json
import numpy as np
import cv2
import os
from utils.ddd_utils import compute_box_3d, project_to_image, alpha2rot_y,\
   project_to_image_nuscenes
from utils.ddd_utils import draw_box_3d, unproject_2d_to_3d_nuscenes


DATA_PATH = '/media/jintain/wd/permanent/datasets/nuScenes/v1.0mini/kitti/object'
if not os.path.exists(DATA_PATH):
  print('{} not exist on your machine, you should change'
   'it to your KITTI root path.'.format(DATA_PATH))
  exit(0)
DEBUG = True


# VAL_PATH = DATA_PATH + 'training/label_val/'
SPLITS = ['subcnn'] 

'''
the calibration in nuScenes are actually fixed
CAM_FRONT camera intrinsic: [[1.26641720e+03 0.00000000e+00 8.16267020e+02]
 [0.00000000e+00 1.26641720e+03 4.91507066e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]

CAM_FRONT_LEFT camera intrinsic: [[1.27259795e+03 0.00000000e+00 8.26615493e+02]
 [0.00000000e+00 1.27259795e+03 4.79751654e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]

CAM_FRONT_RIGHT camera intrinsic: [[1.26084744e+03 0.00000000e+00 8.07968245e+02]
 [0.00000000e+00 1.26084744e+03 4.95334427e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]


'''

def _bbox_to_coco_bbox(bbox):
  return [(bbox[0]), (bbox[1]),
          (bbox[2] - bbox[0]), (bbox[3] - bbox[1])]

def read_clib(calib_path):
  f = open(calib_path, 'r')
  for i, line in enumerate(f):
    if i == 2:
      calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
      calib = calib.reshape(3, 4)
      return calib

cats = ['Pedestrian', 'Car', 'Cyclist', 'Van', 'Truck',  'Person_sitting',
        'Tram', 'Misc', 'DontCare']
cat_ids = {cat: i + 1 for i, cat in enumerate(cats)}
# cat_info = [{"name": "pedestrian", "id": 1}, {"name": "vehicle", "id": 2}]
cablib_front = np.array([[1.26641720e+03, 0.00000000e+00, 8.16267020e+02],
 [0.00000000e+00, 1.26641720e+03, 4.91507066e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

cat_info = []
for i, cat in enumerate(cats):
  cat_info.append({'name': cat, 'id': i + 1})

for SPLIT in SPLITS:
  image_set_path = os.path.join(DATA_PATH, 'ImageSets_{}/image_0/'.format(SPLIT))
  ann_dir = os.path.join(DATA_PATH, 'label_0/')

  splits = ['train', 'val']
  # splits = ['trainval', 'test']
  calib_type = {'train': 'training', 'val': 'training', 'trainval': 'training',
                'test': 'testing'}

  for split in splits:
    ret = {'images': [], 'annotations': [], "categories": cat_info}
    image_set = open(image_set_path + '{}.txt'.format(split), 'r')
    image_to_id = {}
    for line in image_set:
      if line[-1] == '\n':
        line = line[:-1]
      image_id = int(line)
      calib = cablib_front
      image_info = {'file_name': '{}.jpg'.format(line),
                    'id': int(image_id),
                    'calib': calib.tolist()}
      ret['images'].append(image_info)
      if split == 'test':
        continue
      ann_path = ann_dir + '{}.txt'.format(line)
      # if split == 'val':
      #   os.system('cp {} {}/'.format(ann_path, VAL_PATH))
      anns = open(ann_path, 'r')
      
      if DEBUG:
        img_path = os.path.join(DATA_PATH, 'image_0/' + image_info['file_name'])
        print('showing image: {}'.format(img_path))
        image = cv2.imread(img_path)

      for ann_ind, txt in enumerate(anns):
        tmp = txt[:-1].split(' ')
        cat_id = cat_ids[tmp[0]]
        truncated = int(float(tmp[1]))
        occluded = int(tmp[2])
        alpha = float(tmp[3])
        bbox = [float(tmp[4]), float(tmp[5]), float(tmp[6]), float(tmp[7])]
        dim = [float(tmp[8]), float(tmp[9]), float(tmp[10])]
        location = [float(tmp[11]), float(tmp[12]), float(tmp[13])]
        rotation_y = float(tmp[14])

        ann = {'image_id': image_id,
               'id': int(len(ret['annotations']) + 1),
               'category_id': cat_id,
               'dim': dim,
               'bbox': _bbox_to_coco_bbox(bbox),
               'depth': location[2],
               'alpha': alpha,
               'truncated': truncated,
               'occluded': occluded,
               'location': location,
               'rotation_y': rotation_y}
        ret['annotations'].append(ann)
        if DEBUG and tmp[0] != 'DontCare':
          box_3d = compute_box_3d(dim, location, rotation_y)
          box_2d = project_to_image_nuscenes(box_3d, calib)
          # print('box_2d', box_2d)
          image = draw_box_3d(image, box_2d)
          x = (bbox[0] + bbox[2]) / 2
          '''
          print('rot_y, alpha2rot_y, dlt', tmp[0], 
                rotation_y, alpha2rot_y(alpha, x, calib[0, 2], calib[0, 0]),
                np.cos(
                  rotation_y - alpha2rot_y(alpha, x, calib[0, 2], calib[0, 0])))
          '''
          depth = np.array([location[2]], dtype=np.float32)
          pt_2d = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                            dtype=np.float32)
          # pt_3d = unproject_2d_to_3d_nuscenes(pt_2d, depth, calib)
          # pt_3d[1] += dim[0] / 2
          # print('pt_3d', pt_3d)
          print('location', location)
      if DEBUG:
        cv2.imshow('image', image)
        cv2.waitKey()


    print("# images: ", len(ret['images']))
    print("# annotations: ", len(ret['annotations']))
    # import pdb; pdb.set_trace()
    os.makedirs(os.path.join(DATA_PATH, 'annotations'), exist_ok=True)
    out_path = '{}/annotations/kitti_{}_{}.json'.format(DATA_PATH, SPLIT, split)
    json.dump(ret, open(out_path, 'w'))
  
