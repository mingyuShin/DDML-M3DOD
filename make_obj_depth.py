import os
import numpy as np
from lib.datasets.kitti.kitti_utils import get_objects_from_label
from PIL import Image
from lib.datasets.kitti.kitti_utils import get_affine_transform
from lib.datasets.kitti.kitti_utils import affine_transform
from lib.datasets.kitti.kitti_utils import Calibration
import cv2

def get_label(label_dir, idx):
    label_file = os.path.join(label_dir, '%06d.txt' % idx)
    assert os.path.exists(label_file)
    return get_objects_from_label(label_file)
def get_image(image_dir, idx):
    img_file = os.path.join(image_dir, '%06d.png' % idx)
    assert os.path.exists(img_file)
    return Image.open(img_file)
def get_calib(calib_dir, idx):
    calib_file = os.path.join(calib_dir, '%06d.txt' % idx)
    assert os.path.exists(calib_file)
    return Calibration(calib_file)

def make_obj_depth(root_dir, all_class=False):
    if all_class:
        obj_depth_dir = os.path.join(root_dir, "data/KITTI3D/training/obj_depth_all")
    else:
        obj_depth_dir = os.path.join(root_dir, "data/KITTI3D/training/obj_depth")
    os.makedirs(obj_depth_dir, exist_ok=True)
    split_file = os.path.join(root_dir, 'data/KITTI3D/ImageSets', 'trainval.txt')
    idx_list = [x.strip() for x in open(split_file).readlines()]
    image_dir = os.path.join(root_dir, 'data/KITTI3D/training/image_2')
    label_dir = os.path.join(root_dir, 'data/KITTI3D/training/label_2')                       
    calib_dir = os.path.join(root_dir, 'data/KITTI3D/training/calib')    
    for idx in idx_list:
        img = get_image(image_dir, int(idx))
        img_size = np.array(img.size)
        objects = get_label(label_dir, int(idx))
        
        depthmap = np.zeros((img_size[1], img_size[0], 1), dtype=np.float32)
        obj = []
        f_name = '%06d.npy' % int(idx)
        feature1_name = os.path.join(obj_depth_dir, f_name)

        for i in range(len(objects)):
            if not all_class:
                if objects[i].cls_type !=  'Car':
                    continue
                  
            if objects[i].level_str == 'UnKnown' or objects[i].pos[-1] < 2:
                continue
            
            bbox_2d = objects[i].box2d.copy()

            # filter boxes with zero or negative 2D height or width
            temp_w, temp_h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
            if temp_w <= 0 or temp_h <= 0:
                continue            
            
            obj.append(objects[i])
            
        obj = sorted(obj, key= lambda x:x.pos[-1], reverse=True)
        
        for i in range(len(obj)):
            depth = obj[i].pos[-1]
            bbox_2d = obj[i].box2d.copy()

            depthmap[round(bbox_2d[1]):round(bbox_2d[3]), round(bbox_2d[0]): round(bbox_2d[2])] = depth
            
        np.save(feature1_name, depthmap.squeeze())


if __name__ == '__main__':
    path_dir = '/home/cv3/hdd'
    make_obj_depth(path_dir)
    make_obj_depth(path_dir, all_class=True)
    
