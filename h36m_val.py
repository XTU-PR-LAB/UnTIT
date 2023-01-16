import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import h5py
import random
import cv2 as cv
import os


val_list = ['val - Directions.h5',
            'val - Discussion.h5',
            'val - Greeting.h5',
            'val - Posing.h5',
            'val - Waiting.h5',
            'val - Walking.h5']


class HM36_Val_Dataset(Dataset):

    def __init__(self, config, mode = 'test_pose', val_type = 0):
        # if val_type = 6, then use all data for testing
        super(HM36_Val_Dataset, self).__init__()
        self.config = config
        self.root_dir = config['root_dir']
        self.annot_dir = config['annot_dir']
        self.mode = mode
        self.num_keypoints = self.config['num_keypoints']
        self.img_paths = []
        self.keypoints = []

        for num, val_set in enumerate(val_list):
            if val_type == 6 or val_type == num:
                annot_path = os.path.join(self.annot_dir, val_set)
                if os.path.exists(annot_path):
                    val_h5 = h5py.File(annot_path)
                    for seq, k in enumerate(val_h5.keys()):
                        seq_len = len(val_h5[k]['data_set_image_name'])
                        for i in range(seq_len):
                            img_name = str(val_h5[k]['data_set_image_name'][i])
                            img_path = os.path.join(self.root_dir, img_name).replace('\\', '/')
                            keypoints = val_h5[k]['data_set_2d'][i]
                            if self.is_train_simplified(img_name) and os.path.exists(img_path):
                                self.img_paths.append(img_path)
                                self.keypoints.append(keypoints)
                    print(len(self.img_paths), 'images have been loaded.')

    def is_train_simplified(self, img_name):
        return img_name[: 4] == 's_11' and \
               (img_name[5: 11] == 'act_02' or \
               img_name[5: 11] == 'act_03' or \
               img_name[5: 11] == 'act_05' or \
               img_name[5: 11] == 'act_08' or \
               img_name[5: 11] == 'act_13' or \
               img_name[5: 11] == 'act_15')

    def __len__(self):
        return len(self.img_paths)

    def img_to_tensor(self, img):
        trans_f = transforms.ToTensor()
        return trans_f(img)

    def img_resize(self, img, img_size):
        img = img.resize(img_size, resample=2)
        return img

    def img_resize_with_keypoints(self, img, keypoints, img_size):
        scale_x = img_size[0] / img.width
        scale_y = img_size[1] / img.height
        img = img.resize(img_size, resample=2)
        keypoints[0] *= scale_x
        keypoints[1] *= scale_y
        return img, keypoints

    def img_crop_with_keypoints(self, img, keypoints, box):
        img_height = img.height
        img_width = img.width
        x1 = max(int(box[0]), 0)
        y1 = max(int(box[1]), 0)
        x2 = min(int(box[2]), img_width)
        y2 = min(int(box[3]), img_height)
        img = img.crop((x1, y1, x2, y2))
        keypoints[0] -= x1
        keypoints[1] -= y1
        return img, keypoints

    def img_normalize(self, img):
        trans_f = transforms.Normalize((0.5, 0.5, 0.5),
                                       (0.5, 0.5, 0.5))
        return trans_f(img)

    def generate_heatmap(self, keypoints, img_size, sigma=6):
        img_width = img_size[0]
        img_height = img_size[1]
        result = np.zeros((self.num_keypoints, img_height, img_width))
        for i in range(self.num_keypoints):
            x, y = keypoints[0][i], keypoints[1][i]
            xx, yy = np.meshgrid(np.arange(img_width), np.arange(img_height))
            result[i] = np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma ** 2))
        return result

    def img_transform_with_keypoints(self, img, keypoints):
        origin_img_height = self.config['origin_img_height']
        origin_img_width = self.config['origin_img_width']
        origin_img_size = (origin_img_width, origin_img_height)
        img = self.img_resize(img, origin_img_size)

        keypoints = keypoints.reshape(-1, 2).transpose()

        max_x = np.max(keypoints[0])
        min_x = np.min(keypoints[0])
        max_y = np.max(keypoints[1])
        min_y = np.min(keypoints[1])
        delta = self.config['crop_delta']
        box = (min_x - delta, min_y - delta, max_x + delta, max_y + delta)

        img, keypoints = self.img_crop_with_keypoints(img, keypoints, box)

        cropped_img_height = self.config['cropped_img_height']
        cropped_img_width = self.config['cropped_img_width']
        cropped_img_size = (cropped_img_width, cropped_img_height)

        img, keypoints = self.img_resize_with_keypoints(img, keypoints, cropped_img_size)
        heatmap = self.generate_heatmap(keypoints, cropped_img_size)

        return img, keypoints, heatmap

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        keypoints = self.keypoints[idx].copy()
        img = Image.open(img_path).convert('RGB')
        img, keypoints, heatmap = self.img_transform_with_keypoints(img, keypoints)
        img = self.img_to_tensor(img)
        img = self.img_normalize(img)
        return img, keypoints


if __name__ == '__main__':
    config = {
        'root_dir': 'Human3.6M/resized_images_299/',
        'annot_dir': 'Human3.6M/h5_file/',
        'num_keypoints': 17,
        'origin_img_height': 1000,
        'origin_img_width': 1000,
        'cropped_img_height': 256,
        'cropped_img_width': 128,
        'crop_delta': 50
    }
    hm36 = HM36_Val_Dataset(config)