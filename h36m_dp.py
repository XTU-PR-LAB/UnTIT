import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import h5py
import random
import cv2 as cv
import torch.nn.functional as F
import os
import copy

dataset_mode = ['train_transfer', 'train_pose', 'train_disc', 'test_pose']
chains = [[0, 1],
          [1, 2],
          [2, 3],
          [0, 4],
          [4, 5],
          [5, 6],
          [0, 7],
          [7, 8],
          [8, 9],
          [9, 10],
          [8, 11],
          [11, 12],
          [12, 13],
          [8, 14],
          [14, 15],
          [15, 16]]

class HM36_Dataset(Dataset):

    def __init__(self, config, mode = 'train_transfer', need_transform = True):
        super(HM36_Dataset, self).__init__()
        self.config = config
        self.root_dir = self.config['root_dir']
        self.annot_dir = self.config['annot_dir']
        self.num_keypoints = self.config['num_keypoints']
        self.mode = mode
        self.need_transform = need_transform
        train_annot_name = 'prior_unsup_train.h5'
        val_annot_name = ['val - Directions.h5',
                          'val - Discussion.h5',
                          'val - Greeting.h5',
                          'val - Posing.h5',
                          'val - Waiting.h5',
                          'val - Walking.h5']

        if self.mode == 'train_transfer':
            self.img_paths = [[] for i in range(5)]
            self.keypoints = [[] for i in range(5)]
            self.img_maps = []
            train_annot_path = os.path.join(self.annot_dir, train_annot_name).replace('\\', '/')
            if os.path.exists(train_annot_path):
                train_h5 = h5py.File(train_annot_path, 'r')
                for seq, k in enumerate(train_h5.keys()):
                    seq_len = len(train_h5[k]['data_set_2d'])
                    seq_num = int(''.join(list(filter(str.isdigit, k)))) // 120
                    for i in range(seq_len):
                        if i % 10 == 0:
                            img_name = str(train_h5[k]['data_set_image_name'][i])
                            img_path = os.path.join(self.root_dir, img_name).replace('\\', '/')
                            keypoints = train_h5[k]['data_set_2d'][i]
                            if self.is_train_simplified(img_name) == True and os.path.exists(img_path):
                                self.img_maps.append((seq_num, len(self.img_paths[seq_num])))
                                self.img_paths[seq_num].append(img_path)
                                self.keypoints[seq_num].append(keypoints)
                    print(len(self.img_maps))

        elif self.mode == 'train_pose':
            self.img_paths = [[] for i in range(5)]
            self.keypoints = [[] for i in range(5)]
            self.img_maps = []
            train_annot_path = os.path.join(self.annot_dir, train_annot_name).replace('\\', '/')
            if os.path.exists(train_annot_path):
                train_h5 = h5py.File(train_annot_path)
                for seq, k in enumerate(train_h5.keys()):
                    seq_len = len(train_h5[k]['data_set_2d'])
                    seq_num = int(''.join(list(filter(str.isdigit, k)))) // 120
                    for i in range(seq_len):
                        if i % 2 != 0:
                            img_name = str(train_h5[k]['data_set_image_name'][i])
                            img_path = os.path.join(self.root_dir, img_name).replace('\\', '/')
                            keypoints = train_h5[k]['data_set_2d'][i]
                            if self.is_train_simplified(img_name) == True and os.path.exists(img_path):
                                self.img_maps.append((seq_num, len(self.img_paths[seq_num])))
                                self.img_paths[seq_num].append(img_path)
                                self.keypoints[seq_num].append(keypoints)
                    print(len(self.img_maps))

        elif self.mode == 'train_disc':
            self.img_paths = []
            self.keypoints = []
            train_annot_path = os.path.join(self.annot_dir, train_annot_name).replace('\\', '/')
            if os.path.exists(train_annot_path):
                train_h5 = h5py.File(train_annot_path)
                for seq, k in enumerate(train_h5.keys()):
                    seq_len = len(train_h5[k]['data_set_2d'])
                    for i in range(seq_len):
                        if i % 2 == 0:
                            img_name = str(train_h5[k]['data_set_image_name'][i])
                            img_path = os.path.join(self.root_dir, img_name).replace('\\', '/')
                            keypoints = train_h5[k]['data_set_2d'][i]
                            if self.is_train_simplified(img_name) == True and os.path.exists(img_path):
                                self.img_paths.append(img_path)
                                self.keypoints.append(keypoints)
                    print(len(self.img_paths))

        elif self.mode == 'test_pose':
            pass

    def __len__(self):
        if self.mode == 'train_transfer' or self.mode == 'train_pose':
            return len(self.img_maps)
        elif self.mode == 'train_disc' or self.mode == 'test_pose':
            return len(self.img_paths)

    def is_train_simplified(self, img_name):
        return img_name[5: 11] == 'act_02' or \
               img_name[5: 11] == 'act_03' or \
               img_name[5: 11] == 'act_05' or \
               img_name[5: 11] == 'act_08' or \
               img_name[5: 11] == 'act_13' or \
               img_name[5: 11] == 'act_15'

    def is_test_simplified(self, img_name):
        return  img_name[: 4] == 's_09' and (
                img_name[5: 11] == 'act_02' or
                img_name[5: 11] == 'act_03' or
                img_name[5: 11] == 'act_05' or
                img_name[5: 11] == 'act_08' or
                img_name[5: 11] == 'act_13' or
                img_name[5: 11] == 'act_15')

    def img_to_tensor(self, img):
        trans_f = transforms.ToTensor()
        return trans_f(img)

    def img_resize(self, img, img_size):
        img = img.resize(img_size, resample=2)
        return img

    def get_theta(self, T, img_size):
        img_width, img_height = img_size[0], img_size[1]
        theta1 = torch.zeros(3, 3)
        theta1[0, 0] = 2 / img_width
        theta1[0, 2] = -1
        theta1[1, 1] = 2 / img_height
        theta1[1, 2] = -1
        theta1[2, 2] = 1
        theta2 = T.inverse()
        theta3 = theta1.inverse()
        theta = theta1.matmul(theta2).matmul(theta3)
        return theta

    def img_affine_with_keypoints(self, img, annot, img_size, T):
        new_img = copy.deepcopy(img)
        new_annot = copy.deepcopy(annot)
        theta = self.get_theta(T, img_size)[: 2, :].unsqueeze(0)
        new_img = new_img.unsqueeze(0)
        grid = F.affine_grid(theta, size = new_img.shape, align_corners = False)
        output = F.grid_sample(new_img, grid, align_corners = False)[0]
        for i in range(self.num_keypoints):
            v = torch.Tensor([new_annot[0][i], new_annot[1][i], 1])
            new_annot[0][i], new_annot[1][i] = T[0].dot(v).item(), T[1].dot(v).item()
        return output, new_annot

    def random_transform(self, img_size):
        img_width, img_height = img_size[0], img_size[1]
        T1 = self.translate(-(img_width // 2), -(img_height // 2))
        T2 = self.scale(np.random.uniform(0.9, 1), np.random.uniform(0.9, 1))
        T3 = self.rotate(np.random.uniform(-np.pi / 30, np.pi / 30))
        T4 = self.shear(np.random.uniform(-0.02, 0.02), np.random.uniform(-0.02, 0.02))
        T5 = self.translate(img_width // 2, img_height // 2)
        T = T5.matmul(T4).matmul(T3).matmul(T2).matmul(T1)
        return T

    def scale(self, scale_x, scale_y):
        T = torch.zeros(3, 3)
        T[0, 0] = scale_x
        T[1, 1] = scale_y
        T[2, 2] = 1
        return T

    def rotate(self, degree):
        T = torch.zeros(3, 3)
        degree = torch.Tensor([degree])
        T[0, 0] = torch.cos(degree).item()
        T[0, 1] = -torch.sin(degree).item()
        T[1, 0] = torch.sin(degree).item()
        T[1, 1] = torch.cos(degree).item()
        T[2, 2] = 1
        return T

    def translate(self, translate_x, translate_y):
        T = torch.zeros(3, 3)
        T[0, 0] = T[1, 1] = T[2, 2] = 1
        T[0, 2] = translate_x
        T[1, 2] = translate_y
        return T

    def shear(self, shear_x, shear_y):
        T = torch.zeros(3, 3)
        T[0, 0] = T[1, 1] = T[2, 2] = 1
        T[0, 1] = shear_x
        T[1, 0] = shear_y
        return T

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

    def img_transform_with_keypoints(self, img, keypoints, need_transform = True):
        new_img = img.copy()
        new_keypoints = keypoints.copy()

        origin_img_height = self.config['origin_img_height']
        origin_img_width = self.config['origin_img_width']
        origin_img_size = (origin_img_width, origin_img_height)
        new_img = self.img_resize(new_img, origin_img_size)
        new_keypoints = new_keypoints.reshape(-1, 2).transpose()

        max_x = np.max(new_keypoints[0])
        min_x = np.min(new_keypoints[0])
        max_y = np.max(new_keypoints[1])
        min_y = np.min(new_keypoints[1])
        delta = self.config['crop_delta']
        box = (min_x - delta, min_y - delta, max_x + delta, max_y + delta)

        new_img, new_keypoints = self.img_crop_with_keypoints(new_img, new_keypoints, box)

        cropped_img_height = self.config['cropped_img_height']
        cropped_img_width = self.config['cropped_img_width']
        cropped_img_size = (cropped_img_width, cropped_img_height)

        new_img, new_keypoints = self.img_resize_with_keypoints(new_img, new_keypoints, cropped_img_size)

        new_img = self.img_to_tensor(new_img)

        if need_transform == True:

            T, tran_img, tran_keypoints = 0, 0, 0
            while True:
                T = self.random_transform(cropped_img_size)
                tran_img, tran_keypoints = self.img_affine_with_keypoints(new_img, new_keypoints, cropped_img_size, T)
                flag = 1
                for i in range(self.num_keypoints):
                    if tran_keypoints[0][i] < 0 or tran_keypoints[1][i] < 0 or tran_keypoints[0][i] >= cropped_img_width or tran_keypoints[1][i] >= cropped_img_height:
                        flag = 0
                        break
                if flag == 1:
                    break
            heatmap = self.generate_heatmap(tran_keypoints, cropped_img_size)

            return tran_img, tran_keypoints, heatmap, T

        else:
            heatmap = self.generate_heatmap(new_keypoints, cropped_img_size)
            return new_img, new_keypoints, heatmap

    def __getitem__(self, idx):
        if self.mode == 'train_transfer':
            img_map = self.img_maps[idx]
            seq_id = img_map[0]
            pos = img_map[1]
            seq_len = len(self.keypoints[seq_id])
            pos_2 = random.randint(0, seq_len - 1)
            img_path = self.img_paths[seq_id][pos]
            img_path_2 = self.img_paths[seq_id][pos_2]
            img = Image.open(img_path).convert('RGB')
            img_2 = Image.open(img_path_2).convert('RGB')
            keypoints = self.keypoints[seq_id][pos].copy()
            keypoints_2 = self.keypoints[seq_id][pos_2].copy()

            img, keypoints, heatmap, T = self.img_transform_with_keypoints(img, keypoints, need_transform = True)
            img_2, keypoints_2, heatmap_2 = self.img_transform_with_keypoints(img_2, keypoints_2, need_transform = False)
            img = self.img_normalize(img)
            img_2 = self.img_normalize(img_2)
            heatmap = torch.Tensor(heatmap)
            heatmap_2 = torch.Tensor(heatmap_2)
            return img, heatmap, img_2, heatmap_2

        elif self.mode == 'train_pose':
            img_map = self.img_maps[idx]
            seq_id = img_map[0]
            pos = img_map[1]
            img_path = self.img_paths[seq_id][pos]
            img = Image.open(img_path).convert('RGB')
            keypoints = self.keypoints[seq_id][pos].copy()

            img_0, keypoints_0, heatmap_0 = self.img_transform_with_keypoints(img, keypoints, need_transform = False)
            img_1, keypoints_1, heatmap_1, T_1 = self.img_transform_with_keypoints(img, keypoints, need_transform = True)
            img_2, keypoints_2, heatmap_2, T_2 = self.img_transform_with_keypoints(img, keypoints, need_transform = True)

            img_0 = self.img_normalize(img_0)
            img_1 = self.img_normalize(img_1)
            img_2 = self.img_normalize(img_2)

            return img_0, img_1, T_1, img_2, T_2

        elif self.mode == 'train_disc':
            img_path = self.img_paths[idx]
            img = Image.open(img_path).convert('RGB')
            keypoints = self.keypoints[idx].copy()

            img_0, keypoints_0, heatmap_0, T_0 = self.img_transform_with_keypoints(img, keypoints, need_transform = True)
            img_0 = self.img_normalize(img_0)

            return img_0, keypoints_0

        elif self.mode == 'test_pose':
            pass


def get_theta(T, img_size):
    img_width, img_height = img_size[0], img_size[1]
    theta1 = torch.zeros(3, 3)
    theta1[0, 0] = 2 / img_width
    theta1[0, 2] = -1
    theta1[1, 1] = 2 / img_height
    theta1[1, 2] = -1
    theta1[2, 2] = 1
    theta2 = T.inverse()
    theta3 = theta1.inverse()
    theta = theta1.matmul(theta2).matmul(theta3)
    return theta

def img_affine_with_keypoints(img, annot, img_size, T):
    new_img = copy.deepcopy(img)
    new_annot = copy.deepcopy(annot)
    theta = get_theta(T, img_size)[: 2, :].unsqueeze(0)
    new_img = new_img.unsqueeze(0)
    grid = F.affine_grid(theta, size = new_img.shape, align_corners = False)
    output = F.grid_sample(new_img, grid, align_corners = False)[0]
    for i in range(17):
        v = torch.Tensor([new_annot[0][i], new_annot[1][i], 1])
        new_annot[0][i], new_annot[1][i] = T[0].dot(v).item(), T[1].dot(v).item()
    return output, new_annot

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
    hm36 = HM36_Dataset(config, mode = 'train_disc', need_transform = True)
    for i in range(len(hm36)):
        img, kpts = hm36[i]
        print(i)
        for j in range(0, 17):
            if kpts[0][j] < 0 or kpts[1][j] < 0 or kpts[0][j] >= 128 or kpts[1][j] >= 256:
                print('the model is crashed!')
                break
        '''img_0, img_1, T_1, img_2, T_2 = hm36[i]
        tran = transforms.ToPILImage()
        img_size = (config['cropped_img_width'], config['cropped_img_height'])
        theta_1 = get_theta(T_1.inverse(), img_size)[: 2, :].unsqueeze(0)
        theta_2 = get_theta(T_2.inverse(), img_size)[: 2, :].unsqueeze(0)
        inv_img_1 = img_1.unsqueeze(0)
        inv_img_2 = img_2.unsqueeze(0)
        grid_1 = F.affine_grid(theta_1, size = inv_img_1.shape, align_corners=False)
        grid_2 = F.affine_grid(theta_2, size = inv_img_2.shape, align_corners=False)
        output_1 = F.grid_sample(inv_img_1, grid_1, align_corners=False)[0]
        output_2 = F.grid_sample(inv_img_2, grid_2, align_corners=False)[0]
        new_img = np.array(tran(img_0), dtype=np.uint8)[:, :, [2, 1, 0]].copy()
        new_img_2 = np.array(tran(img_1), dtype=np.uint8)[:, :, [2, 1, 0]].copy()
        new_img_3 = np.array(tran(img_2), dtype=np.uint8)[:, :, [2, 1, 0]].copy()
        output_1 = np.array(tran(output_1), dtype=np.uint8)[:, :, [2, 1, 0]].copy()
        output_2 = np.array(tran(output_2), dtype=np.uint8)[:, :, [2, 1, 0]].copy()
        new_img = np.concatenate([new_img, new_img_2, new_img_3, output_1, output_2], axis=1)
        cv.imshow('img', new_img)
        cv.waitKey(0)'''
