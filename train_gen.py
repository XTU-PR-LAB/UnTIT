import time
import os
import torch

import gan_dp
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
import time
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from stacked_hourglass.posenet import PoseNet
from stacked_hourglass.config import __config__
from losses.L1_plus_perceptualLoss import L1_plus_perceptualLoss
from torch import optim
from losses.MS_SSIM_plus_L1 import MS_SSIM_L1_LOSS
from torch import nn
from pose_gan import PoseDiscriminator
from h36m_dp import HM36_Dataset
from coco_dataset import ref
import pose_gan

import numpy as np
import cv2 as cv
import pose_gan
import copy

from tensorboardX import SummaryWriter

config = {
    'root_dir': 'Human3.6M/resized_images_299/',
    'annot_dir': 'Human3.6M/h5_file/',
    'num_keypoints': 17,
    'origin_img_height': 1000,
    'origin_img_width': 1000,
    'cropped_img_height': 256,
    'cropped_img_width': 128,
    'crop_delta': 50,
    'num_epochs': 100
}

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


class GenerateHeatmap(nn.Module):

    def __init__(self):
        super(GenerateHeatmap, self).__init__()

    def forward(self, x):
        x = x.view(-1, 2, 17)
        return self.cords_to_map(x)

    def cords_to_map(self, cords, img_size=(256, 128), sigma=6):
        batchsize = cords.size()[0]
        out = []
        for batch in range(batchsize):
            result = []
            for i in range(17):
                yy, xx = torch.meshgrid(torch.arange(img_size[0]), torch.arange(img_size[1]))
                yy = yy.cuda()
                xx = xx.cuda()
                result.append(
                    torch.exp(-((xx - cords[batch][0][i]) ** 2 + (yy - cords[batch][1][i]) ** 2) / (2 * sigma ** 2)))
            out.append(torch.stack(result, dim=0))
        return torch.stack(out, dim=0)


def get_bbox(fasterrcnn, batchsize, img):
    bbox = fasterrcnn(img)
    bbox_result = []
    for batch in range(img.size()[0]):
        flag = -1
        for j in range(len(bbox[batch]['labels'])):
            if bbox[batch]['labels'][j] == 1:
                bbox_result.append(bbox[batch]['boxes'][j].detach())
                flag = 1
                break
        if flag == -1:
            bbox_result.append(torch.Tensor([-1, -1, -1, -1]).cuda())
    return torch.stack(bbox_result, dim = 0)

def get_pred_bbox(kpts):
    cords = kpts.view(-1, 2, 17)
    b_x1 = torch.min(cords[:, 0, :], dim=1)[0]
    b_x2 = torch.max(cords[:, 0, :], dim=1)[0]
    b_y1 = torch.min(cords[:, 1, :], dim=1)[0]
    b_y2 = torch.max(cords[:, 1, :], dim=1)[0]
    result = torch.stack((b_x1, b_y1, b_x2, b_y2), dim=0).transpose(-1, -2)
    return result


def bbox_iou(box1, box2):
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    if torch.cuda.is_available():
        inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1, torch.zeros(inter_rect_x2.shape).cuda()) * torch.max(
            inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape).cuda())
    else:
        inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1, torch.zeros(inter_rect_x2.shape)) * torch.max(
            inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape))

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return (1 - iou).mean()


def get_inverse_pose(batchsize, num_keypoints, kpts, T):
    kpts = kpts.view(-1, 2, 17)
    new_kpts = torch.zeros_like(kpts)
    batchsize - kpts.size()[0]
    for i in range(batchsize):
        inv_T = T[i].inverse()
        for j in range(num_keypoints):
            x = inv_T[0][0] * kpts[i][0][j] + inv_T[0][1] * kpts[i][1][j] + inv_T[0][2]
            y = inv_T[1][0] * kpts[i][0][j] + inv_T[1][1] * kpts[i][1][j] + inv_T[1][2]
            new_kpts[i][0][j] = x
            new_kpts[i][1][j] = y
    return new_kpts


def main():
    # -----------------------------------#
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = TestOptions().parse()
    opt.nThreads = 0
    opt.batchSize = 8
    opt.serial_batches = True
    opt.no_flip = True

    # -----------------------------------#

    inp_dim = __config__['inference']['inp_dim']
    oup_dim = __config__['inference']['oup_dim']
    img_height = __config__['train']['img_height']
    img_width = __config__['train']['img_width']
    num_keypoints = __config__['inference']['num_parts']
    lr = __config__['train']['learning_rate']
    num_epochs = __config__['train']['num_epochs']
    batchsize = 4#__config__['train']['batchsize']
    lambda_A = 10
    lambda_B = 10
    perceptual_layers = 3
    alpha = 1
    beta = 150

    # ----------------------------------- #

    dataset = HM36_Dataset(config, mode='train_pose')
    dataset2 = HM36_Dataset(config, mode='train_disc')
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=2, drop_last=True)
    dataloader2 = DataLoader(dataset2, batch_size=batchsize, shuffle=True, num_workers=2, drop_last=True)

    # 目标检测器
    fasterrcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).cuda()
    #fasterrcnn = nn.DataParallel(fasterrcnn)
    gen = PoseNet(inp_dim, oup_dim).cuda()
    #gen = nn.DataParallel(gen)
    disc = PoseDiscriminator().cuda()
    #disc = nn.DataParallel(disc)
    model = create_model(opt)
    generate_heatmap = GenerateHeatmap()
    visualizer = Visualizer(opt)
    writer = SummaryWriter('./log_test')

    # ----------------------------------- #
    optimizer_gen = optim.Adam(filter(lambda p: p.requires_grad, gen.parameters()), lr=lr, betas=(0.5, 0.9))
    optimizer_disc = optim.Adam(filter(lambda p: p.requires_grad, disc.parameters()), lr=lr, betas=(0.5, 0.9))
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_gen, gamma=0.99)
    #scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer_disc, gamma=0.99)
    # -----------------------------------#

    loss_f = L1_plus_perceptualLoss(lambda_A, lambda_B, perceptual_layers, opt.gpu_ids, percep_is_l1=1)
    # loss_f = MS_SSIM_L1_LOSS()
    loss_mse = nn.MSELoss()

    # -----------------------------------#

    model.eval()
    fasterrcnn.eval()

    count1, count2, count3 = 0, 0, 0

    # freeze
    for p in model.parameters():
        p.requires_grad = False
    for p in fasterrcnn.parameters():
        p.requires_grad = False

    for epoch in range(num_epochs):
        print('current_epoch =', epoch)

        for i, data in enumerate(dataloader):
            count2 = pose_gan.train_discriminator(dataset2, dataloader2, gen, disc, optimizer_disc, scheduler2, pose_gan.disc_config, writer,
                                                  count2)
            img_0, img_1, T_1, img_2, T_2 = data[0].cuda(), data[1].cuda(), data[2].cuda(), data[3].cuda(), data[
                4].cuda()

            out_1 = gen(img_1).view(-1, 2 * 17)
            out_2 = gen(img_2).view(-1, 2 * 17)

            disc_result_1 = disc(out_1)
            disc_result_2 = disc(out_2)

            loss_g = (- disc_result_1.mean() - disc_result_2.mean()) / 2

            bbox = get_bbox(fasterrcnn, batchsize, img_0 * 0.5 + 0.5)
            # bbox_1 = get_bbox(fasterrcnn, batchsize, img_1 * 0.5 + 0.5)
            # bbox_2 = get_bbox(fasterrcnn, batchsize, img_2 * 0.5 + 0.5)

            heatmap_1_1 = generate_heatmap(out_1)
            heatmap_2_1 = generate_heatmap(out_2)

            inv_out_1 = get_inverse_pose(batchsize, num_keypoints, out_1, T_1)
            inv_out_2 = get_inverse_pose(batchsize, num_keypoints, out_2, T_2)

            bbox_pred_1 = get_pred_bbox(inv_out_1)
            bbox_pred_2 = get_pred_bbox(inv_out_2)
            loss_b = (bbox_iou(bbox, bbox_pred_1) + bbox_iou(bbox, bbox_pred_2)) / 2

            loss_c = loss_mse(inv_out_1, inv_out_2)

            heatmap_1_2 = generate_heatmap(inv_out_1)
            heatmap_2_2 = generate_heatmap(inv_out_2)

            pose_1 = {}
            pose_1['P1'] = img_1
            pose_1['P2'] = img_0
            pose_1['BP1'] = heatmap_1_1
            pose_1['BP2'] = heatmap_1_2
            pose_1['P1_path'] = pose_1['P2_path'] = '_', '_'
            model.set_input(pose_1)
            model.test()
            transfer_img_1 = model.fake_p2
            loss_p1 = loss_f(transfer_img_1, img_0)[0]

            pose_2 = {}
            pose_2['P1'] = img_2
            pose_2['P2'] = img_0
            pose_2['BP1'] = heatmap_2_1
            pose_2['BP2'] = heatmap_2_2
            pose_2['P1_path'] = pose_2['P2_path'] = '_', '_'
            model.set_input(pose_2)
            model.test()
            transfer_img_2 = model.fake_p2
            loss_p2 = loss_f(transfer_img_2, img_0)[0]
            loss_p = (loss_p1 + loss_p2) / 2
            loss = loss_g + alpha * loss_b + beta * loss_p + loss_c
            print('Generator_Loss =', loss_g.item())
            print('Bounding_Box_Loss =', loss_b.item())
            print('Pose_Transfer_Loss =', loss_p.item())
            print('Agreement_Loss =', loss_c.item())

            if count1 % 10 == 0:
                writer.add_scalar('BoundingBox_Loss', loss_b.item(), count1 // 10)
                writer.add_scalar('Generator_Loss', loss_g.item(), count1 // 10)
                writer.add_scalar('Pose_Transfer_Loss', loss_p.item(), count1 // 10)
                writer.add_scalar('Agreement_Loss', loss_c.item(), count1 // 10)
                writer.add_scalar('Generator_Learning_Rate', optimizer_gen.state_dict()['param_groups'][0]['lr'], count1 // 10)
            count1 += 1
            if count1 % 100 == 0:
                scheduler.step()
            optimizer_gen.zero_grad()
            loss.backward()
            optimizer_gen.step()

            if i % 100 == 0:
                tran = transforms.ToPILImage()
                result1 = np.array(tran(img_1[0] * 0.5 + 0.5))[:, :, [2, 1, 0]].copy()
                result2 = np.zeros_like(result1).copy()
                result3 = result1.copy()
                result4 = np.array(tran(img_0[0] * 0.5 + 0.5))[:, :, [2, 1, 0]].copy()
                result5 = result4.copy()
                result6 = np.array(tran(transfer_img_1[0] * 0.5 + 0.5))[:, :, [2, 1, 0]]
                pred_1 = out_1[0].view(2, 17)
                pred_2 = inv_out_1[0].view(2, 17)
                result_bbox = bbox[0]
                result_pred_bbox = bbox_pred_1[0]

                x1, y1, x2, y2 = int(result_bbox[0]), int(result_bbox[1]), int(result_bbox[2]), int(result_bbox[3])
                x3, y3, x4, y4 = int(result_pred_bbox[0]), int(result_pred_bbox[1]), int(result_pred_bbox[2]), int(result_pred_bbox[3])

                cv.line(result4, (x1, y1), (x1, y2), color = (255, 0, 0), thickness=2)
                cv.line(result4, (x1, y1), (x2, y1), color = (255, 0, 0), thickness=2)
                cv.line(result4, (x2, y2), (x1, y2), color = (255, 0, 0), thickness=2)
                cv.line(result4, (x2, y2), (x2, y1), color = (255, 0, 0), thickness=2)
                cv.line(result4, (x3, y3), (x3, y4), color = (0, 255, 0), thickness=2)
                cv.line(result4, (x3, y3), (x4, y3), color = (0, 255, 0), thickness=2)
                cv.line(result4, (x4, y4), (x3, y4), color = (0, 255, 0), thickness=2)
                cv.line(result4, (x4, y4), (x4, y3), color = (0, 255, 0), thickness=2)

                mark = [0 for j in range(17)]
                for j in range(17):
                    if pred_1[0][j] >= 0 and pred_1[0][j] < 128 and pred_1[1][j] >= 0 and pred_1[1][j] < 256:
                        mark[j] = 1
                        cv.circle(result2, (int(pred_1[0][j]), int(pred_1[1][j])), radius=5,
                                  color=(255, 255, 255), thickness=-1)
                        cv.circle(result3, (int(pred_1[0][j]), int(pred_1[1][j])), radius=5,
                                  color=(255, 255, 255), thickness=-1)
                for j in range(16):
                    pos1, pos2 = chains[j][0], chains[j][1]
                    if mark[chains[j][0]] == 1 and mark[chains[j][1]] == 1:
                        cv.line(result2, (int(pred_1[0][pos1]), int(pred_1[1][pos1])),
                                (int(pred_1[0][pos2]), int(pred_1[1][pos2])), color=(255, 255, 255),
                                thickness=2)
                        cv.line(result3, (int(pred_1[0][pos1]), int(pred_1[1][pos1])),
                                (int(pred_1[0][pos2]), int(pred_1[1][pos2])), color=(255, 255, 255),
                                thickness=2)
                mark = [0 for j in range(17)]
                for j in range(17):
                    if pred_2[0][j] >= 0 and pred_2[0][j] < 128 and pred_2[1][j] >= 0 and pred_2[1][j] < 256:
                        mark[j] = 1
                        cv.circle(result5, (int(pred_2[0][j]), int(pred_2[1][j])), radius=5,
                                  color=(255, 255, 255), thickness=-1)
                for j in range(16):
                    pos1, pos2 = chains[j][0], chains[j][1]
                    if mark[chains[j][0]] == 1 and mark[chains[j][1]] == 1:
                        cv.line(result5, (int(pred_2[0][pos1]), int(pred_2[1][pos1])),
                                (int(pred_2[0][pos2]), int(pred_2[1][pos2])), color=(255, 255, 255),
                                thickness=2)

                result = np.concatenate((result1, result2, result3, result4, result5, result6), axis=1)
                cv.imwrite('results/test_{}_{}.png'.format(epoch, i // 100), result)

        torch.save(gen, 'checkpoints/test_generator_epoch_{}.pth'.format(epoch))


if __name__ == '__main__':
    main()