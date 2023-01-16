import time
import numpy as np
import torch
import cv2 as cv
import datetime
from torchvision import transforms
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from gan_dp import GAN_Dataset
from coco_dataset import ref
from torch import nn
from torch.utils.data import DataLoader
from h36m_dp import HM36_Dataset
from mpi_dp import MPI_Dataset
from pose_transfer_dp import Pose_Transfer_Dataset
import os
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
    'batchsize': 16,
    'num_epochs': 1000
}

def main():
    opt = TrainOptions().parse()
    #data_loader = CreateDataLoader(opt)
    #dataset = data_loader.load_data()
    #dataset_size = len(data_loader)
    #print('#training images = %d' % dataset_size)

    dataset = HM36_Dataset(config)
    dataloader = DataLoader(dataset, batch_size = opt.batchSize, num_workers = 2, shuffle = True, drop_last = True)

    #model_path = 'checkpoints/market_PATN/latest_net_netG.pth'
    model = create_model(opt)
    #model = nn.DataParallel(model)
    #model = torch.load(model_path)
    visualizer = Visualizer(opt)
    #generate_heatmap = GenerateHeatmap()
    total_steps = 0
    writer = SummaryWriter('./train_log')
    count = 0
    for epoch in range(config['num_epochs']):
        epoch_start_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataloader):
            iter_start_time = time.time()
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            img, heatmap, img_2, heatmap_2 = data[0].cuda(), data[1].cuda(), data[2].cuda(), data[3].cuda()
            pose = {}
            pose['P1'] = img
            pose['P2'] = img_2
            pose['BP1'] = heatmap
            pose['BP2'] = heatmap_2
            now = datetime.datetime.now()
            pose['P1_path'] = pose['P2_path'] = now.strftime("%Y-%m-%d_%H-%M-%S")
            model.set_input(pose)
            model.optimize_parameters()
            result = model.fake_p2

            if count % 20 == 0:
                loss = model.pair_L1loss
                writer.add_scalar('Pair_Loss', loss, count // 20)
            count += 1
            if i % 100 == 0:
                tran = transforms.ToPILImage()
                origin = np.array(tran(img[0] * 0.5 + 0.5))[:, :, [2, 1, 0]]
                target = np.array(tran(img_2[0] * 0.5 + 0.5))[:, :, [2, 1, 0]]
                fake = np.array(tran(result[0] * 0.5 + 0.5))[:, :, [2, 1, 0]]
                cat_img = np.concatenate([origin, target, fake], axis = 1)
                cv.imwrite('results/{0}_{1}.png'.format(epoch, i // 100), cat_img)

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

if __name__ == '__main__':
    main()