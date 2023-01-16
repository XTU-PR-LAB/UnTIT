import torch
from torch import nn
from h36m_val import HM36_Val_Dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2 as cv

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

val_list = ['val - Directions.h5',
            'val - Discussion.h5',
            'val - Greeting.h5',
            'val - Posing.h5',
            'val - Waiting.h5',
            'val - Walking.h5',
            'val - Total']

class MeanLoss(nn.Module):
    def __init__(self, img_size):
        super(MeanLoss, self).__init__()
        self.img_size = torch.Tensor(img_size).unsqueeze(1).cuda()

    def forward(self, pre, gt):
        pre = pre.view(-1, 2, 17) / self.img_size
        gt = gt.view(-1, 2, 17) / self.img_size
        return torch.mean(torch.abs(pre - gt), dim = -1).mean(dim = -1)


def main():
    gen = torch.load('checkpoints/generator_ablation3_epoch_2.pth').cuda()
    gen.eval()
    eval_f = MeanLoss(img_size=(128, 256))
    color_set = [(255, 0, 0),
                 (255, 63, 0),
                 (255, 127, 0),
                 (255, 191, 0),
                 (255, 255, 0),
                 (191, 255, 0),
                 (127, 255, 0),
                 (63, 255, 0),
                 (0, 255, 0),
                 (0, 255, 63),
                 (0, 255, 127),
                 (0, 255, 191),
                 (0, 255, 255),
                 (0, 191, 255),
                 (0, 127, 255),
                 (0, 63, 255),
                 (0, 0, 255)]
    total_ans = []
    final_len, final_ans = 0, 0
    for k in range(0, 6):
        print('testing {}'.format(val_list[k]))
        dataset = HM36_Val_Dataset(config, mode = 'test_pose', val_type = k)
        dataloader = DataLoader(dataset, batch_size = 1, shuffle = True, drop_last = True)
        dataset_len = dataset.__len__()
        with torch.no_grad():
            ans, cnt = 0, 0
            for i, data in enumerate(dataloader):
                if i >= 2000:
                    break
                img, kpts = data[0].cuda(), data[1].to(torch.float32).cuda()
                out = gen(img)
                distance = eval_f(out, kpts)
                #print('Current_Mean_Distance =', distance.item())
                ans += distance.item()
                if i % 50 == 0:
                    print('Current_Mean_Distance =', ans / (i + 1))
                if i % 50 == 0:
                    tran = transforms.ToPILImage()
                    result1 = np.array(tran(img[0] * 0.5 + 0.5))[:, :, [2, 1, 0]].copy()
                    result2 = result1.copy()
                    pred = out[0].view(2, 17)
                    for j in range(16):
                        pos1, pos2 = chains[j][0], chains[j][1]
                        cv.line(result2, (int(pred[0][pos1]), int(pred[1][pos1])),
                                (int(pred[0][pos2]), int(pred[1][pos2])), color=color_set[(j * 8) % 17],
                                thickness=2)
                    result = np.concatenate((result1, result2), axis=1)
                    cv.imwrite('results/test_results/ablation3_{0}_{1}.png'.format(k, i // 50), result)
                cnt += 1
            total = ans / cnt
            print('Total_Mean_Distance =', total)
            total_ans.append(total)
            final_len += cnt
            final_ans += ans
    final_total = final_ans / final_len
    print('All_Mean_Distance =', final_total)
    total_ans.append(final_total)
    print(' ')
    print('****************************Ablation3 Results ****************************')
    for i in range(7):
        print(val_list[i], 'result:', total_ans[i])


if __name__ == '__main__':
    main()