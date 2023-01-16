import torch
from torch import nn
import torch.autograd as autograd
import numpy as np
from torch import optim
from stacked_hourglass.layers import Conv, Pool, Residual, Hourglass
from stacked_hourglass.posenet import PoseNet
from stacked_hourglass.config import __config__
from gan_dp import GAN_Dataset
from torch.utils.data import DataLoader


disc_config = {
    'batchsize': 20,
    'input_dim': 17,
    'smooth_decay': 0.998,
    'img_height': 256,
    'img_width': 128,
    'eps': 0.1,
    'lr': 1e-4,
    'weight_decay': 1e-5,
    'optim': 'Adam'
}


class KCS_layer(nn.Module):

    def __init__(self):
        super(KCS_layer, self).__init__()


    def forward(self, x):
        self.C = torch.Tensor([[1,  0,  0,  1,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                               [-1, 1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                               [0, -1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                               [0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                               [0,  0,  0, -1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                               [0,  0,  0,  0, -1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                               [0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                               [0,  0,  0,  0,  0,  0, -1,  1,  0,  0,  0,  0,  0,  0,  0,  0],
                               [0,  0,  0,  0,  0,  0,  0, -1,  1,  0,  1,  0,  0,  1,  0,  0],
                               [0,  0,  0,  0,  0,  0,  0,  0, -1,  1,  0,  0,  0,  0,  0,  0],
                               [0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0],
                               [0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  1,  0,  0,  0,  0],
                               [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  1,  0,  0,  0],
                               [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0],
                               [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  1,  0],
                               [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  1],
                               [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1]]).cuda()
        out = torch.matmul(x, self.C)
        return out


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()
        self.flatten = nn.Flatten()

    def forward(self, x):
        return self.flatten(x)


class PoseDiscriminator(nn.Module):

    def __init__(self):
        super(PoseDiscriminator, self).__init__()
        self.kcs = KCS_layer()
        self.fc1_1 = nn.Linear(16 * 16, 1024)
        self.fc1_2 = nn.Linear(1024, 1024)
        self.fc1_3 = nn.Linear(1024, 1024)
        self.fc2_1 = nn.Linear(2 * 17, 128)
        self.fc2_2 = nn.Linear(128, 128)
        self.fc2_3 = nn.Linear(128, 128)
        self.fc2_4 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(1024 + 1024 + 128, 128)
        self.fc4 = nn.Linear(128, 1)
        self.leaky_relu = nn.LeakyReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x0 = x.view(-1, 2, 17)
        b = self.kcs(x0)
        b_t = b.transpose(-1, -2)
        psi = torch.matmul(b_t, b)
        psi_vec = self.flatten(psi)
        psi_vec = self.fc1_1(psi_vec)
        psi_vec = self.leaky_relu(psi_vec)
        d1_psi = self.fc1_2(psi_vec)
        d1_psi = self.leaky_relu(d1_psi)
        d1_psi = self.fc1_3(d1_psi)
        d2_psi = torch.cat((psi_vec, d1_psi), dim = -1)

        d0 = x
        d1 = self.fc2_1(d0)
        d1 = self.leaky_relu(d1)
        d2 = self.fc2_2(d1)
        d2 = self.leaky_relu(d2)
        d2 = self.fc2_3(d2)
        d3 = torch.cat((d1, d2), dim = -1)
        d3 = self.leaky_relu(d3)
        d3 = self.fc2_4(d3)

        out = torch.cat((d3, d2_psi), dim = -1)
        out = self.fc3(out)
        out = self.leaky_relu(out)
        out = self.fc4(out)

        return out


def calc_gradient_penalty(batchsize, disc, real_data, fake_data):
    alpha = torch.rand(batchsize, 1).expand(real_data.size()).cuda()
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = disc(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim = 1) - 1) ** 2).mean() * 10
    return gradient_penalty


def train_discriminator(dataset, dataloader, gen, disc, optimizer, conf, writer, count):
    batchsize = conf['batchsize']
    input_dim = conf['input_dim']
    eps = conf['eps']
    lr = conf['lr']
    img_width = conf['img_width']
    img_height = conf['img_height']
    weight_decay = conf['weight_decay']
    smooth_decay = conf['smooth_decay']

    for i, data in enumerate(dataloader):
        img, kpts = data[0].cuda(), data[1].to(torch.float32).cuda()
        out = gen(img).detach()
        kpts = kpts.view(-1, 2 * 17)
        out = out.view(-1, 2 * 17)
        d1, d2 = disc(kpts), disc(out)
        loss = - d1.mean() + d2.mean() + calc_gradient_penalty(batchsize, disc, kpts, out)
        print('Discriminator_Loss =', loss.item())
        if count % 10 == 0:
            writer.add_scalar('Discriminator_Loss', loss.item(), count // 10)
            #writer.add_scalar('Discriminator_Learning_Rate', optimizer.state_dict()['param_groups'][0]['lr'], count // 10)
        count += 1
        #if count % 500 == 0:
            #scheduler.step()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i == 4:
            return count
    return count

if __name__ == '__main__':
    pass