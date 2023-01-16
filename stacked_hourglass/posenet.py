import torch
from torch import nn
from stacked_hourglass.layers import Conv, Hourglass, Pool, Residual
from stacked_hourglass.loss import HeatmapLoss


class PoseNet(nn.Module):

    def __init__(self, inp_dim, oup_dim, bn=False, increase=0, **kwargs):
        super(PoseNet, self).__init__()
        self.pre = nn.Sequential(
            Conv(3, 64, 7, 2, bn=True, relu=True),
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 128),
            Residual(128, inp_dim)
        )
        self.hgs = Hourglass(4, inp_dim, bn, increase)
        self.feature = nn.Sequential(
                Residual(inp_dim, inp_dim),
                Conv(inp_dim, inp_dim, 1, bn=True, relu=True))
        self.flatten = nn.Flatten()
        self.out = nn.Linear(524288, 2 * 17)

    def forward(self, imgs):
        x = self.pre(imgs)
        hg = self.hgs(x)
        feature = self.flatten(self.feature(hg))
        pred = self.out(feature)
        return pred
