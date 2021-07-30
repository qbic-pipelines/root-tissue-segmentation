from typing import Any, Optional

import torch
from torch import nn

__all__ = ['U2NET']

# Adopted from https://github.com/xuebinqin/U-2-Net
from models.unet_super import UNetsuper
from models.unet_utils import _upsample_like, _size_map, RSU


class U2NET(UNetsuper):
    def __init__(self, num_classes, len_test_set: int, hparams: dict, input_channels=1, min_filter=32, **kwargs):
        super().__init__(num_classes, len_test_set, hparams, input_channels, min_filter, **kwargs)
        self._make_layers(input_channels, min_filter)

    def forward(self, x):
        sizes = _size_map(x, self.height)
        maps = []  # storage for maps

        # side saliency map
        def unet(x, height=1):
            if height < 6:
                x1 = getattr(self, f'stage{height}')(x)
                x2 = unet(getattr(self, 'downsample')(x1), height + 1)
                x = getattr(self, f'stage{height}d')(torch.cat((x2, x1), 1))
                side(x, height)
                return _upsample_like(x, sizes[height - 1]) if height > 1 else x
            else:
                x = getattr(self, f'stage{height}')(x)
                side(x, height)
                return _upsample_like(x, sizes[height - 1])

        def side(x, h):
            # side output saliency map (before sigmoid)
            x = getattr(self, f'side{h}')(x)
            x = _upsample_like(x, sizes[1])
            maps.append(x)

        def fuse():
            # fuse saliency probability maps
            maps.reverse()
            x = torch.cat(maps, 1)
            x = getattr(self, 'outconv')(x)
            maps.insert(0, x)
            return [torch.sigmoid(x) for x in maps]

        unet(x)
        maps = fuse()
        return maps

    def _make_layers(self, input_channels, min_filter):
        cfgs = {
            # cfgs for building RSUs and sides
            # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
            'stage1': ['En_1', (7, input_channels, min_filter, min_filter * 2), -1],
            'stage2': ['En_2', (6, min_filter * 2, min_filter, min_filter * 2 ** 2), -1],
            'stage3': ['En_3', (5, min_filter * 2 ** 2, min_filter * 2, min_filter * 2 ** 3), -1],
            'stage4': ['En_4', (4, min_filter * 2 ** 3, min_filter * 2 ** 2, min_filter * 2 ** 4), -1],
            'stage5': ['En_5', (4, min_filter * 2 ** 4, min_filter * 2 ** 3, min_filter * 2 ** 4, True), -1],
            'stage6': ['En_6', (4, min_filter * 2 ** 4, min_filter * 2 ** 3, min_filter * 2 ** 4, True),
                       min_filter * 2 ** 4],
            'stage5d': ['De_5', (4, min_filter * 2 ** 5, min_filter * 2 ** 3, min_filter * 2 ** 4, True),
                        min_filter * 2 ** 4],
            'stage4d': ['De_4', (4, min_filter * 2 ** 5, min_filter * 2 ** 2, min_filter * 2 ** 3),
                        min_filter * 2 ** 3],
            'stage3d': ['De_3', (5, min_filter * 2 ** 4, min_filter * 2, min_filter * 2 ** 2), min_filter * 2 ** 2],
            'stage2d': ['De_2', (6, min_filter * 2 ** 3, min_filter, min_filter * 2), min_filter * 2],
            'stage1d': ['De_1', (7, min_filter * 2 ** 2, int(min_filter * 2 ** (1 / 2)), min_filter * 2),
                        min_filter * 2],
        }
        cfgs = {
            # cfgs for building RSUs and sides
            # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
            'stage1': ['En_1', (7, 1, 32, 64), -1],
            'stage2': ['En_2', (6, 64, 32, 128), -1],
            'stage3': ['En_3', (5, 128, 64, 256), -1],
            'stage4': ['En_4', (4, 256, 128, 512), -1],
            'stage5': ['En_5', (4, 512, 256, 512, True), -1],
            'stage6': ['En_6', (4, 512, 256, 512, True), 512],
            'stage5d': ['De_5', (4, 1024, 256, 512, True), 512],
            'stage4d': ['De_4', (4, 1024, 128, 256), 256],
            'stage3d': ['De_3', (5, 512, 64, 128), 128],
            'stage2d': ['De_2', (6, 256, 32, 64), 64],
            'stage1d': ['De_1', (7, 128, 16, 64), 64],
        }
        self.height = int((len(cfgs) + 1) / 2)
        self.add_module('downsample', nn.MaxPool2d(2, stride=2, ceil_mode=True))
        for k, v in cfgs.items():
            # build rsu block
            self.add_module(k, RSU(v[0], *v[1]))
            if v[2] > 0:
                # build side layer
                self.add_module(f'side{v[0][-1]}', nn.Conv2d(v[2], self.num_classes, 3, padding=1))
        # build fuse layer
        self.add_module('outconv', nn.Conv2d(int(self.height * self.num_classes), self.num_classes, 1))

    def loss(self, logits, labels):
        """
        Initializes the loss function

        :return: output - Initialized cross entropy loss function
        """
        labels = labels.long()
        loss = 0
        for logit in logits:
            loss += self.criterion(logit, labels)
        return loss

    def predict(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None):
        data, target = batch
        output = self.forward(data)
        _, prediction = torch.max(output[0], dim=1)
        return data, target, output, prediction
