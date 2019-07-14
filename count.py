# -*- coding: utf-8 -*-

import models
from torchsummary import summary
from flopth import flopth
#import torch

model = models.UNet(10).cuda()

summary(model, input_size=(3, 256, 256))

flops = flopth(model, in_size=(3, 256, 256))
print(flops)