import os

# os.chdir(r'C:\Users\horsepurve\Documents\Project\UB\Style\StyleTransferTrilogy\code')

# os.chdir(r'C:\Users\horsepurve\Dropbox\UBR\Analysis\StyleTransferTrilogy\code')
#%%
# import library

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

from utils import *
from models import *

import numpy as np

#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
# load image

'''
image_1 = 'picasso.jpg'
image_2 = 'c.jpg'
'''
image_dir = r'/home/chunweim/.mxnet/datasets/voc/VOC2012/JPEGImages'

#%%

image_1 = 'training_axial_crop_pat3_v0_53.jpg'
image_2 = 'testing_axial_crop_pat14_v0_106.jpg'

width = 512 # 512 300

style_img = read_image(os.path.join(image_dir, image_1), target_width=width).to(device)
content_img = read_image(os.path.join(image_dir,image_2), target_width=width).to(device)

#%%
# build model

vgg16 = models.vgg16(pretrained=True)
vgg16 = VGG(vgg16.features[:23]).to(device).eval()

#%%
# compute feature
style_features = vgg16(style_img)
content_features = vgg16(content_img)

[x.shape for x in content_features]
#%%
# compute Gram matrix
def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

style_grams = [gram_matrix(x) for x in style_features]
[x.shape for x in style_grams]
#%%
# trining image
#% %time
 
import timeit
tic = timeit.default_timer()

input_img = content_img.clone()
optimizer = optim.LBFGS([input_img.requires_grad_()])
style_weight = 1e6 # 1e6
content_weight = 1 # 1

run = [0]
while run[0] <= 50: # 300
    def f():
        optimizer.zero_grad()
        features = vgg16(input_img)
        
        content_loss = F.mse_loss(features[2], content_features[2]) * content_weight
        style_loss = 0
        grams = [gram_matrix(x) for x in features]
        for a, b in zip(grams, style_grams):
            style_loss += F.mse_loss(a, b) * style_weight
        
        loss = style_loss + content_loss
        
        if run[0] % 50 == 0:
            print('Step {}: Style Loss: {:4f} Content Loss: {:4f}'.format(
                run[0], style_loss.item(), content_loss.item()))
        run[0] += 1
        
        loss.backward()
        return loss
    
    optimizer.step(f)
    
toc = timeit.default_timer()
print('time:', toc-tic)

#%%
import imageio
img = content_img.cpu().numpy()[0,:,:,:]
img = img.transpose((1, 2, 0))
imageio.imwrite('content_img.jpg',img)
img = input_img.cpu().detach().numpy()[0,:,:,:]
img = img.transpose((1, 2, 0))
imageio.imwrite('content_img_output.jpg',img)

