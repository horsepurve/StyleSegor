raw_image_dir = r'/home/chunweim/chunweim/projects/StyleSegor/JPEGImages_test/image/'
output_dir = r'/home/chunweim/chunweim/projects/StyleSegor/JPEGImages_test_styled'
style_image = r'training_axial_crop_pat3_v0_53.jpg'

from os import listdir
from os.path import isfile, join
raw_images = [f for f in listdir(raw_image_dir) if isfile(join(raw_image_dir, f))]
raw_images.sort()

print('all images:', len(raw_images))

raw_images = raw_images[4000:] # [:2000] # [2000:4000] # [4000:]

import timeit
import imageio
import os

#%% ========== ========== ========== ========== ========== ========== 
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
print('device:', device)

# build model
vgg16 = models.vgg16(pretrained=True)
vgg16 = VGG(vgg16.features[:23]).to(device).eval()

#%% ========== ========== ========== ========== ========== ========== 
def generate_styled_image(image_1, image_2, image_2_styled):
    """
    parameters: full path
    image_1 = 'training_axial_crop_pat0_v0_36.jpg'
    image_2 = 'testing_axial_crop_pat15_v0_63.jpg'
    image_2_styles = 'testing_axial_crop_pat15_v0_63_0.jpg' # using pat0 style
    """
    width = 512
    style_img = read_image(image_1, target_width=width).to(device)
    content_img = read_image(image_2, target_width=width).to(device)
    #%% ========== ========== ========== ========== ========== ==========

    # compute feature
    style_features = vgg16(style_img)
    content_features = vgg16(content_img)
    #%% ========== ========== ========== ========== ========== ==========
    # compute Gram matrix
    def gram_matrix(y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram

    style_grams = [gram_matrix(x) for x in style_features]
    #%% ========== ========== ========== ========== ========== ==========
    # trining image
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
    #%% ========== ========== ========== ========== ========== ==========
    img = input_img.cpu().detach().numpy()[0,:,:,:]
    img = img.transpose((1, 2, 0))
    imageio.imwrite(image_2_styled,img)    
    #%% ========== ========== ========== ========== ========== ==========


#%% ========== ========== ========== ========== ========== ========== 
for k in range(len(raw_images)):
    print('\n')
    image_path = os.path.join(raw_image_dir, raw_images[k])
    
    image_2 = image_path # current image is the content image
    image_1 = style_image
    image_2_styled = os.path.join(output_dir, raw_images[k].split('.')[0]+'_styled_.jpg')

    print('content image:', image_2)
    print('style image:', image_1)
    print('content image output:', image_2_styled)

    tic = timeit.default_timer()

    generate_styled_image(image_1, image_2, image_2_styled)

    toc = timeit.default_timer()
    print('time:', toc-tic)

    # break
    #%% ========== ========== ========== ========== ========== ==========





