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
image_dir = r'D:\HVSMR\HVSMR\JPEGImages' # image_dir = r'/home/chunweim/.mxnet/datasets/voc/VOC2012/JPEGImages'

#%%

image_1 = 'training_axial_crop_pat3_v0_53.jpg'
image_2 = 'testing_axial_crop_pat17_v0_136.jpg'

width = 512 # 512 300

style_img = read_image(os.path.join(image_dir, image_1), target_width=width).to(device)
content_img = read_image(os.path.join(image_dir,image_2), target_width=width).to(device)
#%%
# plot image
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
imshow(style_img, title='Style Image')

plt.subplot(1, 2, 2)
imshow(content_img, title='Content Image')
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
# display styled image
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
imshow(style_img, title='Style Image')

plt.subplot(1, 3, 2)
imshow(content_img, title='Content Image')

plt.subplot(1, 3, 3)
imshow(input_img, title='Output Image')

# save_name = "styled_image.png"
save_name = "styled_image_med.png"

# plt.savefig(save_name)
#%%
import imageio
img = content_img.cpu().numpy()[0,:,:,:]
img = img.transpose((1, 2, 0))
imageio.imwrite('content_img.jpg',img)
img = input_img.cpu().detach().numpy()[0,:,:,:]
img = img.transpose((1, 2, 0))
imageio.imwrite('content_img_output.jpg',img)

#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from utils import *
from models import *

import numpy as np

from tqdm import tqdm
import random

#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%%
# load image
style_path = "training_axial_crop_pat3_v0_53.jpg" # "picasso.jpg"
style_img = read_image(style_path).to(device)
imshow(style_img, title='Style Image')
#%%
# build VGG16 model

# extract patial feature
vgg16 = models.vgg16(pretrained=True)
vgg16 = VGG(vgg16.features[:23]).to(device).eval()
#%%
# build transformation network
def ConvLayer(in_channels, out_channels, kernel_size=3, stride=1, 
    upsample=None, instance_norm=True, relu=True):
    layers = []
    if upsample:
        layers.append(nn.Upsample(mode='nearest', scale_factor=upsample))
    layers.append(nn.ReflectionPad2d(kernel_size // 2))
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride))
    if instance_norm:
        layers.append(nn.InstanceNorm2d(out_channels))
    if relu:
        layers.append(nn.ReLU())
    return layers

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            *ConvLayer(channels, channels, kernel_size=3, stride=1), 
            *ConvLayer(channels, channels, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.conv(x) + x

class TransformNet(nn.Module):
    def __init__(self, base=32):
        super(TransformNet, self).__init__()
        self.downsampling = nn.Sequential(
            *ConvLayer(3, base, kernel_size=9), 
            *ConvLayer(base, base*2, kernel_size=3, stride=2), 
            *ConvLayer(base*2, base*4, kernel_size=3, stride=2), 
        )
        self.residuals = nn.Sequential(*[ResidualBlock(base*4) for i in range(5)])
        self.upsampling = nn.Sequential(
            *ConvLayer(base*4, base*2, kernel_size=3, upsample=2),
            *ConvLayer(base*2, base, kernel_size=3, upsample=2),
            *ConvLayer(base, 3, kernel_size=9, instance_norm=False, relu=False),
        )
    
    def forward(self, X):
        y = self.downsampling(X)
        y = self.residuals(y)
        y = self.upsampling(y)
        return y
#%%
# Gram matrix funtion
def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram
#%%
# training model

# load COCO dataset
# !rm -rf /home/ypw/COCO/*/.AppleDouble

batch_size = 4
width = 512 # 256

data_transform = transforms.Compose([
    transforms.Resize(width), 
    transforms.CenterCrop(width), 
    transforms.ToTensor(), 
    tensor_normalizer, 
])

dataset = torchvision.datasets.ImageFolder('/home/chunweim/chunweim/projects/StyleSegor/JPEGImages_test', # '/media/MyDataStor2/chunweim/data/COCO',
                                           transform=data_transform)
data_loader = torch.utils.data.DataLoader(dataset, 
                                          batch_size=batch_size,
                                          shuffle=True)
print(dataset)
#%%
# compute Gram matrix
style_features = vgg16(style_img)
style_grams = [gram_matrix(x) for x in style_features]
style_grams = [x.detach() for x in style_grams]
[x.shape for x in style_grams]
#%%
# start training

#%mkdir -p debug

def tensor_to_array(tensor):
    x = tensor.cpu().detach().numpy()
    x = (x*255).clip(0, 255).transpose(0, 2, 3, 1).astype(np.uint8)
    return x

def save_debug_image(style_images, content_images, transformed_images, filename):
    style_image = Image.fromarray(recover_image(style_images))
    content_images = [recover_image(x) for x in content_images]
    transformed_images = [recover_image(x) for x in transformed_images]
    
    new_im = Image.new('RGB', (style_image.size[0] + (width + 5) * 4, max(style_image.size[1], width*2 + 5)))
    new_im.paste(style_image, (0,0))
    
    x = style_image.size[0] + 5
    for i, (a, b) in enumerate(zip(content_images, transformed_images)):
        new_im.paste(Image.fromarray(a), (x + (width + 5) * i, 0))
        new_im.paste(Image.fromarray(b), (x + (width + 5) * i, width + 5))
    
    new_im.save(filename)
#%%
transform_net = TransformNet(32).to(device)
#%%
verbose_batch = 800
style_weight = 1e5
content_weight = 1
tv_weight = 1e-6

optimizer = optim.Adam(transform_net.parameters(), 1e-3)
transform_net.train()

n_batch = len(data_loader)

for epoch in range(1):
    print('Epoch: {}'.format(epoch+1))
    smooth_content_loss = Smooth()
    smooth_style_loss = Smooth()
    smooth_tv_loss = Smooth()
    smooth_loss = Smooth()
    with tqdm(enumerate(data_loader), total=n_batch) as pbar:
        for batch, (content_images, _) in pbar:
            optimizer.zero_grad()

            # use style model to predict styled image
            content_images = content_images.to(device)
            transformed_images = transform_net(content_images)
            transformed_images = transformed_images.clamp(-3, 3)

            # use vgg16 to compute feature
            content_features = vgg16(content_images)
            transformed_features = vgg16(transformed_images)

            # content loss
            content_loss = content_weight * F.mse_loss(transformed_features[1], content_features[1])
            
            # total variation loss
            y = transformed_images
            tv_loss = tv_weight * (torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + 
            torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))

            # style loss
            style_loss = 0.
            transformed_grams = [gram_matrix(x) for x in transformed_features]
            for transformed_gram, style_gram in zip(transformed_grams, style_grams):
                style_loss += style_weight * F.mse_loss(transformed_gram, 
                                                        style_gram.expand_as(transformed_gram))

            # add up
            loss = style_loss + content_loss + tv_loss

            loss.backward()
            optimizer.step()

            smooth_content_loss += content_loss.item()
            smooth_style_loss += style_loss.item()
            smooth_tv_loss += tv_loss.item()
            smooth_loss += loss.item()
            
            s = f'Content: {smooth_content_loss:.2f} '
            s += f'Style: {smooth_style_loss:.2f} '
            s += f'TV: {smooth_tv_loss:.4f} '
            s += f'Loss: {smooth_loss:.2f}'
            if batch % verbose_batch == 0:
                s = '\n' + s
                save_debug_image(style_img, content_images, transformed_images, 
                                 f"debug/s2_{epoch}_{batch}.jpg")
            
            pbar.set_description(s)
    torch.save(transform_net.state_dict(), 'transform_net.pth')
#%%
# output styled image
content_img = random.choice(dataset)[0].unsqueeze(0).to(device)
output_img = transform_net(content_img)

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
imshow(style_img, title='Style Image')

plt.subplot(1, 3, 2)
imshow(content_img, title='Content Image')

plt.subplot(1, 3, 3)
imshow(output_img.detach(), title='Output Image')

plt.savefig("styled_image_situation2.png")
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%%

# import library

import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
from PIL import Image
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import shutil
from glob import glob

from tensorboardX import SummaryWriter

import numpy as np
import multiprocessing

import copy
from tqdm import tqdm
from collections import defaultdict

import horovod.torch as hvd
import torch.utils.data.distributed

from utils import *
from models import *
import time

from pprint import pprint
display = pprint

hvd.init()
torch.cuda.set_device(hvd.local_rank())

device = torch.device("cuda:%s" %hvd.local_rank() if torch.cuda.is_available() else "cpu")
#%%
is_hvd = False
tag = 'nohvd'
base = 32
style_weight = 50
content_weight = 1
tv_weight = 1e-6
epochs = 22

batch_size = 4 # 8
width = 256

verbose_hist_batch = 100
verbose_image_batch = 800


model_name = f'metanet_base{base}_style{style_weight}_tv{tv_weight}_tag{tag}'
print(f'model_name: {model_name}, rank: {hvd.rank()}')
'''
model_name = 'metanet_base32_style25_tv1e-07_l21e-05_taghvd'
'''
#%%
def rmrf(path):
    try:
        shutil.rmtree(path)
    except:
        pass

for f in glob('runs/*/.AppleDouble'):
    rmrf(f)

rmrf('runs/' + model_name)
#%%
# build the model

vgg16 = models.vgg16(pretrained=True)
vgg16 = VGG(vgg16.features[:23]).to(device).eval()

transform_net = TransformNet(base).to(device)
transform_net.get_param_dict()
#%%
metanet = MetaNet(transform_net.get_param_dict()).to(device)
#%%
# load dataset
data_transform = transforms.Compose([
    transforms.RandomResizedCrop(width, scale=(256/480, 1), ratio=(1, 1)), 
    transforms.ToTensor(), 
    tensor_normalizer
])

style_dataset = torchvision.datasets.ImageFolder('HVSMR_test', # '/home/ypw/WikiArt/', 
                                                 transform=data_transform)
content_dataset = torchvision.datasets.ImageFolder('HVSMR_train', # '/home/ypw/COCO/',
                                                   transform=data_transform)

if is_hvd:
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        content_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    content_data_loader = torch.utils.data.DataLoader(content_dataset, batch_size=batch_size, 
        num_workers=multiprocessing.cpu_count(),sampler=train_sampler)
else:
    content_data_loader = torch.utils.data.DataLoader(content_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=multiprocessing.cpu_count())

if not is_hvd or hvd.rank() == 0:
    print(style_dataset)
    print('-'*20)
    print(content_dataset)
#%%
# test infer
metanet.eval()
transform_net.eval()

rands = torch.rand(4, 3, 256, 256).to(device)
features = vgg16(rands);
weights = metanet(mean_std(features));
transform_net.set_weights(weights)
transformed_images = transform_net(torch.rand(4, 3, 256, 256).to(device));

if not is_hvd or hvd.rank() == 0:
    print('features:')
    display([x.shape for x in features])
    
    print('weights:')
    display([x.shape for x in weights.values()])

    print('transformed_images:')
    display(transformed_images.shape)
#%%
# initialization
visualization_style_image = random.choice(style_dataset)[0].unsqueeze(0).to(device)
visualization_content_images = torch.stack([random.choice(content_dataset)[0] for i in range(4)]).to(device)
#%%
if not is_hvd or hvd.rank() == 0:
    for f in glob('runs/*/.AppleDouble'):
        rmrf(f)

    rmrf('runs/' + model_name)
    writer = SummaryWriter('runs/'+model_name)
else:
    writer = SummaryWriter('/tmp/'+model_name)

visualization_style_image = random.choice(style_dataset)[0].unsqueeze(0).to(device)
visualization_content_images = torch.stack([random.choice(content_dataset)[0] for i in range(4)]).to(device)

'''
writer.add_image('content_image', recover_tensor(visualization_content_images), 0)
writer.add_graph(transform_net, (rands, ))
'''

del rands, features, weights, transformed_images
#%%
trainable_params = {}
trainable_param_shapes = {}
for model in [vgg16, transform_net, metanet]:
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params[name] = param
            trainable_param_shapes[name] = param.shape
#%%
# start training
optimizer = optim.Adam(trainable_params.values(), 1e-3)

if is_hvd:
    optimizer = hvd.DistributedOptimizer(optimizer, 
                                         named_parameters=trainable_params.items())
    params = transform_net.state_dict()
    params.update(metanet.state_dict())
    hvd.broadcast_parameters(params, root_rank=0)
#%%
n_batch = len(content_data_loader)
metanet.train()
transform_net.train()

for epoch in range(epochs):
    smoother = defaultdict(Smooth)
    with tqdm(enumerate(content_data_loader), total=n_batch) as pbar:
        for batch, (content_images, _) in pbar:
            n_iter = epoch*n_batch + batch
            
            # 每 20 个 batch 随机挑选一张新的风格图像，计算其特征
            if batch % 20 == 0:
                style_image = random.choice(style_dataset)[0].unsqueeze(0).to(device)
                style_features = vgg16(style_image)
                style_mean_std = mean_std(style_features)
            
            # 检查纯色
            x = content_images.cpu().numpy()
            if (x.min(-1).min(-1) == x.max(-1).max(-1)).any():
                continue
            
            optimizer.zero_grad()
            
            # 使用风格图像生成风格模型
            weights = metanet(mean_std(style_features))
            transform_net.set_weights(weights, 0)
            
            # 使用风格模型预测风格迁移图像
            content_images = content_images.to(device)
            transformed_images = transform_net(content_images)

            # 使用 vgg16 计算特征
            content_features = vgg16(content_images)
            transformed_features = vgg16(transformed_images)
            transformed_mean_std = mean_std(transformed_features)
            
            # content loss
            content_loss = content_weight * F.mse_loss(transformed_features[2], content_features[2])
            
            # style loss
            style_loss = style_weight * F.mse_loss(transformed_mean_std, 
                                                   style_mean_std.expand_as(transformed_mean_std))
            
            # total variation loss
            y = transformed_images
            tv_loss = tv_weight * (torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + 
                                    torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))
            
            # 求和
            loss = content_loss + style_loss + tv_loss 
            
            loss.backward()
            optimizer.step()
            
            smoother['content_loss'] += content_loss.item()
            smoother['style_loss'] += style_loss.item()
            smoother['tv_loss'] += tv_loss.item()
            smoother['loss'] += loss.item()
            
            max_value = max([x.max().item() for x in weights.values()])
        
            writer.add_scalar('loss/loss', loss, n_iter)
            writer.add_scalar('loss/content_loss', content_loss, n_iter)
            writer.add_scalar('loss/style_loss', style_loss, n_iter)
            writer.add_scalar('loss/total_variation', tv_loss, n_iter)
            writer.add_scalar('loss/max', max_value, n_iter)
            
            s = 'Epoch: {} '.format(epoch+1)
            s += 'Content: {:.2f} '.format(smoother['content_loss'])
            s += 'Style: {:.1f} '.format(smoother['style_loss'])
            s += 'Loss: {:.2f} '.format(smoother['loss'])
            s += 'Max: {:.2f}'.format(max_value)
            
            if (batch + 1) % verbose_image_batch == 0:
                transform_net.eval()
                visualization_transformed_images = transform_net(visualization_content_images)
                transform_net.train()
                visualization_transformed_images = torch.cat([style_image, visualization_transformed_images])
                writer.add_image('debug', recover_tensor(visualization_transformed_images), n_iter)
                del visualization_transformed_images
            
            if (batch + 1) % verbose_hist_batch == 0:
                for name, param in weights.items():
                    writer.add_histogram('transform_net.'+name, param.clone().cpu().data.numpy(), 
                                         n_iter, bins='auto')
                
                for name, param in transform_net.named_parameters():
                    writer.add_histogram('transform_net.'+name, param.clone().cpu().data.numpy(), 
                                         n_iter, bins='auto')
                
                for name, param in metanet.named_parameters():
                    l = name.split('.')
                    l.remove(l[-1])
                    writer.add_histogram('metanet.'+'.'.join(l), param.clone().cpu().data.numpy(), 
                                         n_iter, bins='auto')

            pbar.set_description(s)
            
            del transformed_images, weights
        
    if not is_hvd or hvd.rank() == 0:
        torch.save(metanet.state_dict(), 'checkpoints/{}_{}.pth'.format(model_name, epoch+1))
        torch.save(transform_net.state_dict(), 
                   'checkpoints/{}_transform_net_{}.pth'.format(model_name, epoch+1))
        
        torch.save(metanet.state_dict(), 'models/{}.pth'.format(model_name))
        torch.save(transform_net.state_dict(), 'models/{}_transform_net.pth'.format(model_name))
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ==========
#%% Situation3_test_speed.ipynb
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from tensorboardX import SummaryWriter

import random
import shutil
from glob import glob
from tqdm import tqdm

from utils import *
from models import *

#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
#%%
def rmrf(path):
    try:
        shutil.rmtree(path)
    except:
        pass

for f in glob('runs/*/.AppleDouble'):
    rmrf(f)

rmrf('runs/metanet')
rmrf('runs/transform_net')
#%%
# build the model
vgg16 = models.vgg16(pretrained=True)
vgg16 = VGG(vgg16.features[:23]).to(device).eval()

base = 32
transform_net = TransformNet(base).to(device)
transform_net.get_param_dict()
#%%
transform_net
#%%
class MetaNet(nn.Module):
    def __init__(self, param_dict):
        super(MetaNet, self).__init__()
        self.param_num = len(param_dict)
        self.hidden = nn.Linear(1920, 128*self.param_num)
        self.fc_dict = {}
        for i, (name, params) in enumerate(param_dict.items()):
            self.fc_dict[name] = i
            setattr(self, 'fc{}'.format(i+1), nn.Linear(128, params))
    
    # ONNX 要求输出 tensor 或者 list，不能是 dict
    def forward(self, mean_std_features):
        hidden = F.relu(self.hidden(mean_std_features))
        filters = {}
        for name, i in self.fc_dict.items():
            fc = getattr(self, 'fc{}'.format(i+1))
            filters[name] = fc(hidden[:,i*128:(i+1)*128])
        return list(filters.values())
    
    def forward2(self, mean_std_features):
        hidden = F.relu(self.hidden(mean_std_features))
        filters = {}
        for name, i in self.fc_dict.items():
            fc = getattr(self, 'fc{}'.format(i+1))
            filters[name] = fc(hidden[:,i*128:(i+1)*128])
        return filters
#%%
metanet = MetaNet(transform_net.get_param_dict()).to(device)
#%%
metanet
#%%
# output to tensorboard
mean_std_features = torch.rand(4, 1920).to(device)
writer = SummaryWriter('runs/metanet')
writer.add_graph(metanet, (mean_std_features, ))

rands = torch.rand(4, 3, 256, 256).to(device)
writer = SummaryWriter('runs/transform_net')
writer.add_graph(transform_net, (rands, ))
#%%
# speed test
metanet.load_state_dict(torch.load('models/metanet_base32_style50_tv1e-06_tagnohvd.pth'))
transform_net.load_state_dict(torch.load('models/metanet_base32_style50_tv1e-06_tagnohvd_transform_net.pth'))
#%%
X = torch.rand((1, 3, 256, 256)).to(device)
#%%
%%time
for i in range(1000):
    features = vgg16(X)
    mean_std_features = mean_std(features)
    weights = metanet.forward2(mean_std_features)
    transform_net.set_weights(weights)
    del features, mean_std_features, weights
'''
CPU times: user 10.3 s, sys: 5.21 s, total: 15.5 s
Wall time: 15.6 s
'''
%%time
for i in range(1000):
    transform_net(X)
'''
CPU times: user 7.08 s, sys: 3.23 s, total: 10.3 s
Wall time: 10.3 s
'''
%%time
for i in range(1000):
    features = vgg16(X)
    mean_std_features = mean_std(features)
    weights = metanet.forward2(mean_std_features)
    transform_net.set_weights(weights)
    transform_net(X)
    del features, mean_std_features, weights
'''
CPU times: user 20.7 s, sys: 10.1 s, total: 30.8 s
Wall time: 30.8 s
'''
#%%
# visulization
width = 256

data_transform = transforms.Compose([
    transforms.RandomResizedCrop(width, scale=(256/480, 1), ratio=(1, 1)), 
    transforms.ToTensor(), 
    tensor_normalizer
])

style_dataset = torchvision.datasets.ImageFolder('HVSMR_test', # '/home/ypw/WikiArt/', 
                                                 transform=data_transform)
content_dataset = torchvision.datasets.ImageFolder('HVSMR_train', # '/home/ypw/COCO/', 
                                                   transform=data_transform)
#%%
# epoch = 19
# metanet.load_state_dict(torch.load(
#     f'checkpoints/metanet_base32_style50_tv1e-06_tag1_{epoch}.pth'))
# transform_net.load_state_dict(torch.load(
#     f'checkpoints/metanet_base32_style50_tv1e-06_tag1_transform_net_{epoch}.pth'))
#%%
style_weight = 50
content_weight = 1
tv_weight = 1e-6
batch_size = 8

trainable_params = {}
trainable_param_shapes = {}
for model in [vgg16, transform_net, metanet]:
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params[name] = param
            trainable_param_shapes[name] = param.shape

optimizer = optim.Adam(trainable_params.values(), 1e-3)
content_data_loader = torch.utils.data.DataLoader(content_dataset, batch_size=batch_size, shuffle=True)
#%%
style_image = read_image('testing_axial_crop_pat16_v2_99.jpg', # '../images/test.jpg', 
                         target_width=256).to(device)
style_features = vgg16(style_image)
style_mean_std = mean_std(style_features)

metanet.load_state_dict(torch.load('models/metanet_base32_style50_tv1e-06_tagnohvd.pth'))
transform_net.load_state_dict(torch.load('models/metanet_base32_style50_tv1e-06_tagnohvd_transform_net.pth'))

n_batch = 20
with tqdm(enumerate(content_data_loader), total=n_batch) as pbar:
    for batch, (content_images, _) in pbar:
        x = content_images.cpu().numpy()
        if (x.min(-1).min(-1) == x.max(-1).max(-1)).any():
            continue
        
        optimizer.zero_grad()
        
        # 使用风格图像生成风格模型
        weights = metanet.forward2(mean_std(style_features))
        transform_net.set_weights(weights, 0)

        # 使用风格模型预测风格迁移图像
        content_images = content_images.to(device)
        transformed_images = transform_net(content_images)

        # 使用 vgg16 计算特征
        content_features = vgg16(content_images)
        transformed_features = vgg16(transformed_images)
        transformed_mean_std = mean_std(transformed_features)

        # content loss
        content_loss = content_weight * F.mse_loss(transformed_features[2], content_features[2])

        # style loss
        style_loss = style_weight * F.mse_loss(transformed_mean_std, 
                                               style_mean_std.expand_as(transformed_mean_std))

        # total variation loss
        y = transformed_images
        tv_loss = tv_weight * (torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + 
                                torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))

        # 求和
        loss = content_loss + style_loss + tv_loss 

        loss.backward()
        optimizer.step()
        
        if batch > n_batch:
            break

content_images = torch.stack([random.choice(content_dataset)[0] for i in range(4)]).to(device)
# while content_images.min() < -2:
#     print('.', end=' ')
#     content_images = torch.stack([random.choice(content_dataset)[0] for i in range(4)]).to(device)
transformed_images = transform_net(content_images)

transformed_images_vis = torch.cat([x for x in transformed_images], dim=-1)
content_images_vis = torch.cat([x for x in content_images], dim=-1)


plt.figure(figsize=(20, 12))
plt.subplot(3, 1, 1)
imshow(style_image)
plt.subplot(3, 1, 2)
imshow(content_images_vis)
plt.subplot(3, 1, 3)
imshow(transformed_images_vis)
#%%
# style_image = random.choice(style_dataset)[0].unsqueeze(0).to(device)
style_image = read_image('../images/mosaic.jpg', target_width=256).to(device)
# style_image = style_image[:,[2, 1, 0]]
features = vgg16(style_image)
mean_std_features = mean_std(features)
weights = metanet.forward2(mean_std_features)
transform_net.set_weights(weights)

content_images = torch.stack([random.choice(content_dataset)[0] for i in range(4)]).to(device)
# while content_images.min() < -2:
#     print('.', end=' ')
#     content_images = torch.stack([random.choice(content_dataset)[0] for i in range(4)]).to(device)
transformed_images = transform_net(content_images)

transformed_images_vis = torch.cat([x for x in transformed_images], dim=-1)
content_images_vis = torch.cat([x for x in content_images], dim=-1)


plt.figure(figsize=(20, 12))
plt.subplot(3, 1, 1)
imshow(style_image)
plt.subplot(3, 1, 2)
imshow(content_images_vis)
plt.subplot(3, 1, 3)
imshow(transformed_images_vis)
#%%

















#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 
#%% ========== ========== ========== ========== ========== ========== 



