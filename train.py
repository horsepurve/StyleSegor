horse_changed = False # True # False

import os
import shutil
import argparse
import numpy as np
from tqdm import tqdm

import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon.data.vision import transforms

import gluoncv
from gluoncv.loss import *
from gluoncv.utils import LRScheduler
from gluoncv.model_zoo.segbase import *
from gluoncv.model_zoo import get_model
from gluoncv.utils.parallel import *
from gluoncv.data import get_segmentation_dataset

import sys
sys.setrecursionlimit(10000)
print('>>> set up maximum recursion depth')
print_shape = True
import pickle

def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='MXNet Gluon \
                                     Segmentation')
    # model and dataset 
    parser.add_argument('--outdir', type=str, default='outdir', 
                        help='outdir (default: outdir)')
    parser.add_argument('--scoredir', type=str, default='scoredir', 
                        help='scoredir (default: scoredir)')
    parser.add_argument('--save_name', type=str, default='checkpoint', 
    					help='name for saving parameters (default: checkpoint)')
    parser.add_argument('--model', type=str, default='fcn',
                        help='model name (default: fcn)')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='backbone name (default: resnet50)')
    parser.add_argument('--dataset', type=str, default='pascal_voc',
                        help='dataset name (default: pascal)')
    parser.add_argument('--workers', type=int, default=16,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=520,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=480,
                        help='crop image size')
    parser.add_argument('--train-split', type=str, default='train',
                        help='dataset train split (default: train)')
    # training hyper params
    parser.add_argument('--aux', action='store_true', default= False,
                        help='Auxiliary loss')
    parser.add_argument('--aux-weight', type=float, default=0.5, # default=0.5
                        help='auxiliary loss weight')
    parser.add_argument('--epochs', type=int, default=30, metavar='N', # 50
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=16,
                        metavar='N', help='input batch size for \
                        training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=16,
                        metavar='N', help='input batch size for \
                        testing (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--momentum', type=float, default=0.9, # default=0.9
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, # default=1e-4
                        metavar='M', help='w-decay (default: 1e-4)')
    parser.add_argument('--no-wd', action='store_true',
                        help='whether to remove weight decay on bias, \
                        and beta/gamma for batchnorm layers.')
    # cuda and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--ngpus', type=int,
                        default=len(mx.test_utils.list_gpus()),
                        help='number of GPUs (default: 4)')
    parser.add_argument('--kvstore', type=str, default='device',
                        help='kvstore to use for trainer/module.')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='data type for training. default is float32')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default='default',
                        help='set the checkpoint name')
    parser.add_argument('--model-zoo', type=str, default=None,
                        help='evaluating on model zoo model')
    # evaluation only
    parser.add_argument('--eval', action='store_true', default= False,
                        help='evaluation only')
    parser.add_argument('--no-val', action='store_true', default= False,
                            help='skip validation during training')
    # synchronized Batch Normalization
    parser.add_argument('--syncbn', action='store_true', default= False,
                        help='using Synchronized Cross-GPU BatchNorm')
    # the parser
    args = parser.parse_args()
    # handle contexts
    if args.no_cuda:
        print('Using CPU')
        args.kvstore = 'local'
        args.ctx = [mx.cpu(0)]
    else:
        print('Number of GPUs:', args.ngpus)
        args.ctx = [mx.gpu(i) for i in range(args.ngpus)]
    # Synchronized BatchNorm
    args.norm_layer = mx.gluon.contrib.nn.SyncBatchNorm if args.syncbn \
        else mx.gluon.nn.BatchNorm
    args.norm_kwargs = {'num_devices': args.ngpus} if args.syncbn else {}
    print(args)
    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args

        self.two_model = False ##
        self.semi = False

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
            # transforms.Normalize([0, 0, 0], [1, 1, 1]), # ([0, 0, 0], [1, 1, 1])
            # transforms.Normalize([0], [1]), # this is for 1 channel: ([0], [1]) ([556.703], [482.175])
        ])

        # dataset and dataloader
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size,
                       'crop_size': args.crop_size}

        trainset = get_segmentation_dataset(args.dataset, 
                                            split=args.train_split, 
                                            mode='train', 
                                            **data_kwargs)

        valset = get_segmentation_dataset(args.dataset, 
                                          split='val', 
                                          mode='val', 
                                          **data_kwargs)

        self.train_data = gluon.data.DataLoader(trainset, 
                                                args.batch_size, 
                                                shuffle=True, 
                                                last_batch='rollover',
                                                num_workers=args.workers)

        self.eval_data = gluon.data.DataLoader(valset, 
                                               args.batch_size, # args.test_batch_size, [horse changed this]
                                               last_batch='rollover', 
                                               num_workers=args.workers)

        # create network
        if args.model_zoo is not None:
            print('get model from the zoo.')
            model = get_model(args.model_zoo, pretrained=True)
            if self.two_model:
                self.model2 = get_model(args.model_zoo, pretrained=True) ## 2nd identical model
        else:
            print('create model.')
            model = get_segmentation_model(model=args.model, 
                                           dataset=args.dataset,
                                           backbone=args.backbone, 
                                           norm_layer=args.norm_layer,
                                           norm_kwargs=args.norm_kwargs, 
                                           aux=args.aux,
                                           crop_size=args.crop_size,
                                           pretrained=False)
            if self.two_model:
                self.model2 = get_segmentation_model(model=args.model, 
                                               dataset=args.dataset,
                                               backbone=args.backbone, 
                                               norm_layer=args.norm_layer,
                                               norm_kwargs=args.norm_kwargs, 
                                               aux=args.aux,
                                               crop_size=args.crop_size,
                                               pretrained=False)

        model.cast(args.dtype)
        if self.two_model:
            self.model2.cast(args.dtype)
        # print(model) # don't print model
        # print(help(model.collect_params))
        # >>> Notice here <<<
        # model.initialize() # horse ref: https://discuss.mxnet.io/t/object-detection-transfer-learning/2477/2

        ''' '''
        self.net = DataParallelModel(model, args.ctx, args.syncbn)
        self.evaluator = DataParallelModel(SegEvalModel(model), args.ctx)

        if self.two_model:
            self.evaluator2 = DataParallelModel(SegEvalModel(self.model2), args.ctx)
        
        # resume checkpoint if needed
        if args.resume is not None:
            if os.path.isfile(args.resume):
                if not horse_changed:
                    model.load_parameters(args.resume, ctx=args.ctx)
                if horse_changed:
                    model.load_parameters(args.resume, ctx=args.ctx, allow_missing=True, ignore_extra=True)
            else:
                raise RuntimeError("=> no checkpoint found at '{}'" \
                    .format(args.resume))

        ''' 
        self.net = DataParallelModel(model, args.ctx, args.syncbn)
        self.evaluator = DataParallelModel(SegEvalModel(model), args.ctx)
        '''

        # create criterion
        criterion = MixSoftmaxCrossEntropyLoss(args.aux, aux_weight=args.aux_weight)
        self.criterion = DataParallelCriterion(criterion, args.ctx, args.syncbn)

        # optimizer and lr scheduling
        self.lr_scheduler = LRScheduler(mode='poly', 
                                        baselr=args.lr,
                                        niters=len(self.train_data), 
                                        nepochs=args.epochs)

        kv = mx.kv.create(args.kvstore)
        optimizer_params = {'lr_scheduler': self.lr_scheduler,
                            'wd':args.weight_decay,
                            'momentum': args.momentum}
        
        if args.dtype == 'float16':
            optimizer_params['multi_precision'] = True

        if args.no_wd:
            for k, v in self.net.module.collect_params('.*beta|.*gamma|.*bias').items():
                v.wd_mult = 0.0

        self.optimizer = gluon.Trainer(self.net.module.collect_params(), 
                                       'sgd',
                                       optimizer_params, 
                                       kvstore = kv)
        # evaluation metrics
        self.metric = gluoncv.utils.metrics.SegmentationMetric(trainset.num_class)

    def training(self, epoch):
        if self.two_model: 
            if self.two_model:
                self.model2.load_parameters('runs/pascal_voc/deeplab/HVSMR/res50_backup.params', ctx=args.ctx) # args.resume
                self.model2.cast(args.dtype)
                self.evaluator2 = DataParallelModel(SegEvalModel(self.model2), args.ctx)

        if horse_changed:
            print('>>> start training.') # [horse]
            tbar = tqdm(self.train_data)
            train_loss = 0.0
            alpha = 0.2
            for i, (data, target) in enumerate(tbar):
                self.lr_scheduler.update(i, epoch)
                with autograd.record(True):
                    # >>>>>>>>>>>>>>>>>>>>
                    global print_shape
                    if print_shape:
                        print('>>> data of one batch:')
                        print(data.shape, target.shape) # horse
                        '''
                        with open('have_a_look.pkl', 'wb') as fo:
                            pickle.dump(data.asnumpy(), fo)
                            pickle.dump(target.asnumpy(), fo)
                        '''
                        for ii in range(data.shape[1]): 
                            one_sample = data[0,ii,:,:].asnumpy()
                            s_mean = np.mean(one_sample.flatten())
                            s_std = np.std(one_sample.flatten())
                            s_min = min(one_sample.flatten())
                            s_max = max(one_sample.flatten())
                            print('dim | mean | std | min | max', ii, s_mean, s_std, s_min, s_max)
                        print_shape = False
                    # >>>>>>>>>>>>>>>>>>>>
                    outputs = self.net(data.astype(args.dtype, copy=False))
                    # print('outputs:', len(outputs[0]), outputs[0][0].shape) # [horse]
                    # print('target:', target.shape)
                    # outputs: 2 (14, 3, 250, 250)
                    # target: (14, 250, 250)

                    # +++++ +++++ +++++
                    _outputs = outputs
                    _target = mx.ndarray.reshape(target, shape=(-3,-2)) # to be (batch_size*NUM_SEQ, 250, 250)
                    # +++++ +++++ +++++

                    # losses = self.criterion(outputs, target)
                    losses = self.criterion(_outputs, _target)
                    mx.nd.waitall()
                    autograd.backward(losses)
                self.optimizer.step(self.args.batch_size)
                for loss in losses:
                    train_loss += loss.asnumpy()[0] / len(losses)
                tbar.set_description('Epoch %d, training loss %.3f'%\
                    (epoch, train_loss/(i+1)))
                mx.nd.waitall()

            # save every epoch
            save_checkpoint(self.net.module, self.args, False)
            # ++++++++++ ++++++++++ ++++++++++ ++++++++++ ++++++++++

        if not horse_changed:
            tbar = tqdm(self.train_data)
            train_loss = 0.0
            alpha = 0.2
            for i, (data, target) in enumerate(tbar):
                self.lr_scheduler.update(i, epoch)
                with autograd.record(True):
                    outputs = self.net(data.astype(args.dtype, copy=False))
                    # print('target:', target.shape) # target: (4, 480, 480)
                    ## print('target sum before:', [i.sum() for i in target.asnumpy()]) # target sum: [389344.0, 0.0, 0.0, 188606.0]

                    # ++++++++++ ++++++++++ ++++++++++ ++++++++++ ++++++++++
                    # ++++++++++ ++++++++++ ++++++++++ ++++++++++ ++++++++++                    
                    if self.semi:                        
                        pos = np.where(np.array([i.sum() for i in target.asnumpy()])==0)[0]
                        ## print('pos',pos)
                        if len(pos) != 0:
                            data2 = data[pos,:,:,:]
                            _outputs = self.evaluator2(data2.astype(args.dtype, copy=False))
                            _outputs = [x[0] for x in _outputs]  
                            label_generated = np.zeros((len(pos),target.shape[1],target.shape[2])) 
                            for k in range(len(pos)): 
                                ## print(_outputs[0].shape)
                                label_slice = labeler_random(_outputs[0].asnumpy()[k,0:3,:,:],
                                                             crop_size=target.shape[1],
                                                             prob_cut=0.46)
                                label_generated[k,:,:] = label_slice                        
                            target[pos,:,:] = mx.nd.array(label_generated)
                            ## print('target sum after:', [i.sum() for i in target.asnumpy()])  
                    '''         
                    if True:
                        # print('targets and outputs shape:', len(outputs), outputs[0].shape) # outputs: 1 (18, 3, 250, 250); targets: 1 (18, 250, 250)                        
                        for sample in range(2):
                            mx2img(data[sample,:,:,:], str(sample)+'.jpg')
                            mx2img(target[sample,:,:], str(sample)+'.png')       
                    '''
                    # ++++++++++ ++++++++++ ++++++++++ ++++++++++ ++++++++++
                    # ++++++++++ ++++++++++ ++++++++++ ++++++++++ ++++++++++

                    losses = self.criterion(outputs, target)
                    mx.nd.waitall()
                    autograd.backward(losses)
                self.optimizer.step(self.args.batch_size)
                for loss in losses:
                    train_loss += loss.asnumpy()[0] / len(losses)
                tbar.set_description('Epoch %d, training loss %.3f'%\
                    (epoch, train_loss/(i+1)))
                mx.nd.waitall()

            # save every epoch
            save_checkpoint(self.net.module, self.args, False)
            # ++++++++++ ++++++++++ ++++++++++ ++++++++++ ++++++++++
        ''' <- this is backup
        if not horse_changed:
            tbar = tqdm(self.train_data)
            train_loss = 0.0
            alpha = 0.2
            for i, (data, target) in enumerate(tbar):
                self.lr_scheduler.update(i, epoch)
                with autograd.record(True):
                    outputs = self.net(data.astype(args.dtype, copy=False))
                    losses = self.criterion(outputs, target)
                    mx.nd.waitall()
                    autograd.backward(losses)
                self.optimizer.step(self.args.batch_size)
                for loss in losses:
                    train_loss += loss.asnumpy()[0] / len(losses)
                tbar.set_description('Epoch %d, training loss %.3f'%\
                    (epoch, train_loss/(i+1)))
                mx.nd.waitall()

            # save every epoch
            save_checkpoint(self.net.module, self.args, False)
            # ++++++++++ ++++++++++ ++++++++++ ++++++++++ ++++++++++
        ''' 

    def validation(self, epoch):
        if not horse_changed:
            output_to_see = False # False # [horse added]
            output_score_map = False # [horse added]
            #total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
            self.metric.reset()
            tbar = tqdm(self.eval_data)

            output_index = 0 # [horse added]
            for i, (data, target) in enumerate(tbar):
                # print('target', target)
                outputs = self.evaluator(data.astype(args.dtype, copy=False))
                outputs = [x[0] for x in outputs]
                # print(outputs)
                '''
                if i == 50:
                    with open('have_a_look.pkl', 'wb') as fo:
                        pickle.dump(outputs[0].asnumpy(),fo)
                '''
                targets = mx.gluon.utils.split_and_load(target, args.ctx, even_split=False)

                # ++++++++++ ++++++++++ ++++++++++
                if output_to_see:
                    # print('targets and outputs shape:', len(outputs), outputs[0].shape) # outputs: 1 (18, 3, 250, 250); targets: 1 (18, 250, 250)
                    output_prefix = 'outdir_tosee'
                    if not os.path.exists(output_prefix):
                        os.makedirs(output_prefix)                    
                    batch_size = self.args.batch_size
                    crop_size = self.args.crop_size
                    
                    for sample in range(batch_size):
                        path = os.path.join(output_prefix, str(output_index)+'.png')
                        mx2img(outputs[0][sample, :,:,:], path)
                        output_index += 1
                # ++++++++++ ++++++++++ ++++++++++
                if output_score_map:
                    score_map_dir = 'scoredir_tosee' # args.scoredir
                    if not os.path.exists(score_map_dir):
                        os.makedirs(score_map_dir)

                    batch_size = self.args.batch_size
                    for sample in range(batch_size):
                        # score_map_name = os.path.splitext(impath)[0] + '.pkl'
                        # score_map_path = os.path.join(score_map_dir, score_map_name)
                        score_map_path = os.path.join(score_map_dir, str(output_index)+'.pkl')
                        with open(score_map_path, 'wb') as fo:
                            pickle.dump(outputs[0].asnumpy()[sample,0:3,:,:], fo)
                        output_index += 1

                self.metric.update(targets, outputs)
                '''
                pixAcc, mIoU = self.metric.get()
                tbar.set_description('Epoch %d, validation pixAcc: %.3f, mIoU: %.3f'%\
                    (epoch, pixAcc, mIoU))
                '''
                pixAcc, mIoU, dice = self.metric.get() # [horse changed]
                tbar.set_description('Epoch %d, validation pixAcc: %.3f, mIoU: %.3f, dice: %.3f, %.3f, %.3f'%\
                    (epoch, pixAcc, mIoU, dice[0], dice[1], dice[2]))

                mx.nd.waitall()

        if horse_changed:
            output_to_see = True # False
            #total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
            self.metric.reset()
            tbar = tqdm(self.eval_data)

            output_index = 0
            for i, (data, target) in enumerate(tbar):
                # print('target', target)
                outputs = self.evaluator(data.astype(args.dtype, copy=False))
                outputs = [x[0] for x in outputs]
                
                _target = mx.ndarray.reshape(target, shape=(-3,-2))
                targets = mx.gluon.utils.split_and_load(_target, args.ctx, even_split=False)
                
                # ++++++++++ ++++++++++ ++++++++++
                if output_to_see:
                    # print('targets and outputs shape:', len(outputs), outputs[0].shape) # outputs: 1 (18, 3, 250, 250); targets: 1 (18, 250, 250)
                    output_prefix = 'outdir_seq'
                    batch_size = self.args.batch_size
                    crop_size = self.args.crop_size
                    NUM_SEQ = int(outputs[0].shape[0] / batch_size)
                    # print(batch_size, NUM_SEQ, crop_size)

                    outputs_out = mx.ndarray.reshape(outputs[0], shape=(batch_size, NUM_SEQ, 3, crop_size, crop_size)) # 3 is the class number not image channel, just for convenience 
                    targets_out = mx.ndarray.reshape(targets[0], shape=(batch_size, NUM_SEQ, crop_size, crop_size))

                    for sample in range(batch_size):
                        for seq in range(NUM_SEQ):
                            path = os.path.join(output_prefix, str(output_index)+'_'+str(seq)+'.png')
                            path_mask = os.path.join(output_prefix, str(output_index)+'_gt_'+str(seq)+'.png')
                            
                            mx2img(outputs_out[sample, seq, :,:,:], path)
                            mx2img(targets_out[sample, seq, :,:], path_mask)
                        output_index += 1
                # ++++++++++ ++++++++++ ++++++++++

                self.metric.update(targets, outputs)
                '''
                pixAcc, mIoU = self.metric.get()
                tbar.set_description('Epoch %d, validation pixAcc: %.3f, mIoU: %.3f'%\
                    (epoch, pixAcc, mIoU))
                '''
                pixAcc, mIoU, dice = self.metric.get() # [horse changed]
                tbar.set_description('Epoch %d, validation pixAcc: %.3f, mIoU: %.3f, dice: %.3f, %.3f, %.3f'%\
                    (epoch, pixAcc, mIoU, dice[0], dice[1], dice[2]))                
                
                mx.nd.waitall()    

                # break        

def save_checkpoint(net, args, is_best=False): # [horse added params name]
    """Save Checkpoint"""
    directory = "runs/%s/%s/%s/" % (args.dataset, args.model, args.checkname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # filename='checkpoint.params'
    filename = args.save_name+'.params'
    filename = directory + filename
    net.save_parameters(filename)
    if is_best:
        shutil.copyfile(filename, directory + 'model_best.params')

def save_checkpoint_old(net, args, is_best=False):
    """Save Checkpoint"""
    directory = "runs/%s/%s/%s/" % (args.dataset, args.model, args.checkname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename='checkpoint.params'
    filename = directory + filename
    net.save_parameters(filename)
    if is_best:
        shutil.copyfile(filename, directory + 'model_best.params')

from PIL import Image
def mx2img(a, path):
    r"""
    save mxnet array to image
    """    
    # print(img.shape)
    if len(a.shape) == 3:
        img = a.asnumpy()[0:3,:,:] # convert to numpy array
        img = img.transpose((1, 2, 0))  # Move channel to the last dimension
        result = Image.fromarray((img * 255).astype(np.uint8)) # 10 here and 100 below are for illustration
    else:
        img = a.asnumpy()
        result = Image.fromarray((img * 100).astype(np.uint8))

    result.save(path)

def labeler_random(pred,crop_size=480,prob_cut=0.5):
    """
    pred: prediction matrix (3, 480, 480)
    input: numpy; output: numpy.
    """
    pred[pred<0] = 0
    pred /= pred[0,:,:]+pred[1,:,:]+pred[2,:,:]
    
    # generate random mask
    random_mask = np.random.rand(crop_size,crop_size)
    
    # return this    
    label = np.zeros((crop_size,crop_size))
    
    # random labeler
    label[random_mask<pred[0,:,:]] = 0
    label[random_mask>(pred[0,:,:]+pred[1,:,:])] = 2    
    label[np.logical_and(random_mask>pred[0,:,:],random_mask<(pred[0,:,:]+pred[1,:,:]))] = 1
    
    # high prob
    # prob_cut = 0.5
    label[pred[0,:,:]>prob_cut] = 0
    label[pred[1,:,:]>prob_cut] = 1
    label[pred[2,:,:]>prob_cut] = 2
    
    return label

if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    if args.eval:
        print('Evaluating model: ', args.resume)
        trainer.validation(args.start_epoch)
    else:
        print('Starting Epoch:', args.start_epoch)
        print('Total Epochs:', args.epochs)
        for epoch in range(args.start_epoch, args.epochs):
            trainer.training(epoch)
            if not trainer.args.no_val:
                trainer.validation(epoch)
