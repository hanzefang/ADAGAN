import torch
import torch.nn as nn
from torch.nn import init
import functools
#import ops
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torchvision import models
from scipy import misc
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from collections import OrderedDict
import math

class Bottleneck(nn.Module):
    def __init__(self,in_channels,out_channels,):
        super(Bottleneck,self).__init__()
        m  = OrderedDict()
        m['conv1'] = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        m['relu1'] = nn.ReLU(True)
        m['conv2'] = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, bias=False,dilation=2)
        m['relu2'] = nn.ReLU(True)
        m['conv3'] = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.group1 = nn.Sequential(m)
        self.relu= nn.Sequential(nn.ReLU(True))

    def forward(self, x):
        out = self.group1(x) 
        return out

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)



class BottleneckBlock1(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock1, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=5, stride=1,
                               padding=2, bias=False)
        self.droprate = dropRate
    def forward(self, input_cpu):
        out = self.conv1(self.relu(self.bn1(input_cpu)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([input_cpu, out], 1)



class BottleneckBlock2(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock2, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=7, stride=1,
                               padding=3, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.upsample_nearest(out, scale_factor=2)



class TransitionBlock1(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock1, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)



class TransitionBlock3(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock3, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out
def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'switchable':
        norm_layer = SwitchNorm2d
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


# update learning rate (called once every epoch)
def update_learning_rate(scheduler, optimizer):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print('learning rate = %.7f' % lr)


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                m.weight.data.normal_(0.0, 0.02)
                #init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            #init.normal_(m.weight.data, 1.0, gain)
            m.weight.data.normal_(1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
    net.to(gpu_id)
    init_weights(net, init_type, gain=init_gain)
    return net

def define_D(init_type='normal', init_gain=0.02, gpu_id='cuda:0'):   
    #net = MultiScaleDiscriminator()
    net = Discriminator()
    return init_net(net,init_type, init_gain, gpu_id)


def define_G(init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
    net = Dense_rain()  
    return init_net(net,init_type, init_gain, gpu_id)

class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0], 
                    [0, 0, 0], 
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0], 
                    [-1, 0, 1], 
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False)
        
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False)
        

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        x = torch.cat(x_list, dim = 1)
        
        return x

class Dense_rain(nn.Module):
    def __init__(self):
        super(Dense_rain, self).__init__()
        # self.conv_refin=nn.Conv2d(9,20,3,1,1)
        #self.conv_refin=nn.Conv2d(47,47,3,1,1)
        self.conv_refin=nn.Conv2d(40,47,3,1,1)
        self.tanh=nn.Tanh()


        self.conv1010 = nn.Conv2d(47, 2, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(47, 2, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(47, 2, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(47, 2, kernel_size=1,stride=1,padding=0)  # 1mm

        self.refine3= nn.Conv2d(47+8, 3, kernel_size=3,stride=1,padding=1)

        self.refineclean1= nn.Conv2d(6, 8, kernel_size=7,stride=1,padding=3)
        self.refineclean2= nn.Conv2d(8, 3, kernel_size=3,stride=1,padding=1)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)


        self.batchnorm20=nn.BatchNorm2d(20)
        self.batchnorm1=nn.BatchNorm2d(1)

        self.dense0=Dense_base_down0()
        self.dense1=Dense_base_down1()
        self.dense2=Dense_base_down2()
        self.Generator=Generator()
        self.attloss=ATTLoss()
        #self.detail_repair_network=Detail_repair_network()
        #self.self_attn=Self_Attn()
        #self.calayer = CALayer()
        self.get_gradient=Get_gradient_nopadding()

    def forward(self, x):
        mask,mask_list = self.Generator(x)
        
        x3=self.dense2(x)
        x2=self.dense1(x)
        x1=self.dense0(x)

        shape_out = x3.data.size()
        sizePatchGAN = shape_out[3]
        x8=torch.cat([x1,x2,x3,x,mask],1)
        #x8=torch.cat([x1,x,x2,x3],1)
        # print(x8.size())

        x9=self.relu((self.conv_refin(x8)))

        shape_out = x9.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(x9, 32)
        x102 = F.avg_pool2d(x9, 16)
        x103 = F.avg_pool2d(x9, 8)
        x104 = F.avg_pool2d(x9, 4)

        x1010 = self.upsample(self.relu((self.conv1010(x101))), size=shape_out)
        x1020 = self.upsample(self.relu((self.conv1020(x102))), size=shape_out)
        x1030 = self.upsample(self.relu((self.conv1030(x103))), size=shape_out)
        x1040 = self.upsample(self.relu((self.conv1040(x104))), size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        residual = self.tanh(self.refine3(dehaze))
        clean = x-residual
        gradient = self.get_gradient(x)
        #detail = self.detail_repair_network(gradient)
        #clean = clean + detail
        #mask1,mask_list1 = self.Generator(clean)
        #clean = self.calayer(clean)
        clean = torch.cat((gradient, clean), 1)
        clean1=self.relu(self.refineclean1(clean))
        clean2=self.tanh(self.refineclean2(clean1))
        return residual,gradient,clean2,mask,mask_list



class Dense_base_down2(nn.Module):
    def __init__(self):
        super(Dense_base_down2, self).__init__()

        self.dense_block1=BottleneckBlock2(3,13)
        self.trans_block1=TransitionBlock1(16,8)

        ############# Block2-down 32-32  ##############
        self.dense_block2=BottleneckBlock2(8,16)
        self.trans_block2=TransitionBlock1(24,16)

        ############# Block3-down  16-16 ##############
        self.dense_block3=BottleneckBlock2(16,16)
        self.trans_block3=TransitionBlock1(32,16)

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock2(16,16)
        self.trans_block4=TransitionBlock(32,16)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock2(32,8)
        self.trans_block5=TransitionBlock(40,8)

        self.dense_block6=BottleneckBlock2(16,8)
        self.trans_block6=TransitionBlock(24,4)


        self.conv_refin=nn.Conv2d(11,20,3,1,1)
        self.tanh=nn.Tanh()


        self.conv11 = nn.Conv2d(8, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv21 = nn.Conv2d(16, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv31 = nn.Conv2d(16, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv41 = nn.Conv2d(32, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv51 = nn.Conv2d(16, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv61 = nn.Conv2d(8, 1, kernel_size=3,stride=1,padding=1)  # 1mm

        self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)
        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)


        self.batchnorm20=nn.BatchNorm2d(20)
        self.batchnorm1=nn.BatchNorm2d(1)



    def forward(self, x):
        ## 256x256
        x1=self.dense_block1(x)
        x1=self.trans_block1(x1)

        ###  32x32
        x2=(self.dense_block2(x1))
        x2=self.trans_block2(x2)

        # print x2.size()
        ### 16 X 16
        x3=(self.dense_block3(x2))
        x3=self.trans_block3(x3)

        ## Classifier  ##

        x4=(self.dense_block4(x3))
        x4=self.trans_block4(x4)

        # x4=x4+x2
        x4=torch.cat([x4,x2],1)

        x5=(self.dense_block5(x4))
        x5=self.trans_block5(x5)

        # x5=x5+x1
        x5=torch.cat([x5,x1],1)

        x6=(self.dense_block6(x5))
        x6=(self.trans_block6(x6))

        shape_out = x6.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]


        x11 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x21 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x31 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x41 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x51 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)


        x6=torch.cat([x6,x51,x41,x31,x21,x11,x],1)

        return x6


class Dense_base_down1(nn.Module):
    def __init__(self):
        super(Dense_base_down1, self).__init__()

        self.dense_block1=BottleneckBlock1(3,13)
        self.trans_block1=TransitionBlock1(16,8)

        ############# Block2-down 32-32  ##############
        self.dense_block2=BottleneckBlock1(8,16)
        self.trans_block2=TransitionBlock1(24,16)

        ############# Block3-down  16-16 ##############
        self.dense_block3=BottleneckBlock1(16,16)
        self.trans_block3=TransitionBlock3(32,16)

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock1(16,16)
        self.trans_block4=TransitionBlock3(32,16)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock1(32,8)
        self.trans_block5=TransitionBlock(40,8)

        self.dense_block6=BottleneckBlock1(16,8)
        self.trans_block6=TransitionBlock(24,4)


        self.conv_refin=nn.Conv2d(11,20,3,1,1)
        self.tanh=nn.Tanh()


        self.conv11 = nn.Conv2d(8, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv21 = nn.Conv2d(16, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv31 = nn.Conv2d(16, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv41 = nn.Conv2d(32, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv51 = nn.Conv2d(16, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv61 = nn.Conv2d(8, 1, kernel_size=3,stride=1,padding=1)  # 1mm

        self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)
        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)


        self.batchnorm20=nn.BatchNorm2d(20)
        self.batchnorm1=nn.BatchNorm2d(1)



    def forward(self, x):
        ## 256x256
        x1=self.dense_block1(x)
        x1=self.trans_block1(x1)

        ###  32x32
        x2=(self.dense_block2(x1))
        x2=self.trans_block2(x2)

        # print x2.size()
        ### 16 X 16
        x3=(self.dense_block3(x2))
        x3=self.trans_block3(x3)

        ## Classifier  ##

        x4=(self.dense_block4(x3))
        x4=self.trans_block4(x4)

        x4=torch.cat([x4,x2],1)

        x5=(self.dense_block5(x4))
        x5=self.trans_block5(x5)

        # x5=x5+x1
        x5=torch.cat([x5,x1],1)

        x6=(self.dense_block6(x5))
        x6=(self.trans_block6(x6))

        shape_out = x6.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]
        x11 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x21 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x31 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x41 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x51 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)



        x6=torch.cat([x6,x51,x41,x31,x21,x11,x],1)

        return x6

class Dense_base_down0(nn.Module):
    def __init__(self):
        super(Dense_base_down0, self).__init__()

        self.dense_block1=BottleneckBlock(3,5)
        self.trans_block1=TransitionBlock1(8,4)

        ############# Block2-down 32-32  ##############
        self.dense_block2=BottleneckBlock(4,8)
        self.trans_block2=TransitionBlock3(12,12)

        ############# Block3-down  16-16 ##############
        self.dense_block3=BottleneckBlock(12,4)
        self.trans_block3=TransitionBlock3(16,12)

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock(12,4)
        self.trans_block4=TransitionBlock3(16,12)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock(24,8)
        self.trans_block5=TransitionBlock3(32,4)

        self.dense_block6=BottleneckBlock(8,8)
        self.trans_block6=TransitionBlock(16,4)

        self.conv11 = nn.Conv2d(4, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv21 = nn.Conv2d(12, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv31 = nn.Conv2d(12, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv41 = nn.Conv2d(24, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv51 = nn.Conv2d(8, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv61 = nn.Conv2d(8, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.upsample = F.upsample_nearest
        self.relu=nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        ## 256x256
        x1=self.dense_block1(x)
        x1=self.trans_block1(x1)

        ###  32x32
        x2=(self.dense_block2(x1))
        x2=self.trans_block2(x2)

        # print x2.size()
        ### 16 X 16
        x3=(self.dense_block3(x2))
        x3=self.trans_block3(x3)

        ## Classifier  ##

        x4=(self.dense_block4(x3))
        x4=self.trans_block4(x4)

        x4=torch.cat([x4,x2],1)

        x5=(self.dense_block5(x4))
        x5=self.trans_block5(x5)

        # x5=x5+x1
        x5=torch.cat([x5,x1],1)

        x6=(self.dense_block6(x5))
        x6=(self.trans_block6(x6))
        shape_out = x6.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]
        x11 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x21 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x31 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x41 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x51 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)



        x6=torch.cat([x6,x51,x41,x31,x21,x11,x],1)


        return x6


ITERATION = 4
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.det_conv0 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace=True)
            )
        self.det_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace=True)
            )
        self.det_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True)
            )
        self.det_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True)
            )
        self.det_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.det_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True)
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.det_conv_mask = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1)
            )
            

    def forward(self,x):
        batch_size, channel , row, col = x.size(0), x.size(1),x.size(2), x.size(3)#1,3,512,512
        #mask = Variable(torch.ones(batch_size, 1, row, col)).cuda()#1,1,512,512
        mask = Variable(torch.ones(batch_size, 1, row, col)).cuda() / 2.#1,1,512,512
        h = Variable(torch.zeros(batch_size, 32, row, col)).cuda()#1,32,512,512 
        c = Variable(torch.zeros(batch_size, 32, row, col)).cuda()#1,32,512,512
        mask_list = []
        for i in range(ITERATION):
            x1 = torch.cat([x, mask], 1)#1,4,512,512
            x1 = self.det_conv0(x1)
            resx = x1
            x1 = F.relu(self.det_conv1(x1) + resx)
            resx = x1
            x1 = F.relu(self.det_conv2(x1) + resx)
            resx = x1
            x1 = F.relu(self.det_conv3(x1) + resx)
            resx = x1
            x1 = F.relu(self.det_conv4(x1) + resx)
            resx = x1
            x1 = F.relu(self.det_conv5(x1) + resx)
            #resx = x1
            #x1 = F.relu(self.det_conv6(x1) + resx)
            x1 = torch.cat((x1, h), 1)
            i = self.conv_i(x1)
            f = self.conv_f(x1)
            g = self.conv_g(x1)
            o = self.conv_o(x1)
            c = f * c + i * g
            h = o * F.tanh(c)
            mask = self.det_conv_mask(h)
            mask_list.append(mask)      
        #return mask,loss
        return mask,mask_list



class ATTLoss(nn.Module):
    def __init__(self, gpu_ids = '0'):
        super(ATTLoss, self).__init__()        
        self.generator = Generator().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0] 
        #self.weights = [2, 4, 6, 8]       

    def forward(self,weiyingreal,mask,mask_list):             
        #mask,mask_list = self.generator(x)
        #weiying = x-y
        weiying = weiyingreal.data.cpu().numpy()
        #weiying = rgb2kgray(weiying)
        thresh = threshold_otsu(weiying)
        weiying = weiying>thresh
        weiying = weiying.astype(np.float32)
        weiying = torch.from_numpy(weiying)
        weiying = weiying.data.cuda()
        loss = 0
        for i in range(len(mask_list)):
            loss += self.weights[i] * self.criterion(mask_list[i], weiying.detach())       
        return loss 

class Bottle2neck1(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=64, scale = 4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck1, self).__init__()
        self.conv11 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            )
        self.discrim_conv11 = nn.Sequential(
            nn.Conv2d(128, 64, 1, 1, 0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
            )
        width = int(math.floor(planes * (baseWidth/64.0)))#32×1
        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)#64,32×4,1
        self.bn1 = nn.BatchNorm2d(width*scale)#32×4
        
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1#num = 3
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride = stride, padding=1)
        convs = []
        bns = []#create null
        for i in range(self.nums):
          convs.append(nn.Conv2d(width, width, kernel_size=3, stride = stride, padding=1, bias=False))#32,32,3,1,1
          bns.append(nn.BatchNorm2d(width))#32
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width*scale, planes * self.expansion, kernel_size=1, bias=False)#128,64,1
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)#64

        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width  = width
        

    def forward(self, x):
        residual = x#256×256×64

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakyrelu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.leakyrelu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype=='normal':
          out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
          out = torch.cat((out, self.pool(spx[self.nums])),1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)#jump

        out += residual
        out = self.conv11(out)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.Bottle2neck1 = Bottle2neck1(64, 32)
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.discrim_conv2 = nn.Sequential(
            nn.Conv2d(128, 1, 4, 1, 1),
            nn.Sigmoid()
            )
        
    def forward(self, x):
        x = self.conv1(x)#256×256×64
        x = self.Bottle2neck1(x)
        x = self.discrim_conv2(x)
        return x


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock,self).__init__()
        self.add_module('conv',nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        self.add_module('norm',nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.cuda.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)
        

#TV loss(total variation regularizer)
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight
 
    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
 
    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


class VGGLoss(nn.Module):
    def __init__(self, gpu_ids = '0'):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):             
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss 


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out



    
