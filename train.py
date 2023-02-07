from __future__ import print_function
import argparse
import os
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from networks import define_G,define_D,Get_gradient_nopadding,GANLoss ,TVLoss,ATTLoss,get_scheduler, update_learning_rate 
from data import get_training_set, get_test_set
import time
import scipy.io as sio
import numpy as np
#from cal_ssim import SSIM

# Training settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', required=True, help='facades')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--num_D', type=int, default=1, help='number of discriminators to use')
parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb1', type=int, default=25, help='weight on L1 term in objective')
parser.add_argument('--lamb2', type=int, default=10, help='weight on SSIM term in objective')
parser.add_argument('--lamb3', type=int, default=0.1, help='weight on VGG term in objective')
#parser.add_argument('--lamb4', type=int, default=10, help='weight on tv term in objective')
#parser.add_argument('--lamb5', type=int, default=10, help='weight on l1 term in objective')
opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
root_path = "dataset/"
train_set = get_training_set(root_path + opt.dataset, opt.direction)
test_set = get_test_set(root_path + opt.dataset, opt.direction)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False)

device = torch.device("cuda:0" if opt.cuda else "cpu")

print('===> Building models')
net_g = define_G('normal', '0.02', gpu_id=device)
total_params=sum(p.numel() for p in net_g.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params=sum(p.numel() for p in net_g.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} total parameters.')
net_d = define_D('normal', 0.02, gpu_id=device)
total_params=sum(p.numel() for p in net_d.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params=sum(p.numel() for p in net_d.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} total parameters.')
#Detail_repair_network1 = Detail_repair_network1().to(device)

criterionGAN = GANLoss().to(device)
#criterionVGG = VGGLoss().to(device)
criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)
criterionATT = ATTLoss().to(device)
criterionTV = TVLoss().to(device)
#ssim = SSIM().to(device)
get_gradient = Get_gradient_nopadding().to(device)

# setup optimizer
#optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_g = optim.Adam(filter(lambda p: p.requires_grad,net_g.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
net_g_scheduler = get_scheduler(optimizer_g, opt)
net_d_scheduler = get_scheduler(optimizer_d, opt)
lossl1=[]
lossl11=[]
lossg=[]
lossd=[]
lossatt=[]
losstv=[]
start=time.time()
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    # train
    lossdzong=lossgzong=lossl1zong=lossl11zong=lossattzong=losstvzong=0
    for iteration, batch in enumerate(training_data_loader, 1):
        # forward
        real_a, real_b = batch[0].to(device), batch[1].to(device)
        weiyingreal = real_a-real_b
        residual,gradient,clean2,mask,mask_list = net_g(real_a)
        gradientreal = get_gradient(real_b)
        weiyingfake = residual
        fake_b=clean2
        gradientfake1=gradient
        gradientfake2 = get_gradient(fake_b)
        ######################
        # (1) Update D network
        ######################

        optimizer_d.zero_grad()
        
        # train with fake
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab.detach())
        #pred_fake = net_d.forward(fake_ab)
        loss_d_fake = criterionGAN(pred_fake, False)

        # train with real
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = net_d.forward(real_ab.detach())
        loss_d_real = criterionGAN(pred_real, True)
        
        # Combined D loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5 
        loss_d.backward(retain_graph=True)
       
        optimizer_d.step()
        lossdzong=lossdzong+loss_d.item()
        ######################
        # (2) Update G network
        ######################

        optimizer_g.zero_grad()

        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)

        # Second, G(A) = B
        loss_g_l11 = criterionL1(weiyingfake, weiyingreal) 
        loss_g_gradient = criterionL1(gradientfake1,gradientreal)*opt.lamb2
        loss_g_l1 = criterionL1(fake_b, real_b) * opt.lamb2 + criterionL1(gradientfake2,gradientreal)*0.5
        #loss_g_l1 = criterionL1(fake_b, real_b)* opt.lamb5 
        #loss_g_vgg = criterionVGG(fake_b, real_b)* opt.lamb3 
        loss_g_att = criterionATT(weiyingreal,mask,mask_list)*opt.lamb1
        #ssim_loss = ssim(fake_b, real_b)
        #ssim_loss = 1-ssim_loss
        loss_g = loss_g_gan + loss_g_l1 + loss_g_att + loss_g_l11+loss_g_gradient
        #loss_g = loss_g_gan + loss_g_l1+ loss_g_att + loss_g_l11+loss_g_gradient
        loss_g.backward()

        optimizer_g.step()

        #print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G_l1: {:.4f} Loss_G_l11: {:.4f} Loss_G_vgg: {:.4f} Loss_G_att: {:.4f} Loss_G_gan: {:.4f}".format(
            #epoch, iteration, len(training_data_loader), loss_d.item(), loss_g_l1.item(),loss_g_l11.item(),loss_g_vgg.item(),loss_g_att.item(),loss_g_gan.item()))
        print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G_l1: {:.4f} Loss_G_l11: {:.4f} Loss_G_att: {:.4f} Loss_G_gan: {:.4f}".format(
            epoch, iteration, len(training_data_loader), loss_d.item(), loss_g_l1.item(),loss_g_l11.item(),loss_g_att.item(),loss_g_gan.item()))
        lossgzong=lossgzong+loss_g.item()
        lossl1zong=lossl1zong+loss_g_l1.item()
        lossattzong=lossattzong+loss_g_att.item()
        losstvzong=losstvzong+loss_g_gradient.item()
        lossl11zong=lossl11zong+loss_g_l11.item()
    lossgaver=lossgzong/len(training_data_loader)
    lossdaver=lossdzong/len(training_data_loader)
    lossl1aver=lossl1zong/len(training_data_loader)
    lossattaver=lossattzong/len(training_data_loader)
    losstvaver=losstvzong/len(training_data_loader)
    lossl11aver=lossl11zong/len(training_data_loader)
    lossd.append(lossdaver)
    lossg.append(lossgaver)
    lossl1.append(lossl1aver)
    lossatt.append(lossattaver)
    losstv.append(losstvaver)
    lossl11.append(lossl11aver)
    sio.savemat('twoattentionloss.mat',{'loss_g_l1':np.array(lossl1),'loss_d':np.array(lossd),'loss_g':np.array(lossg),'loss_g_att':np.array(lossatt),'loss_g_tv':np.array(losstv),'loss_g_l11':np.array(lossl11)})
    update_learning_rate(net_g_scheduler, optimizer_g)
    update_learning_rate(net_d_scheduler, optimizer_d)


    #checkpoint
    if epoch % 50 == 0:
        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")
        if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
            os.mkdir(os.path.join("checkpoint", opt.dataset))
        net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, epoch)
        #net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format(opt.dataset, epoch)
        torch.save(net_g, net_g_model_out_path)
        #torch.save(net_d, net_d_model_out_path)
        print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))
print(time.time()-start)
