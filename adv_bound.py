from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import config as cf
import numpy as np
import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import argparse
import datetime

from networks import *
from torch.autograd import Variable
from measures import *

def validate(args, model, device, val_loader, criterion):
    sum_loss, sum_correct = 0, 0
    margin = torch.Tensor([]).to(device)

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)

            # compute the output
            output = model(data)

            # compute the classification error and loss
            pred = output.max(1)[1]
            sum_correct += pred.eq(target).sum().item()
            sum_loss += len(data) * criterion(output, target).item()

            # compute the margin
            output_m = output.clone()
            for i in range(target.size(0)):
                output_m[i, target[i]] = output_m[i,:].min()
            margin = torch.cat((margin, output[:, target].diag() - output_m[:, output_m.max(1)[1]].diag()), 0)
        val_margin = np.percentile( margin.cpu().numpy(), 5 )

    return 1 - (sum_correct / len(val_loader.dataset)), sum_loss / len(val_loader.dataset), val_margin

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--net_type', default='vggnet', type=str, help='model')
parser.add_argument('--depth', default=13, type=int, help='depth of model')
parser.add_argument('--no-cuda', default=False, action='store_true',
                    help='disables CUDA training')
parser.add_argument('--datadir', default='datasets', type=str,
                    help='path to the directory that contains the datasets (default: datasets)')
parser.add_argument('--dataset', default='cifar100', type=str,
                    help='name of the dataset (options: MNIST | CIFAR10 | CIFAR100 | SVHN, default: CIFAR10)')
parser.add_argument('--model', default='vgg', type=str,
                    help='architecture (options: fc | vgg, default: vgg)')
parser.add_argument('--epochs', default=1000, type=int,
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--stopcond', default=0.01, type=float,
                    help='stopping condtion based on the cross-entropy loss (default: 0.01)')
parser.add_argument('--batchsize', default=64, type=int,
                    help='input batch size (default: 64)')
parser.add_argument('--learningrate', default=0.01, type=float,
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum (default: 0.9)')
parser.add_argument('--weight_decay', default=5e-4, type = float, help='adversary type')
parser.add_argument('--nosamples', default=10000, type = float, help='adversary type')
args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

nchannels, nclasses, img_dim,  = 3, 10, 32
if args.dataset == 'MNIST': nchannels = 1
if args.dataset == 'CIFAR100': nclasses = 100

# Data Uplaod
print('\n[Phase 1] : Data Preparation')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

if(args.dataset == 'cifar10'):
    print("| Preparing CIFAR-10 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_test)
    num_classes = 10
elif(args.dataset == 'cifar100'):
    print("| Preparing CIFAR-100 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=False, transform=transform_test)
    num_classes = 100

testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Return network & file name
def getNetwork(args):
    if (args.net_type == 'lenet'):
        net = LeNet(num_classes)
        file_name = 'lenet'
    elif (args.net_type == 'vggnet'):
        if args.dataset == 'cifar100':
            net = VGG100(args.depth, num_classes)
        else:
            net = VGG(args.depth, num_classes)
        file_name = 'vgg-'+str(args.depth)
    elif (args.net_type == 'resnet'):
        net = ResNet(args.depth, num_classes)
        file_name = 'resnet-'+str(args.depth)
    elif (args.net_type == 'wide-resnet'):
        net = Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes)
        file_name = 'wide-resnet-'+str(args.depth)+'x'+str(args.widen_factor)
    else:
        print('Error : Network should be either [LeNet / VGGNet / ResNet / Wide_ResNet')
        sys.exit(0)

    return net, file_name

for args.depth in [11,13,16,19]:
    init_model, file_name = getNetwork(args)
    init_model.cuda()
    init_model.eval()

    for std in ['std', 'adv']:

        margin1_list = torch.tensor([])
        margin3_list = torch.tensor([])
        margin5_list = torch.tensor([])
        tr_error_list = torch.tensor([])
        test_error_list = torch.tensor([])
        linfnorm_list = torch.tensor([])
        fronorm_list = torch.tensor([])

        for args.nosamples in [10000.0, 20000.0, 30000.0, 40000.0, 50000.0]:


            subtrainset = subdataset(trainset = trainset, dataset =args.dataset, nosamples=args.nosamples)
            trainloader = torch.utils.data.DataLoader(subtrainset, batch_size=128, shuffle=False, num_workers=2)
            checkpoint = torch.load('../vggcheckpoint/'+std+args.dataset+file_name+'no_samples'+str(args.nosamples)+'_'+str(args.weight_decay)+'.pt')
            model = checkpoint['net']
            model.eval()
            margin1 = checkpoint['margin1']
            margin3 = checkpoint['margin3']
            margin5 = checkpoint['margin5']
            tr_error = checkpoint['tr_error']
            test_error = checkpoint['test_error']

            measure_dict, bound_dict = calculate(model, init_model, device, trainloader, 1, nchannels, nclasses, img_dim)
            linfnorm = measure_dict['L_{1,inf} norm']
            fronorm = measure_dict['Frobenious norm']

            margin1_list = torch.cat((margin1_list,torch.tensor([margin1])), dim = 0)
            margin3_list = torch.cat((margin3_list,torch.tensor([margin3])), dim = 0)
            margin5_list = torch.cat((margin5_list,torch.tensor([margin5])), dim = 0)
            tr_error_list = torch.cat((tr_error_list,torch.tensor([tr_error])), dim = 0)
            test_error_list = torch.cat((test_error_list,torch.tensor([test_error])), dim = 0)
            linfnorm_list = torch.cat((linfnorm_list,torch.tensor([linfnorm])), dim = 0)
            fronorm_list = torch.cat((fronorm_list,torch.tensor([fronorm])), dim = 0)


        np.save('./vggbound100/'+std+args.dataset+file_name+'_'+str(args.weight_decay)+'margin1_list.npy',margin1_list.numpy())
        np.save('./vggbound100/'+std+args.dataset+file_name+'_'+str(args.weight_decay)+'margin3_list.npy',margin3_list.numpy())
        np.save('./vggbound100/'+std+args.dataset+file_name+'_'+str(args.weight_decay)+'margin5_list.npy',margin5_list.numpy())
        np.save('./vggbound100/'+std+args.dataset+file_name+'_'+str(args.weight_decay)+'tr_error_list.npy',tr_error_list.numpy())
        np.save('./vggbound100/'+std+args.dataset+file_name+'_'+str(args.weight_decay)+'test_error_list.npy',test_error_list.numpy())
        np.save('./vggbound100/'+std+args.dataset+file_name+'_'+str(args.weight_decay)+'linfnorm_list.npy',linfnorm_list.numpy())
        np.save('./vggbound100/'+std+args.dataset+file_name+'_'+str(args.weight_decay)+'fronorm_list.npy',fronorm_list.numpy())
