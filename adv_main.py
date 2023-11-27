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
from adversarialbox.attacks import FGSMAttack, LinfPGDAttack, LtwoPGDattack
from adversarialbox.train import adv_train, FGSM_train_rnd
from adversarialbox.utils import to_var, pred_batch, test, attack_over_test_data
from measures import *
import os
import sys
import time
import argparse
import datetime
import copy
from networks import *
from torch.autograd import Variable

def validate(args, model, device, val_loader, criterion, adversary, std):
    sum_loss, sum_correct = 0, 0
    margin = torch.Tensor([]).to(device)

    # switch to evaluation mode
    # for p in model.parameters():
    #     p.requires_grad = False
    model.eval()
    # with torch.no_grad():
    for i, (data, target) in enumerate(val_loader):
        data, target = data.to(device), target.to(device)
        if std == False: 
            data, target = Variable(data), Variable(target)
            y_pred = pred_batch(data, model)
            data = adv_train(data, y_pred, model, criterion, adversary)
        with torch.no_grad():
            output = model(to_var(data))

            # compute the classification error and loss
            pred = output.max(1)[1]
            sum_correct += pred.eq(target).sum().item()
            sum_loss += len(data) * criterion(output, target).item()

            # compute the margin
            output_m = output.clone()
            for i in range(target.size(0)):
                output_m[i, target[i]] = output_m[i,:].min()
            margin = torch.cat((margin, output[:, target].diag() - output_m[:, output_m.max(1)[1]].diag()), 0)
    margin = margin.detach()
    val_margin1 = np.percentile( margin.cpu().numpy(), 1 )
    val_margin3 = np.percentile( margin.cpu().numpy(), 3 )
    val_margin5 = np.percentile( margin.cpu().numpy(), 5 )
    return 1 - (sum_correct / len(val_loader.dataset)), sum_loss / len(val_loader.dataset), val_margin1, val_margin3, val_margin5

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--net_type', default='vggnet', type=str, help='model')
parser.add_argument('--depth', default=19, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dropout', default=0, type=float, help='dropout_rate')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
parser.add_argument('--epsilon', default=8/255, type=float, help='perturbation')
parser.add_argument('--no_dim', default=1500, type=int, help='subspace dimension')
parser.add_argument('--manifold', action='store_true', help='on or off manifold')
parser.add_argument('--PGD_steps', default=20, type=int, help='PGD_steps')
parser.add_argument('--PGD_stepsize', default=2/255, type=float, help='PGD_stepsize')
parser.add_argument('--mixture', action='store_true', help='mixture')
parser.add_argument('--std', action='store_true', help='std training')
parser.add_argument('--attack', default='linf', help='adversary type')
parser.add_argument('--weight_decay', default=5e-4, type = float, help='adversary type')
parser.add_argument('--nosamples', default=50000, type = float, help='adversary type')
args = parser.parse_args()


# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
best_acc = 0
start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type
print(args)

if args.std == True:
    std = 'std'
else:
    std = 'adv'
device = torch.device("cuda" if use_cuda else "cpu")
# Data Uplaod
print('\n[Phase 1] : Data Preparation')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
]) # meanstd transformation

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

subtrainset = subdataset(trainset = trainset, dataset =args.dataset, nosamples=args.nosamples)
print(subtrainset.__len__())
trainloader = torch.utils.data.DataLoader(subtrainset, batch_size=batch_size, shuffle=True, num_workers=2)
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

# Test only option
if (args.testOnly):
    print('\n[Test Phase] : Model setup')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    checkpoint = torch.load('../wide-resnet.pytorch-master/checkpoint/'+args.dataset+file_name+'.pt')
    net = checkpoint['net']

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    net.eval()
    net.training = False
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
        acc = 100.*correct/total
        print("| Test Result\tAcc@1: %.2f%%" %(acc))

    sys.exit(0)

# Model
print('\n[Phase 2] : Model setup')
if args.resume:
    # Load checkpoint
    print('| Resuming from checkpoint...')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    checkpoint = torch.load('../wide-resnet.pytorch-master/checkpoint/'+args.dataset+file_name+'.pt')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('| Building net type [' + args.net_type + ']...')
    net, file_name = getNetwork(args)
    net.apply(conv_init)

if use_cuda:
    net.cuda()
    init_model = copy.deepcopy(net)
    init_model = torch.nn.DataParallel(init_model, device_ids=range(torch.cuda.device_count()))
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
init_model.eval()
criterion = nn.CrossEntropyLoss()
if args.attack == 'linf':
    adversary = LinfPGDAttack(net, epsilon=args.epsilon, k=args.PGD_steps, a=args.PGD_stepsize, random_start=False)
else:
    adversary = LtwoPGDattack(net, epsilon=args.epsilon, k=args.PGD_steps, a=args.PGD_stepsize, random_start=False)

# Training
def train(epoch):
    net.train()
    net.training = True
    train_loss = 0
    correct = 0
    total = 0
    optimizer = optim.SGD(net.parameters(), lr=cf.learning_rate(args.lr, epoch), momentum=0.9, weight_decay=args.weight_decay)

    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, cf.learning_rate(args.lr, epoch)))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)               # Forward Propagation
        loss = criterion(outputs, targets)  # Loss


        # adversarial training
        if args.std == False:
            y_pred = pred_batch(inputs, net)
            x_adv = adv_train(inputs, y_pred, net, criterion, adversary)
            x_adv_var = to_var(x_adv)
            loss_adv = criterion(net(x_adv_var), targets)

            if args.mixture == True:
                loss = (loss + loss_adv) / 2
            else:
                loss = loss_adv

        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, num_epochs, batch_idx+1,
                    (len(trainset)//batch_size)+1, loss.item(), 100.*correct/total))
        sys.stdout.flush()

def test2(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        # Save checkpoint when best model
        acc = 100.*correct/total
        print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.item(), acc))
        print('| Saving Best model...\t\t\tTop1 = %.2f%%' %(acc))

    std_tr_error,_,margin1, margin3, margin5 = validate(args, net, device, trainloader, criterion,adversary= adversary, std=True)
    std_test_error,_,_,_,_ = validate(args, net, device, testloader, criterion,adversary= adversary, std=True)
    ro_tr_error,_,_,_,_ = validate(args, net, device, trainloader, criterion,adversary= adversary, std=False)
    ro_test_error,_,_,_,_ = validate(args, net, device, testloader, criterion,adversary= adversary, std=False)
    print('training error:%.4f, margin1:%.4f, margin3:%.4f, margin5:%.4f'%(std_tr_error, margin1, margin3, margin5))
    state = {
            'net':net.module if use_cuda else net,
            'init_net':init_model.module if use_cuda else init_model,
            'acc':acc,
            'epoch':epoch,
            'margin1':margin1,
            'margin3':margin3,
            'margin5':margin5,
            'std_tr_error':std_tr_error,
            'std_test_error':std_test_error,
            'ro_tr_error':ro_tr_error,
            'ro_test_error':ro_test_error,
    }
    if not os.path.isdir('vggcheckpoint'):
        os.mkdir('vggcheckpoint')
    save_point = '../wide-resnet.pytorch-master/vggcheckpoint/'
    if not os.path.isdir(save_point):
        os.mkdir(save_point)
    torch.save(state, save_point+std+args.dataset+file_name+'no_samples'+str(args.nosamples)+'_'+str(args.weight_decay)+'.pt')

print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(optim_type))

elapsed_time = 0
for epoch in range(start_epoch, start_epoch+num_epochs):
    start_time = time.time()
    train(epoch)
    # test2(epoch)
    if (epoch+1)%50 ==0:
        test2(epoch)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))

test(net,trainloader)
test(net,testloader)
print('\n[Phase 4] : Testing model')
print('* Test results : Acc@1 = %.2f%%' %(best_acc))
