import torch
from torch import nn
from data.datamgr import SimpleDataManager , SetDataManager
from models.predesigned_modules import resnet12
import sys
import os
from utils import *
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
from svrg import SVRG
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# import tqdm
from torch.nn.parallel import DataParallel
# torch.backends.cudnn.benchmark = True

import random

#--------------参数设置--------------------
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', default=84, type=int, choices=[84, 224], help='input image size, 84 for miniImagenet and tieredImagenet, 224 for cub')
parser.add_argument('--dataset', default='mini_imagenet', choices=['mini_imagenet','tiered_imagenet','cub','FC100','dog','car'])
parser.add_argument('--data_path', default='/home/jiangweihao/data/mini-imagenet',type=str, help='dataset path')

parser.add_argument('--train_n_episode', default=10, type=int, help='number of episodes in meta train')
parser.add_argument('--val_n_episode', default=300, type=int, help='number of episodes in meta val')
parser.add_argument('--train_n_way', default=5, type=int, help='number of classes used for meta train')
parser.add_argument('--val_n_way', default=5, type=int, help='number of classes used for meta val')
parser.add_argument('--n_shot', default=5, type=int, help='number of labeled data in each class, same as n_support')
parser.add_argument('--n_query', default=15, type=int, help='number of unlabeled data in each class')
parser.add_argument('--num_classes', default=64, type=int, help='total number of classes in pretrain')

parser.add_argument('--batch_size', default=128, type=int, help='total number of batch_size in pretrain')
parser.add_argument('--freq', default=11, type=int, help='total number of inner frequency')

parser.add_argument('--lr', default=0.1, type=float, help='parameter of optimization')
parser.add_argument('--momentum', default=0.9, type=float, help='parameter of optimization')
parser.add_argument('--weight_decay', default=5.e-4, type=float, help='parameter of optimization')

parser.add_argument('--gpu', default='2')
parser.add_argument('--epochs', default=100)

params = parser.parse_args()

# 设置日志记录路径
log_path = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(log_path,'save/{}_tp{}_ns{}_resnet12_512_svrg_freq{}_bs{}_lr{}_new-svrg'. \
                 format(params.dataset,params.train_n_episode,params.n_shot,params.freq,params.batch_size,params.lr))
ensure_path(log_path)
set_log_path(log_path)
log('log and pth save path:  %s'%(log_path))
# log(params)
#--------------设置随机种子----------------
def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
 
seed_torch()
# -------------设置GPU--------------------
set_gpu(params.gpu)
# -------------导入数据--------------------

json_file_read = False
if params.dataset == 'mini_imagenet':
    base_file = 'train'
    # val_file = 'val'
    val_file = 'test'
    params.num_classes = 64
    params.data_path = '/home/jiangweihao/data/mini-imagenet'
elif params.dataset == 'cub':
    base_file = 'base.json'
    val_file = 'val.json'
    json_file_read = True
    params.num_classes = 100
    params.data_path = '/home/jiangweihao/data/CUB_200_2011'
elif params.dataset == 'tiered_imagenet':
    base_file = 'train'
    val_file = 'val'
    params.num_classes = 351
    params.data_path = '/home/jiangweihao/data/tiered_imagenet'
elif params.dataset == 'FC100':
    base_file = 'train'
    val_file = 'val'
    params.num_classes = 60
    params.data_path = '/home/jiangweihao/data/FC100'
elif params.dataset == 'dog':
    val_file = 'StanfordDog_Images'
    params.data_path = '/home/jiangweihao/data'
elif params.dataset == 'car':
    val_file = 'test'
    params.data_path = '/home/jiangweihao/data/StanfordCar'
else:
    ValueError('dataset error')

log(params)
# -----------  base data ----------------------
base_datamgr = SimpleDataManager(params.data_path, params.image_size, batch_size=params.batch_size, json_read=json_file_read)
base_loader = base_datamgr.get_data_loader(base_file, aug=True)

#-----------  train data ----------------------
train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
train_datamgr = SetDataManager(params.data_path, params.image_size, n_query=params.n_query, n_episode=params.train_n_episode, json_read=json_file_read, **train_few_shot_params)
train_loader = train_datamgr.get_data_loader(base_file, aug=True)

#------------ val data ------------------------
test_few_shot_params = dict(n_way=params.val_n_way, n_support=params.n_shot)
val_datamgr = SetDataManager(params.data_path, params.image_size, n_query=params.n_query, n_episode=params.val_n_episode, json_read=json_file_read, **test_few_shot_params)
val_loader = val_datamgr.get_data_loader(val_file, aug=False)

#   ------查看导入的数据----------
# target, label = next(iter(base_loader))
# print(len(base_loader))
# print(target.size())          # batch_size=128 ==> torch.Size([128, 3, 84, 84])    
# print(label.size())           # torch.Size([128])
# print('--------------------')
# target1, label1 = next(iter(train_loader))
# print(len(train_loader))      # train_n_episode=10  ==>  len=10
# print(target1.size())         # torch.Size([5, 20, 3, 84, 84])
# print(label1.size())          # torch.Size([5, 20])
# print('--------------------')
# target2, label2 = next(iter(val_loader))
# print(len(val_loader))
# print(target2.size())
# print(label2.size())

# ----------- 导入模型 -------------------------
model = resnet12(use_fc= False, num_classes= params.num_classes, use_pooling = True)
model.cuda()
del model.fc                         # 删除最后的全连接层
# from torchinfo import summary
# summary(model,[5,3,84,84])
# classifier = nn.Linear(640, 64)
classifier = nn.Linear(512, params.num_classes)
classifier.cuda()
# model = DataParallel(model, device_ids=[0, 1])
# classifier = DataParallel(classifier, device_ids=[0, 1])
# model2 = resnet12(use_fc= False, num_classes= 5, use_pooling = True).cuda()
# classifier2 = nn.Linear(640, 5)
# classifier2.cuda()
# print(model)
# print(classifier2)

# ---------------------------------------------
loss_fn = torch.nn.CrossEntropyLoss()
# loss_fn.cuda()
alpha = 1e-1
freq = 100 #how often to recompute large gradient
            #The optimizer will not update model parameters on iterations
            #where the large batches are calculated


optimizer = SVRG(model.parameters(), lr = params.lr, freq = params.freq)   # 优化    weight_decay: 5.e-4
# optimizer0 = torch.optim.SGD(model.parameters(), lr = alpha)   # 优化

optimizer1 = torch.optim.SGD(classifier.parameters(), lr = alpha, momentum=params.momentum, weight_decay=params.weight_decay)                
# optimizer2 = torch.optim.SGD(classifier2.parameters(), lr = alpha)
schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[90],gamma=0.1)     #[30,60]
# 指数衰减学习率，衰减率为 gamma=0.9
# schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)
# 周期为50epoch，lr最小值为0（默认）
# schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=25, last_epoch=-1)

schedule1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1,milestones=[30,60],gamma=0.1)

epochs = 100
iterations = int (epochs * len(base_loader))               

#SVRG Training
counter = 0
start = time.time()
model.train()
classifier.train()
# classifier2.train()
max_val_acc = 0.0


# while(counter < iterations):
    #compute large batch gradient
for epoch in range(epochs):   
    # temp2,yt=next(iter(base_loader))
    log('==========start training===============')
    epoch_learning_rate = 0.1
    for param_group in optimizer.param_groups:
        epoch_learning_rate = param_group['lr']
        
    log( 'Train Epoch: {}\tLearning Rate: {:.4f}'.format(
                        epoch, epoch_learning_rate))

    model.train()
    out_avg_loss = []
    timer = Timer()
    for idx,(temp,yt) in enumerate(base_loader):
        temp = temp.cuda()
        yt = yt.cuda()
        y_pred = classifier(model(temp))
    
        loss = loss_fn(y_pred, yt)         # //2 only for cub
        optimizer.zero_grad()
        optimizer1.zero_grad()
        loss.backward()
        
        optimizer.step()
        optimizer1.step()
        out_avg_loss.append(loss.item())
        
        
        #update models using mini batch gradients
              
        avg_loss = 0
        total_correct = 0
        for idy, (temp2,_) in enumerate(train_loader):   
            # temp2, _ =next(iter(train_loader))

            support,query = temp2.split([params.n_shot,params.n_query],dim=1)
            _,_,c,h,w = support.shape
            support = support.reshape(-1,c,h,w)
            support = support.cuda()
            query = query.reshape(-1,c,h,w)
            query = query.cuda()
            support_f = model(support).view(params.val_n_way,params.n_shot,-1)
            support_f = support_f.mean(dim=1).squeeze(1)
            query_f = model(query) 

            y = np.repeat(range(params.val_n_way),params.n_query)
            y = torch.from_numpy(y)
            y = y.cuda()
            y_pred = compute_logits(query_f, support_f, metric='cos', temp=1.0)

            loss = loss_fn(y_pred,y)
            avg_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = y_pred.data.max(1)[1]
            total_correct += pred.eq(y).sum()
            # acc = compute_acc(y_pred, y, reduction='mean')
        
        # total = len(train_loader)*params.train_n_way*params.n_query
        # log('第 {} 轮循环, loss:{:.4f}, acc:{:.4f} , consume time: {:.4f}'.format(idx, \
        #     avg_loss/len(train_loader),(total_correct/total),timer.t()))

    out_avg_loss = np.mean(out_avg_loss)
    
    log('train epoch:%d , loss:%.3f'%(epoch,out_avg_loss))
    log('train epoch time: {}'.format(timer.t()))
    schedule.step()
    schedule1.step()

    # if (epoch+1) % 2 == 0 or epoch == epochs-1:
    print('==========start testing===============')
    model.eval()
    # classifier2.eval()
    val_loss_avg = 0
    total_correct = 0
    total = len(val_loader) * params.val_n_way * params.n_query
    val_accuracies = []
    val_losses = []

    for idx, (x, _) in enumerate(val_loader):
        support,query = x.split([params.n_shot,params.n_query],dim=1)
        _,_,c,h,w = support.shape
        support = support.reshape(-1,c,h,w)
        support = support.cuda()
        query = query.reshape(-1,c,h,w)
        query = query.cuda()
        with torch.no_grad():
            support_f = model(support).view(params.val_n_way,params.n_shot,-1)
            support_f = support_f.mean(dim=1).squeeze(1)
            query_f = model(query) 

        y = np.repeat(range(params.val_n_way),params.n_query)
        y = torch.from_numpy(y)
        y = y.cuda()
        y_pred = compute_logits(query_f, support_f, metric='cos', temp=1.0)

        loss = loss_fn(y_pred,y)
        val_loss_avg += loss.item()
        
        pred = y_pred.data.max(1)[1]
        total_correct += pred.eq(y).sum()
        acc = pred.eq(y).sum().item()
        val_accuracies.append(acc)

    val_loss_avg /= len(val_loader)
    val_acc_avg = float(total_correct) / total * 100
    log('epoch: {} val loss: {:.4f}  val acc: {:.4f}'.format(epoch,val_loss_avg,val_acc_avg))
    # print(val_acc_avg)
    
    # val_acc_avg = np.mean(np.array(val_accuracies))
    val_acc_ci95 = 1.96 * np.std(np.array(val_accuracies)) / np.sqrt(params.val_n_episode)

    # val_loss_avg = np.mean(np.array(val_losses))

    if val_acc_avg > max_val_acc:
        max_val_acc = val_acc_avg
        torch.save({'embedding': model.state_dict(), 'head': classifier.state_dict()},\
                    os.path.join(log_path, 'best_model.pth'))
        log( 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} % (Best)'\
                .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))
    else:
        log( 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} %'\
                .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))

    torch.save({'embedding': model.state_dict(), 'head': classifier.state_dict()}\
                , os.path.join(log_path, 'last_epoch.pth'))

    if epoch % 20 == 0:
        torch.save({'embedding': model.state_dict(), 'head': classifier.state_dict()}\
                    , os.path.join(log_path, 'epoch_{}.pth'.format(epoch)))
        


torch.save(model.state_dict(),'model_with_svrg.pth')
log('time for SVRG ' + str(iterations) +' steps')
log(time.time()-start)
log('===========================training end!===================================')

# torch.save(classifier2.state_dict(),'classifier.pth')
# ------------------val ----------------------
# model.load_state_dict(torch.load('autodl-tmp/jiangweihao/A_FSL_Lab_JWH/model.pth'))



