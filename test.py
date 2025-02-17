import torch
from torch import nn
from data.datamgr import SetDataManager
from models.predesigned_modules import resnet12
import numpy as np
from utils import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# import qpth
import random
#--------------设置随机种子----------------
def seed_torch(seed=1029+123+99+44+110):         # 1029
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
 
seed_torch()
# ------------------ 参数设置 -----------------
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--image_size', default=84, type=int, choices=[84, 224], help='input image size, 84 for miniImagenet and tieredImagenet, 224 for cub')
parser.add_argument('--dataset', default='mini_imagenet', choices=['mini_imagenet','tiered_imagenet','cub','dog','car','aircraft'])
parser.add_argument('--data_path', default='/home/jiangweihao/data/mini-imagenet',type=str, help='dataset path')

parser.add_argument('--train_n_episode', default=600, type=int, help='number of episodes in meta train')
parser.add_argument('--val_n_episode', default=1000, type=int, help='number of episodes in meta val')
parser.add_argument('--train_n_way', default=5, type=int, help='number of classes used for meta train')
parser.add_argument('--val_n_way', default=5, type=int, help='number of classes used for meta val')
parser.add_argument('--n_shot', default=5, type=int, help='number of labeled data in each class, same as n_support')
parser.add_argument('--n_query', default=16, type=int, help='number of unlabeled data in each class')
parser.add_argument('--num_classes', default=351, type=int, help='total number of classes in pretrain')

parser.add_argument('--batch_size', default=128, type=int, help='total number of batch_size in pretrain')
parser.add_argument('--pth_path', default='/home/jiangweihao/code/svrg_fsl/save/mini_imagenet_10_1_resnet12_512_svrg_freq11/epoch_40.pth')
params = parser.parse_args()

#------------ val data ------------------------
test_file = 'test'
json_file_read = False

if params.dataset == 'cub':
    test_file = 'novel.json'
    json_file_read = True

if params.dataset == 'dog':
    test_file = 'StanfordDog_Images'
    params.data_path = '/home/jiangweihao/data'

if params.dataset == 'car':
    test_file = 'test'
    params.data_path = '/home/jiangweihao/data/StanfordCar'

if params.dataset == 'aircraft':
    test_file = 'test'
    params.data_path = '/home/jiangweihao/data/fgvc-aircraft-2013b/data'

test_few_shot_params = dict(n_way=params.val_n_way, n_support=params.n_shot)
test_datamgr = SetDataManager(params.data_path, params.image_size, n_query=params.n_query, n_episode=params.val_n_episode, json_read=json_file_read, **test_few_shot_params)
test_loader = test_datamgr.get_data_loader(test_file, aug=False)

#------------- load model ---------------------
model = resnet12(use_fc= False, num_classes= params.num_classes, use_pooling = True)
model.cuda()
del model.fc   
state_dict = torch.load(params.pth_path)
if len(state_dict)==2:
     model.load_state_dict(state_dict['embedding'])
else:
     model.load_state_dict(state_dict)

loss_fn = torch.nn.CrossEntropyLoss()

model.eval()
# classifier.eval()
avg_loss = 0
total_correct = 0
total = len(test_loader) * params.val_n_way *  params.n_query
acc_list = []
print('=====================',len(test_loader),'episodes test=====================')
for idx, (x, _) in enumerate(test_loader):
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
    y_pred = compute_logits(query_f, support_f, metric='cos', temp=1.0)         # temp = 4.147467613220215  default=1.0 
    loss = loss_fn(y_pred,y)
    avg_loss += loss.item()

    pred = y_pred.data.max(1)[1]
    total_correct += pred.eq(y).sum()

    # acc = compute_acc(y_pred, y, reduction='mean')
    acc = pred.eq(y).sum().item()/len(y) * 100
    acc_list.append(acc)
    # print('the {} batch test,acc is {},the loss is {}'.format(idx+1, pred.eq(y).sum()/x.shape[0], loss))
    print('the {} batch test,acc is {:.2f} %,the loss is {:.4f}'.format(idx+1, acc, loss))
avg_loss /= len(test_loader)
acc = float(total_correct) / total *100
test_acc_ci95 = 1.96 * np.std(np.array(acc_list)) / np.sqrt(params.val_n_episode)
print('avg_loss: {:.4f} '.format(avg_loss))
print('avg_acc: {:.2f} ± {:.2f} %'.format(acc,test_acc_ci95))