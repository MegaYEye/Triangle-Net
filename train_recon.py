import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
# from torchvision import transforms, utils
import time
import numpy as np
from model import Reconstruction, TriangleNet
from dataloader import load_data, ModelNetDataLoader
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser('Triangle-Net')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--datapath', type=str, default=r'./data/modelnet40_ply_hdf5_2048/', help='path of modelnet 40 dataset')
parser.add_argument('--episodes', type=int, default=1000)
parser.add_argument('--n_points', type=int, default=16)
parser.add_argument('--descriptor_type', type=str, default='C', help='[A, B, C]')
parser.add_argument('--rot_type', type=str, default='SO3', help='[SO3, z]')
args = parser.parse_args()

datapath = args.datapath
batch_size = args.batch_size
train_episodes = args.episodes
descriptor_type = args.descriptor_type
n_points = args.n_points
rot_type = args.rot_type

train_data, train_label, test_data, test_label = load_data(datapath, classification=True)
trainDataset = ModelNetDataLoader(train_data, train_label, use_voxel=True, point_num = n_points, rot_type=rot_type)
testDataset = ModelNetDataLoader(test_data, test_label, use_voxel=True, point_num = n_points, rot_type=rot_type)
trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True) #, num_workers = 6
testDataLoader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size, shuffle=True) #, num_workers = 6
inp_lookup={"A":4,"B":12,"C":24}
net = TriangleNet(k=40, inp=inp_lookup[descriptor_type], descriptor_type=descriptor_type).cuda()
recon_net = Reconstruction().cuda()
optimizer_tri = optim.Adam(net.parameters(), lr=0.001, betas=(0.5, 0.999),weight_decay = 1e-4)
optimizer_recon = optim.Adam(recon_net.parameters(), lr=0.001, betas=(0.5, 0.999)) 
bestacc=0
for ep in range(train_episodes):
    print("episode", ep)
    net = net.train()
    for i, (points, norms, voxels, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader)):
        optimizer_tri.zero_grad()
        optimizer_recon.zero_grad()
        target = target[:, 0]
        target = target.cuda()
        points = points.cuda()
        norms = norms.cuda()
        voxels = voxels.cuda()

        pred, z = net(points,norms)
        yhat = recon_net(z)
        loss1 = F.nll_loss(pred, target.long())
        loss2 = F.binary_cross_entropy(yhat.view(yhat.shape[0], 32*32*32), voxels.view(voxels.shape[0], 32*32*32), reduction='sum')
        loss = loss1 + loss2
        loss.backward()
        optimizer_tri.step()
        optimizer_recon.step()

    # print("test")
    net = net.eval()
    total_cnt=0
    correct_cnt=0
    for i, (points, norms,  voxels, target) in enumerate(testDataLoader):

        points = points.cuda()
        norms = norms.cuda()
        target = target[:, 0]
        target = target.cuda()
        # voxels = voxels.cuda()
        
        with torch.no_grad():
            pred, z = net(points, norms)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()
        correct_cnt +=correct.item()
        total_cnt+=points.shape[0]
    test_acc =  correct_cnt/total_cnt
    if test_acc > bestacc:
        bestacc = test_acc
        torch.save(net, f"best_net_{test_acc}_{n_points}.pth")
        torch.save(recon_net, f"best_recon_net_{test_acc}_{n_points}.pth")
    print("test acc: ",test_acc, "best test acc: ", bestacc)

