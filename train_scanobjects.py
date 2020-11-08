import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from model import Reconstruction, TriangleNet
from dataloader import load_h5_scanobjectNN, ScanObjectNNDataLoader
from tqdm import tqdm
import os
import argparse
parser = argparse.ArgumentParser('Triangle-Net')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--datapath_train', type=str, default=r"data/ScanObjectNN_nobg/training_objectdataset_augmentedrot_scale75.h5", help='path of ScanObjectNN dataset (train h5)')
parser.add_argument('--datapath_test', type=str, default=r"data/ScanObjectNN_nobg/test_objectdataset_augmentedrot_scale75.h5", help='path of ScanObjectNN dataset (test h5)')
parser.add_argument('--episodes', type=int, default=1000)
parser.add_argument('--n_points', type=int, default=32)
parser.add_argument('--descriptor_type', type=str, default='C', help='[A, B, C]')
parser.add_argument('--rot_type', type=str, default='SO3', help='[SO3, z]')

args = parser.parse_args()

os.environ["OMP_NUM_THREADS"] = "3" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "3" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "3" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "3" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "3" # export NUMEXPR_NUM_THREADS=1
torch.set_num_threads(2) # this is impo

# datapath = args.datapath
batch_size = args.batch_size
train_episodes = args.episodes
descriptor_type = args.descriptor_type
rot_type = args.rot_type
h5_train = args.datapath_train
h5_test = args.datapath_test
n_points = args.n_points

train_data, train_label, test_data, test_label = load_h5_scanobjectNN(h5_train, h5_test)
trainDataset = ScanObjectNNDataLoader(train_data, train_label,n_points=n_points, rot=True, rot_type=rot_type)
testDataset = ScanObjectNNDataLoader(test_data, test_label,n_points=n_points, rot=True, rot_type=rot_type)
trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True) #, num_workers = 6
testDataLoader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size, shuffle=False) #, num_workers = 6
inp_lookup={"A":4,"B":12,"C":24}
net = TriangleNet(k=15, inp=inp_lookup[descriptor_type], descriptor_type=descriptor_type, scale_invariant=False, encoder_type="full").cuda()
# net = DGCNN_cls().cuda()
optimizer_tri = optim.Adam(net.parameters(), lr=0.0003, betas=(0.5, 0.999),weight_decay = 1e-4)

bestacc=0
for ep in range(train_episodes):
    print("episode", ep)
    net = net.train()
    for i, (points, norms, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader)):
        optimizer_tri.zero_grad()

        target = target.cuda()
        points = points.cuda()
        norms = norms.cuda()

        pred,z = net(points,norms)
        loss = F.nll_loss(pred, target.long())
        loss.backward()
        optimizer_tri.step()

    net = net.eval()
    total_cnt=0
    correct_cnt=0
    for i, (points, norms, target) in enumerate(testDataLoader):

        points = points.cuda()
        norms = norms.cuda()
        target = target.cuda()
        with torch.no_grad():
            pred,z = net(points, norms)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()
        correct_cnt +=correct.item()
        total_cnt+=points.shape[0]
    test_acc =  correct_cnt/total_cnt
    if test_acc > bestacc:
        bestacc = test_acc
        torch.save(net, f"best_net_scanobj_{test_acc}.pth")

    print("test acc: ",test_acc, "best test acc: ", bestacc)

