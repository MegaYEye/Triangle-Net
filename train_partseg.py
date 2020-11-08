

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
# from torchvision import transforms, utils
import numpy as np
from model import Reconstruction, TriangleNet, TriangleNet_Seg
from dataloader import load_data, SegmentationLoader
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser('Triangle-Net')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--episodes', type=int, default=1000)
parser.add_argument('--n_points', type=int, default=64)
parser.add_argument('--n_feature', type=int, default=8192)
parser.add_argument('--descriptor_type', type=str, default='C', help='[A, B, C]')
parser.add_argument('--rot_type', type=str, default='SO3', help='[SO3, z]')
args = parser.parse_args()
print(args)

batch_size = args.batch_size
train_episodes = args.episodes
descriptor_type = args.descriptor_type
n_points = args.n_points
rot_type = args.rot_type
fnum = args.n_feature
num_classes = 16
num_part = 50


seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
def calculate_shape_IoU(pred_np, seg_np, label, class_choice):
    label = label.squeeze()
    shape_ious = []
    for shape_idx in range(seg_np.shape[0]):
        if not class_choice:
            start_index = index_start[label[shape_idx]]
            num = seg_num[label[shape_idx]]
            parts = range(start_index, start_index + num)
        else:
            parts = range(seg_num[label[0]])
        part_ious = []
        for part in parts:
            pred = np.argmax(pred_np[shape_idx,:,start_index:start_index+num],axis=-1)
            a = pred == (part-start_index)
            b = seg_np[shape_idx] == part
            I = np.sum(np.logical_and(a,b))
            U = np.sum(np.logical_or(a,b))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    return shape_ious

# seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
# seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table}
# for cat in seg_classes.keys():
#     for label in seg_classes[cat]:
#         seg_label_to_cat[label] = cat


train_point_norm_np = np.load("./data/segmentation_preprocessed/train_points_seg.npy") #12137,2500,6
train_labels_np = np.load("./data/segmentation_preprocessed/train_labels_seg.npy") #12137,1
train_segs_np = np.load("./data/segmentation_preprocessed/train_segs_seg.npy") #12137,2500

test_point_norm_np = np.load("./data/segmentation_preprocessed/test_points_seg.npy") #2874,2500,6
test_labels_np = np.load("./data/segmentation_preprocessed/test_labels_seg.npy") #2874,1
test_segs_np = np.load("./data/segmentation_preprocessed/test_segs_seg.npy") #2874,2500

trainDataset = SegmentationLoader(train_point_norm_np, train_labels_np, train_segs_np,p_num=n_points, num_classes=num_classes, rot_type=rot_type)
testDataset = SegmentationLoader(test_point_norm_np, test_labels_np, test_segs_np,p_num=n_points, num_classes=num_classes, rot_type=rot_type)

trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True) #, num_workers = 6
testDataLoader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size, shuffle=False) #, num_workers = 6


inp_lookup={"A":4,"B":12,"C":24}
net = TriangleNet_Seg(part_num=num_part, inp=inp_lookup[descriptor_type], descriptor_type=descriptor_type, scale_invariant=False, feature_num=fnum).cuda()

optimizer_tri = optim.Adam(net.parameters(), lr=0.0001, betas=(0.5, 0.999),weight_decay = 1e-4)

best_test_iou=0

for ep in range(train_episodes):
    print("episode", ep)
    net = net.train()

    total=0
    correct=0
    train_true_seg = []
    train_pred_seg = []
    train_label_seg = []
    # for i, (points_norms, lb, segs) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader)):
    for i, (points_norms, lb, segs) in enumerate(trainDataLoader):
        optimizer_tri.zero_grad()

        points = points_norms[:,:,0:3].cuda()
        norms = points_norms[:,:,3:6].cuda()
        lb = lb.cuda()
        segs = segs.cuda()
        seg_pred = net(points, norms, lb)

        seg_pred_c = seg_pred.reshape(-1, num_part)
        seg_target_label = segs.flatten().long()
        loss = F.nll_loss(seg_pred_c, seg_target_label)
        # seg_pred = seg_pred.reshape(segs.shape[0], segs.shape[1],num_part)

        loss.backward()
        optimizer_tri.step()
        total += len(seg_target_label)
        correct += (torch.max(seg_pred_c,dim=-1)[1] == seg_target_label).sum().cpu().item()

        train_true_seg.append(segs.cpu().numpy())
        train_pred_seg.append(seg_pred.detach().cpu().numpy())
        train_label_seg.append(lb.max(dim=-1)[1].cpu().numpy().reshape(-1))
    train_true_seg = np.concatenate(train_true_seg, axis=0)
    train_pred_seg = np.concatenate(train_pred_seg, axis=0)
    train_label_seg = np.concatenate(train_label_seg, axis=0)
    train_ious = calculate_shape_IoU(train_pred_seg, train_true_seg, train_label_seg, None)
    train_ious = np.mean(train_ious)

    train_acc = correct/total

    # print("test")
    net = net.eval()
    total=0
    correct=0
    test_true_seg = []
    test_pred_seg = []
    test_label_seg = []
    for i, (points_norms, lb, segs) in enumerate(testDataLoader):

        points = points_norms[:,:,0:3].cuda()
        norms = points_norms[:,:,3:6].cuda()
        lb = lb.cuda()
        segs = segs.cuda()
        with torch.no_grad():
            seg_pred = net(points, norms, lb)
            seg_pred_c = seg_pred.reshape(-1, num_part)
            seg_target_label = segs.flatten().long()
            total += len(seg_target_label)
            correct += (torch.max(seg_pred_c,dim=-1)[1] == seg_target_label).sum().cpu().item()

        test_true_seg.append(segs.cpu().numpy())
        test_pred_seg.append(seg_pred.cpu().numpy())
        test_label_seg.append(lb.max(dim=-1)[1].cpu().numpy().reshape(-1))
    test_true_seg = np.concatenate(test_true_seg, axis=0)
    test_pred_seg = np.concatenate(test_pred_seg, axis=0)
    test_label_seg = np.concatenate(test_label_seg, axis=0)
    test_ious = calculate_shape_IoU(test_pred_seg, test_true_seg, test_label_seg, None)
    test_ious = np.mean(test_ious)
    test_acc = correct/total

    
    if test_ious>best_test_iou:
        best_test_iou = test_ious
        torch.save(net, f"best_iou_{best_test_iou}.pth")
    print(f"train_acc:{train_acc}, test_acc:{test_acc}, train_iou:{train_ious}, test_iou:{test_ious}, best_test_iou:{best_test_iou}")
