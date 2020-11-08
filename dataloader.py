import open3d as o3d
import numpy as np
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from itertools import combinations
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree


# reference code: # https://github.com/daavoo/pyntcloud/blob/master/pyntcloud/structures/voxelgrid.py
def create_voxel(points_np, size3d):
    x_y_z = size3d
    pts_np = points_np

    xyzmin = pts_np.min(0)
    xyzmax = pts_np.max(0)
    
    margin = max(xyzmax - xyzmin) - (xyzmax - xyzmin)
    xyzmin = xyzmin - margin / 2
    xyzmax = xyzmax + margin / 2
    
    segments = []
    shape = []
    for i in range(3):
            # note the +1 in num
        s, step = np.linspace(xyzmin[i], xyzmax[i], num=(x_y_z[i] + 1), retstep=True)
        segments.append(s)
        shape.append(step)

    n_voxels = x_y_z[0] * x_y_z[1] * x_y_z[2]

    voxel_x = np.clip(np.searchsorted(segments[0], pts_np[:, 0]) - 1, 0, x_y_z[0])
    voxel_y = np.clip(np.searchsorted(segments[1], pts_np[:, 1]) - 1, 0, x_y_z[1])
    voxel_z = np.clip(np.searchsorted(segments[2], pts_np[:, 2]) - 1, 0,  x_y_z[2])

    voxel_n = np.ravel_multi_index([voxel_x, voxel_y, voxel_z], x_y_z)

    vector = np.zeros(n_voxels)

    vector[np.unique(voxel_n)] = 1

    voxel = vector.reshape(x_y_z)
    return voxel

def add_normal(np_pts):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np_pts)
    pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=16))
    return pc


def vlen(q):
    return torch.norm(q, p=2, dim=1, keepdim=True).detach()

def angle_between_batch(v1, v2):
    v1_norm = v1/(1e-8+vlen(v1))
    v2_norm = v2/(1e-8+vlen(v2))

    dot_prod = torch.sum(v1_norm * v2_norm, dim=1, keepdim=True)
    ang = torch.acos(torch.clamp(dot_prod,min=-1,max=1))

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = []
    return (data, label, seg)

#https://github.com/yanx27/Pointnet_Pointnet2_pytorch
# Thanks the author for the codes of loading data
def load_data(dir,classification = False):
    data_train0, label_train0,Seglabel_train0  = load_h5(dir + 'ply_data_train0.h5')
    data_train1, label_train1,Seglabel_train1 = load_h5(dir + 'ply_data_train1.h5')
    data_train2, label_train2,Seglabel_train2 = load_h5(dir + 'ply_data_train2.h5')
    data_train3, label_train3,Seglabel_train3 = load_h5(dir + 'ply_data_train3.h5')
    data_train4, label_train4,Seglabel_train4 = load_h5(dir + 'ply_data_train4.h5')
    data_test0, label_test0,Seglabel_test0 = load_h5(dir + 'ply_data_test0.h5')
    data_test1, label_test1,Seglabel_test1 = load_h5(dir + 'ply_data_test1.h5')
    train_data = np.concatenate([data_train0,data_train1,data_train2,data_train3,data_train4])
    train_label = np.concatenate([label_train0,label_train1,label_train2,label_train3,label_train4])
    train_Seglabel = np.concatenate([Seglabel_train0,Seglabel_train1,Seglabel_train2,Seglabel_train3,Seglabel_train4])
    test_data = np.concatenate([data_test0,data_test1])
    test_label = np.concatenate([label_test0,label_test1])
    test_Seglabel = np.concatenate([Seglabel_test0,Seglabel_test1])

    if classification:
        return train_data, train_label, test_data, test_label
    else:
        return train_data, train_Seglabel, test_data, test_Seglabel

class ModelNetDataLoader(Dataset):
    def __init__(self, data, labels, point_num=16, rot=False, use_buffer=True, use_voxel=False, rot_type="SO3"):
        self.data, self.labels = data, labels
        self.point_num = point_num
        self.o3dmodel={}
        self.voxel_buffer = {}
        self.rot = rot
        self.use_buffer=use_buffer
        self.use_voxel=use_voxel
        self.rot_type = rot_type

    def rotate_point_cloud_random_SO3(self, pc):
        roll, pitch, yaw = np.random.rand(3)*np.pi*2
        rot = R.from_euler('ZYX', (yaw, pitch, roll))
        pc = pc.rotate(rot.as_dcm(),center=np.array([0,0,0]))
        return pc

    def rotate_point_cloud_random_z(self, pc):
        roll, pitch, yaw = np.random.rand(3)*np.pi*2
        pitch, yaw = 0,0
        rot = R.from_euler('ZYX', (yaw, pitch, roll))
        pc = pc.rotate(rot.as_dcm(),center=np.array([0,0,0]))

        return pc

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.use_buffer: 
            if index not in self.o3dmodel:
                self.o3dmodel[index] = add_normal(self.data[index])
            o3dpc = self.o3dmodel[index]
        else:
            o3dpc = add_normal(self.data[index])
        
        if self.rot:
            if self.rot_type == "SO3":
                o3dpc = self.rotate_point_cloud_random_SO3(o3dpc)
            if self.rot_type == "z":
                o3dpc = self.rotate_point_cloud_random_z(o3dpc)

        points = np.asarray(o3dpc.points).astype(np.float32) #2048,30
        norms = np.asarray(o3dpc.normals).astype(np.float32) #2048,30

        sel_pts_idx = np.random.choice(points.shape[0], size=self.point_num, replace=False).reshape(-1)
        points = points[sel_pts_idx]
        norms = norms[sel_pts_idx]

        if self.use_voxel:
            if index not in self.voxel_buffer:
                self.voxel_buffer[index] = create_voxel(self.data[index], [32,32,32]).astype(np.float32)
            voxel = self.voxel_buffer[index]
        if self.use_voxel:
            return points, norms, voxel, self.labels[index]
        else:
            return points, norms, self.labels[index]


class SegmentationLoader(Dataset):
    def __init__(self, point_norm, lb, segs, p_num=1024, num_classes=16, rot=False, rot_type="SO3"):
        self.point_norm = point_norm
        self.lb = lb
        self.segs = segs
        self.p_num=p_num
        self.rot=rot
        self.num_classes = num_classes
        self.rot_type=rot_type
        self.o3dmodel={}

    def rotate_point_cloud_random_SO3(self, pc):
        roll, pitch, yaw = np.random.rand(3)*np.pi*2
        rot = R.from_euler('ZYX', (yaw, pitch, roll))
        # center = np.mean(np.asarray(pc.points).astype(np.float32),0, keepdims=True).T
        center = np.zeros((3,1))
        pc = pc.rotate(rot.as_dcm(), center=center)

        return pc
    def rotate_point_cloud_random_z(self, pc):
        roll, pitch, yaw = np.random.rand(3)*np.pi*2
        pitch, yaw = 0,0
        rot = R.from_euler('ZYX', (yaw, pitch, roll))
        center = np.zeros((3,1))
        pc = pc.rotate(rot.as_dcm(), center=center)

        return pc
    def __len__(self):
        return self.lb.shape[0]

    def one_hot(self, lb):
        lb = lb.flatten()
        a = np.zeros((self.num_classes,)).astype(np.float32)
        a[lb] = 1
        return a


    def __getitem__(self, index):
        rand_idx = np.random.choice(len(self.point_norm[index]),self.p_num, replace=False)
        points = self.point_norm[index][rand_idx][:,:3]
        normals = self.point_norm[index][rand_idx][:,3:]
        if self.rot:
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(points)
            pc.normals = o3d.utility.Vector3dVector(normals)
            if self.rot_type == "z":
                pc = self.rotate_point_cloud_random_z(pc)
            if self.rot_type == "SO3":
                pc = self.rotate_point_cloud_random_SO3(pc)

            points = np.asarray(pc.points).astype(np.float32) #2048,30
            normals = np.asarray(pc.normals).astype(np.float32) #2048,30

        point_normal = np.concatenate([points, normals], axis=-1)
        return point_normal, self.one_hot(self.lb[index]), self.segs[index][rand_idx]


class ScanObjectNNDataLoader(Dataset):
    def __init__(self, point, lb, num_classes=15,n_points=2048, rot=False, rot_type="SO3"):
        points = []
        normals = []
        for p in point:
            pcd=add_normal(p)
            p = np.asarray(pcd.points).astype(np.float32)
            n = np.asarray(pcd.normals).astype(np.float32)
            points.append(p[None,...])
            normals.append(n[None,...])
        self.points = np.concatenate(points, axis=0)
        self.normals = np.concatenate(normals, axis=0)
        self.lb = lb
        self.num_classes = num_classes
        self.n_points = n_points
        self.rot = rot
        self.rot_type=rot_type
        
    def __len__(self):
        return self.lb.shape[0]
    
    def rotate_point_cloud_random_SO3(self, pc):
        roll, pitch, yaw = np.random.rand(3)*np.pi*2
        rot = R.from_euler('ZYX', (yaw, pitch, roll))
        # center = np.mean(np.asarray(pc.points).astype(np.float32),0, keepdims=True).T
        center = np.zeros((3,1))
        pc = pc.rotate(rot.as_dcm(), center=center)

        return pc
    

    def __getitem__(self, index):
        idx = np.random.choice(self.points.shape[1], size=(self.n_points,))
        points = self.points[index][idx]
        normals = self.normals[index][idx]
        if self.rot:
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(points)
            pc.normals = o3d.utility.Vector3dVector(normals)
            if self.rot_type == "z":
                pc = self.rotate_point_cloud_random_z(pc)
            if self.rot_type == "SO3":
                pc = self.rotate_point_cloud_random_SO3(pc)
                
            points = np.asarray(pc.points).astype(np.float32) #2048,30
            normals = np.asarray(pc.normals).astype(np.float32) #2048,30
                
        
        return points, normals, self.lb[index]

    
def load_h5_scanobjectNN(h5_train,h5_test):
    f = h5py.File(h5_train)
    train_data = f['data'][:]
    train_label = f['label'][:]
    f = h5py.File(h5_test)
    test_data = f['data'][:]
    test_label = f['label'][:]
    return train_data, train_label, test_data, test_label

        


