import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
# from torchvision import transforms, utils
import torch.nn.functional as F
import time


def count_parameters(model):
    from prettytable import PrettyTable
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    



def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def vlen(q):
    return torch.norm(q, p=2, dim=2, keepdim=True).detach()

def angle_between_batch(v1, v2):
    v1_norm = v1/(1e-8+vlen(v1))
    v2_norm = v2/(1e-8+vlen(v2))

    dot_prod = torch.sum(v1_norm * v2_norm, dim=2, keepdim=True)
    ang = torch.acos(torch.clamp(dot_prod,min=-1,max=1))

    return ang

class Reconstruction(nn.Module):
    def __init__(self, z_dim=1024):
        super().__init__()
     
        self.main = nn.Sequential(

            nn.ConvTranspose3d(
                in_channels = z_dim,
                out_channels = 64 * 4,
                kernel_size = 4,
                stride = 1,
                padding = 0 
            ), 
            nn.BatchNorm3d(64 * 4),
            nn.ReLU(),

            nn.ConvTranspose3d(
                in_channels = 64 * 4,
                out_channels = 64 * 2,
                kernel_size = 4,
                stride = 2,
                padding = 1
            ),
            nn.BatchNorm3d(64 * 2),
            nn.ReLU(), 
     
            nn.ConvTranspose3d(
                in_channels = 64 * 2,
                out_channels = 64 * 1,
                kernel_size = 4,
                stride = 2,
                padding = 1
            ),
            nn.BatchNorm3d(64 * 1),
            nn.ReLU(), 

            nn.ConvTranspose3d(
                in_channels = 64 * 1,
                out_channels = 1,
                kernel_size = 4,
                stride = 2,
                padding = 1
            ),    

            nn.Sigmoid()                     
        )
        self.main.apply(init_weights)
        
    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], 1, 1, 1) 
        x = self.main(x)
        x = x.view(x.shape[0], x.shape[2], x.shape[3], x.shape[4])
        return x

class Encoder_slim(nn.Module):
    def __init__(self, inp=4):
        super(Encoder_slim, self).__init__()
        self.conv1 = torch.nn.Conv1d(inp, 64, 1) 
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        # x = torch.max(x, 2, keepdim=True)[0]
        # x = x.view(-1, 1024)
        return x
    
class Encoder_middle(nn.Module):
    def __init__(self, inp=4):
        super(Encoder_middle, self).__init__()
        self.conv1 = torch.nn.Conv1d(inp, 64, 1) 
        self.conv2 = torch.nn.Conv1d(64+inp, 128, 1)
        self.conv3 = torch.nn.Conv1d(128+inp, 512, 1)
        self.conv4 = torch.nn.Conv1d(512+inp, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)


    def forward(self, x):
        in_x = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = torch.cat([x, in_x], dim=1)
        x = F.relu(self.bn2(self.conv2(x)))
        x = torch.cat([x, in_x], dim=1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.cat([x, in_x], dim=1)
        x = self.conv4(x)
        # x = torch.max(x, 2, keepdim=True)[0]
        # x = x.view(-1, 1024)
        return x
    
class Encoder(nn.Module):
    def __init__(self, inp=4):
        super(Encoder, self).__init__()
        
        self.conv1 = torch.nn.Conv1d(inp, 64, 1) 
        self.conv2 = torch.nn.Conv1d(64+inp, 128, 1)
        self.conv3 = torch.nn.Conv1d(128+inp, 512, 1)
        self.conv4 = torch.nn.Conv1d(512+inp, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)
        self.f = nn.Sequential(
            nn.Conv1d(64+128+512+inp, 1024, 1),
            # nn.ReLU()
            
        )


    def forward(self, x):
        in_x = x
        x1 = x= F.relu(self.bn1(self.conv1(x)))
        x = torch.cat([x1, in_x], dim=1)
        x2 = x= F.relu(self.bn2(self.conv2(x)))
        x = torch.cat([x2, in_x], dim=1)
        x3 = x= F.relu(self.bn3(self.conv3(x)))
        x = torch.cat([x3, in_x], dim=1)
        x = self.conv4(x) + self.f(torch.cat([x3, x2, x1, in_x], dim=1))
        # x = torch.max(x, 2, keepdim=True)[0]
        # x = x.view(-1, 1024)
        return x


class TriangleNet(nn.Module):
    def __init__(self, k=2,inp=4, feature_num=4096, descriptor_type='A', scale_invariant=False, point_feature=False, encoder_type="slim"):
        super(TriangleNet, self).__init__()
        self.feature_num = feature_num
        self.scale_invariant = scale_invariant
        if encoder_type == "full":
            self.feat = Encoder(inp=inp)
        elif encoder_type == "middle":
            self.feat = Encoder_middle(inp=inp)
        else:
            self.feat = Encoder_slim(inp=inp)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.point_feature = point_feature
        if descriptor_type == 'A':
            self.extractor = self.extract_feature_A
        elif descriptor_type == 'B':
            self.extractor = self.extract_feature_B
        elif descriptor_type == 'C':
            self.extractor = self.extract_feature_C
        else:
            raise Exception 

    def extract_feature_A(self, points, norms, f_num):
        p_num = points.shape[1]
        idx = torch.randint(p_num, size=(f_num, 3))
        idx0=torch.arange(p_num).view(-1,1).repeat((1,f_num//p_num)).flatten()
        p1, p2 = points[:,idx0], points[:,idx[:,1]]
        n1, n2 = norms[:,idx0], norms[:,idx[:,1]]
        dis1 = vlen(p1-p2)
        ang1a = angle_between_batch(p1-p2, n1)
        ang1b = angle_between_batch(p2-p1, n2)
        ang1c = angle_between_batch(n1, n2)
        buf = torch.cat([dis1,ang1a, ang1b, ang1c],dim=2)
        
        if self.scale_invariant:
            max_dis = buf[:,0].max()
            buf[:,0] = buf[:,0]/max_dis

        return buf

    def extract_feature_B(self, points, norms, f_num):
        p_num = points.shape[1]
        idx = torch.randint(p_num, size=(f_num, 2))

        idx0=torch.arange(p_num).view(-1,1).repeat((1,f_num//p_num)).flatten()
        p1, p2, p3 = points[:,idx0], points[:,idx[:,1]], points[:,idx[:,2]]
        n1, n2, n3 = norms[:,idx0], norms[:,idx[:,1]], norms[:,idx[:,2]]
        dis1 = vlen(p1-p2)
        dis2 = vlen(p2-p3)
        dis3 = vlen(p3-p1)
        
        ang1a = angle_between_batch(p1-p2, n1)
        ang1b = angle_between_batch(p1-p3, n1)
        ang2a = angle_between_batch(p2-p1, n2)
        ang2b = angle_between_batch(p2-p3, n2)
        ang3a = angle_between_batch(p3-p1, n3)
        ang3b = angle_between_batch(p3-p2, n3)
        
        angt1 = angle_between_batch(p1-p2, p1-p3)
        angt2 = angle_between_batch(p2-p1, p2-p3)
        angt3 = angle_between_batch(p3-p1, p3-p2)
        
        buf = torch.cat([dis1,dis2,dis3,ang1a,ang1b,ang2a,ang2b,ang3a,ang3b,angt1,angt2,angt3],dim=2)

        if self.scale_invariant:
            dis_index=[0,1,2]
            max_dis = buf[:,dis_index].max()
            buf[:,dis_index] = buf[:,dis_index]/max_dis
        return buf

    def extract_feature_C(self, points, norms, f_num):
        p_num = points.shape[1]
        idx = torch.randint(p_num, size=(f_num, 3))
        idx0=torch.arange(p_num).view(-1,1).repeat((1,f_num//p_num)).flatten()
        
        p1, p2, p3 = points[:,idx0], points[:,idx[:,1]], points[:,idx[:,2]]
        n1, n2, n3 = norms[:,idx0], norms[:,idx[:,1]], norms[:,idx[:,2]]
        dis1 = vlen(p1-p2)
        dis2 = vlen(p2-p3)
        dis3 = vlen(p3-p1)
        
        ang1a = angle_between_batch(p1-p2, n1)
        ang1b = angle_between_batch(p1-p3, n1)
        ang2a = angle_between_batch(p2-p1, n2)
        ang2b = angle_between_batch(p2-p3, n2)
        ang3a = angle_between_batch(p3-p1, n3)
        ang3b = angle_between_batch(p3-p2, n3)
        
        angt1 = angle_between_batch(p1-p2, p1-p3)
        angt2 = angle_between_batch(p2-p1, p2-p3)
        angt3 = angle_between_batch(p3-p1, p3-p2)
        
        buf = torch.cat([dis1,dis2,dis3,ang1a,ang1b,ang2a,ang2b,ang3a,ang3b,angt1,angt2,angt3],dim=2)
        
        mid = (p1+p2+p3)/3
        
        mn1 = angle_between_batch(p1-mid, n1)
        mn2 = angle_between_batch(p2-mid, n2)
        mn3 = angle_between_batch(p3-mid, n3)    
        
        dism1 = vlen(p1-mid)
        dism2 = vlen(p2-mid)
        dism3 = vlen(p3-mid)
        
        angm1a = angle_between_batch(p1-p2, p1-mid)
        angm1b = angle_between_batch(p1-p3, p1-mid)
        angm2a = angle_between_batch(p2-p1, p2-mid)
        angm2b = angle_between_batch(p2-p3, p2-mid)
        angm3a = angle_between_batch(p3-p1, p3-mid)
        angm3b = angle_between_batch(p3-p2, p3-mid)
        
        if self.scale_invariant:
            dis_index=[0,1,2,15,16,17]
            max_dis = buf[:,dis_index].max()
            buf[:,dis_index] = buf[:,dis_index]/max_dis

        return torch.cat([buf, mn1, mn2, mn3, dism1, dism2, dism3, angm1a, angm1b, angm2a, angm2b, angm3a, angm3b], dim=2)


    def forward(self, points, norms):
        batch, n_points = points.shape[0], points.shape[1]
       
        x = self.extractor(points, norms, self.feature_num)
        # t = time.time()
        
        x = x.transpose(2,1) #batch, 24, f_num
        feature = self.feat(x)#batch, 1024,f_num
        x_point = feature.view(batch,1024, n_points, self.feature_num//n_points) #32,1024,16,256
        x_point = torch.max(x_point, 3)[0] #32,1024,16
        x_global = torch.max(x_point, 2)[0]
        x = F.relu(self.bn1(self.fc1(x_global)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        # print(time.time()-t)
        if self.point_feature:
            return F.log_softmax(x, dim=1), x_global, x_point
        return F.log_softmax(x, dim=1), x_global




class TriangleNet_Seg(nn.Module):
    def __init__(self, part_num=50, inp=24, feature_num=4096, descriptor_type='C', scale_invariant=False):
        super(TriangleNet_Seg, self).__init__()
        self.feature_num = feature_num
        self.part_num = part_num
        self.scale_invariant = scale_invariant
        self.feat = Encoder(inp=inp)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        
        if descriptor_type == 'A':
            self.extractor = self.extract_feature_A
        elif descriptor_type == 'B':
            self.extractor = self.extract_feature_B
        elif descriptor_type == 'C':
            self.extractor = self.extract_feature_C
        else:
            raise Exception 


        self.seg_net = nn.Sequential(
            nn.Conv1d(2048+16, 256,1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 256, 1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256,128,1),
            nn.ReLU(),
            nn.Conv1d(128, part_num,1)
        )

    def extract_feature_A(self, points, norms, f_num):
        p_num = points.shape[1]
        idx = torch.randint(p_num, size=(f_num, 3))
        idx0=torch.arange(p_num).view(-1,1).repeat((1,f_num//p_num)).flatten()
        p1, p2 = points[:,idx0], points[:,idx[:,1]]
        n1, n2 = norms[:,idx0], norms[:,idx[:,1]]
        dis1 = vlen(p1-p2)
        ang1a = angle_between_batch(p1-p2, n1)
        ang1b = angle_between_batch(p2-p1, n2)
        ang1c = angle_between_batch(n1, n2)
        buf = torch.cat([dis1,ang1a, ang1b, ang1c],dim=2)
        
        if self.scale_invariant:
            max_dis = buf[:,0].max()
            buf[:,0] = buf[:,0]/max_dis

        return buf

    def extract_feature_B(self, points, norms, f_num):
        p_num = points.shape[1]
        idx = torch.randint(p_num, size=(f_num, 2))

        idx0=torch.arange(p_num).view(-1,1).repeat((1,f_num//p_num)).flatten()
        p1, p2, p3 = points[:,idx0], points[:,idx[:,1]], points[:,idx[:,2]]
        n1, n2, n3 = norms[:,idx0], norms[:,idx[:,1]], norms[:,idx[:,2]]
        dis1 = vlen(p1-p2)
        dis2 = vlen(p2-p3)
        dis3 = vlen(p3-p1)
        
        ang1a = angle_between_batch(p1-p2, n1)
        ang1b = angle_between_batch(p1-p3, n1)
        ang2a = angle_between_batch(p2-p1, n2)
        ang2b = angle_between_batch(p2-p3, n2)
        ang3a = angle_between_batch(p3-p1, n3)
        ang3b = angle_between_batch(p3-p2, n3)
        
        angt1 = angle_between_batch(p1-p2, p1-p3)
        angt2 = angle_between_batch(p2-p1, p2-p3)
        angt3 = angle_between_batch(p3-p1, p3-p2)
        
        buf = torch.cat([dis1,dis2,dis3,ang1a,ang1b,ang2a,ang2b,ang3a,ang3b,angt1,angt2,angt3],dim=2)

        if self.scale_invariant:
            dis_index=[0,1,2]
            max_dis = buf[:,dis_index].max()
            buf[:,dis_index] = buf[:,dis_index]/max_dis
        return buf

    def extract_feature_C(self, points, norms, f_num):
        p_num = points.shape[1]
        idx = torch.randint(p_num, size=(f_num, 3))
        idx0=torch.arange(p_num).view(-1,1).repeat((1,f_num//p_num)).flatten()
        
        p1, p2, p3 = points[:,idx0], points[:,idx[:,1]], points[:,idx[:,2]]
        n1, n2, n3 = norms[:,idx0], norms[:,idx[:,1]], norms[:,idx[:,2]]
        dis1 = vlen(p1-p2)
        dis2 = vlen(p2-p3)
        dis3 = vlen(p3-p1)
        
        ang1a = angle_between_batch(p1-p2, n1)
        ang1b = angle_between_batch(p1-p3, n1)
        ang2a = angle_between_batch(p2-p1, n2)
        ang2b = angle_between_batch(p2-p3, n2)
        ang3a = angle_between_batch(p3-p1, n3)
        ang3b = angle_between_batch(p3-p2, n3)
        
        angt1 = angle_between_batch(p1-p2, p1-p3)
        angt2 = angle_between_batch(p2-p1, p2-p3)
        angt3 = angle_between_batch(p3-p1, p3-p2)
        
        buf = torch.cat([dis1,dis2,dis3,ang1a,ang1b,ang2a,ang2b,ang3a,ang3b,angt1,angt2,angt3],dim=2)
        
        mid = (p1+p2+p3)/3
        
        mn1 = angle_between_batch(p1-mid, n1)
        mn2 = angle_between_batch(p2-mid, n2)
        mn3 = angle_between_batch(p3-mid, n3)    
        
        dism1 = vlen(p1-mid)
        dism2 = vlen(p2-mid)
        dism3 = vlen(p3-mid)
        
        angm1a = angle_between_batch(p1-p2, p1-mid)
        angm1b = angle_between_batch(p1-p3, p1-mid)
        angm2a = angle_between_batch(p2-p1, p2-mid)
        angm2b = angle_between_batch(p2-p3, p2-mid)
        angm3a = angle_between_batch(p3-p1, p3-mid)
        angm3b = angle_between_batch(p3-p2, p3-mid)
        
        if self.scale_invariant:
            dis_index=[0,1,2,15,16,17]
            max_dis = buf[:,dis_index].max()
            buf[:,dis_index] = buf[:,dis_index]/max_dis

        return torch.cat([buf, mn1, mn2, mn3, dism1, dism2, dism3, angm1a, angm1b, angm2a, angm2b, angm3a, angm3b], dim=2)


    def forward(self, points, norms, label):
        batch, n_points = points.shape[0], points.shape[1]
    
        x = self.extractor(points, norms, self.feature_num)
    
        x = x.transpose(2,1) #batch, 24, f_num
        feature = self.feat(x)#batch, 1024,f_num
        x_point = feature.view(batch,1024, n_points, self.feature_num//n_points) #32,1024,16,256
        x_point = torch.max(x_point, 3)[0] #32,1024,16
        x_global = torch.max(x_point, 2)[0] #32,1024

        x_global_label = torch.cat([x_global,label],1) #1024+16
        expand = x_global_label.view(-1, 1024+16, 1).repeat(1, 1, n_points)
        concat = torch.cat([expand, x_point], 1) #32,2064,1024
        seg_result = self.seg_net(concat) #32, 50, 1024

        seg_result = seg_result.transpose(2, 1).contiguous() #32,1024,50
        seg_result = F.log_softmax(seg_result.view(-1, self.part_num), dim=-1)
        seg_result = seg_result.view(batch, n_points, self.part_num) # [B, N, 50]

        return seg_result
    
if __name__ == "__main__":
    net = TriangleNet(inp=24,k=40,descriptor_type='C',encoder_type="full").cuda()
    points = torch.randn(1,2048,3).cuda()
    normals = torch.randn(1,2048,3).cuda()
    count_parameters(net)
    net.eval()
    torch.cuda.synchronize()
    t = time.time()
    for i in range(1000):
        with torch.no_grad():
            out = net(points, normals)
    print(time.time()-t)