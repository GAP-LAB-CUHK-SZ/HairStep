import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize
from skimage.io import imread

from lib.mesh_util import load_obj_mesh

def load_point_ids(mesh_file):
    vertex_data = []
    
    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            #v = list(map(float, values[1]))
            v = int(values[1])
            vertex_data.append(v)

    vert_ids = np.array(vertex_data).astype(int)

    return vert_ids

class OptLandmark(nn.Module):
    def __init__(self, lmk_gt_path, input_img_path, width=512):
        super(OptLandmark, self).__init__()
        self.head_obj_path = './data/head_model.obj'
        self.landmark_id_path = './data/landmark_id_uschair.obj'
        self.input_img_path = input_img_path

        self.width = width
        self.rendering_load_zize = 1024

        self.read_img()
        self.lmk_3D = self.load_lmk_3D()
        self.lmk_gt = self.load_lmk_gt(lmk_gt_path)

        self.lmk_loss = nn.MSELoss()

        self.ortho_ratio = 0.2
        self.register_buffer('scale', nn.Parameter(torch.Tensor([363.1])))
        self.register_buffer('center', nn.Parameter(torch.Tensor([[-0.001],[1.64],[-0.03]])))
        self.register_buffer('rotation', nn.Parameter(torch.Tensor([[0.9, 0., -0.04], [0.001, 0.99, 0.023], [0.037, -0.026, 0.89]])))

        self.register_parameter('scale_dis', nn.Parameter(torch.zeros_like(self.scale)))
        self.register_parameter('center_dis', nn.Parameter(torch.zeros_like(self.center)))
        self.register_parameter('rotation_dis', nn.Parameter(torch.zeros_like(self.rotation)))

    def load_lmk_3D(self):
        vertices, _ = load_obj_mesh(self.head_obj_path)
        landmark_ids = load_point_ids(self.landmark_id_path)

        lmk_3D = vertices[landmark_ids]
        lmk_3D = torch.Tensor(lmk_3D.T).float().unsqueeze(0).cuda()

        return lmk_3D
    
    def load_lmk_gt(self, lmk_gt_path):
        lmk_gt = np.load(lmk_gt_path)[:, :2]
        lmk_gt[:,0] = lmk_gt[:,0]*self.lmk_coeffi[0]
        lmk_gt[:,1] = lmk_gt[:,1]*self.lmk_coeffi[1]
        return torch.from_numpy(lmk_gt).float().cuda().unsqueeze(0)

    def save_param(self, outputpath):
        dic = {'ortho_ratio': self.ortho_ratio, 
            'scale': (self.scale+self.scale_dis).detach().cpu().numpy(), 
            'center': (self.center+self.center_dis).detach().cpu().numpy(), 
            'R': (self.rotation+self.rotation_dis).detach().cpu().numpy()}
        np.save(outputpath, dic)

    def get_camera(self):
        # loading calibration data
        # pixel unit / world unit
        ortho_ratio = self.ortho_ratio
        # world unit / model unit
        scale = self.scale + self.scale_dis
        # camera center world coordinate
        center = self.center + self.center_dis
        # model rotation
        R = self.rotation + self.rotation_dis

        translate = -torch.mm(R, center).reshape(3, 1)
        extrinsic = torch.cat((R, translate), 1)
        extrinsic = torch.cat((extrinsic, torch.Tensor([0, 0, 0, 1]).cuda().reshape(1, 4)), 0)
        # Match camera space to image pixel space
        scale_intrinsic = torch.eye(4).cuda()
        scale_intrinsic[0, 0] = scale / ortho_ratio
        scale_intrinsic[1, 1] = -scale / ortho_ratio
        scale_intrinsic[2, 2] = scale / ortho_ratio
        # Match image pixel space to image uv space
        uv_intrinsic = torch.eye(4).cuda()
        uv_intrinsic[0, 0] = 1.0 / float(self.rendering_load_zize // 2)
        uv_intrinsic[1, 1] = 1.0 / float(self.rendering_load_zize // 2)
        uv_intrinsic[2, 2] = 1.0 / float(self.rendering_load_zize // 2)
        # Transform under image pixel space
        trans_intrinsic = torch.eye(4).cuda()

        intrinsic = torch.mm(trans_intrinsic, torch.mm(uv_intrinsic, scale_intrinsic))
        calib = torch.mm(intrinsic, extrinsic).unsqueeze(0).float()

        return calib

    def orthogonal(self, points, calibrations, transforms=None):
        '''
        Compute the orthogonal projections of 3D points into the image plane by given projection matrix
        :param points: [B, 3, N] Tensor of 3D points
        :param calibrations: [B, 4, 4] Tensor of projection matrix
        :param transforms: [B, 2, 3] Tensor of image transform matrix
        :return: xyz: [B, 3, N] Tensor of xyz coordinates in the image plane
        '''
        rot = calibrations[:, :3, :3]
        trans = calibrations[:, :3, 3:4]
        pts = torch.baddbmm(trans, rot, points)  # [B, 3, N]
        if transforms is not None:
            scale = transforms[:2, :2]
            shift = transforms[:2, 2:3]
            pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
        return pts

    def lmk_proj(self, calibs):
        xyz = self.orthogonal(self.lmk_3D, calibs)
        xy = xyz[:, :2, :]
        z = xyz[:, 2:3, :]

        y_coor = xy.squeeze().permute(1,0)[:,0]#/2.0  #[-1,1]
        x_coor = xy.squeeze().permute(1,0)[:,1]#/2.0

        x_coor = x_coor*self.width/2+self.width/2-1
        y_coor = y_coor*self.width/2+self.width/2-1

        lmk_2D = torch.cat((y_coor.unsqueeze(1),x_coor.unsqueeze(1)),dim=1)

        return lmk_2D.unsqueeze(0)

    def read_img(self):
        #read target image
        images_gt = imread(self.input_img_path, as_gray=False)
        #images_gt = imageio.imread(self.input_img_path)
        self.lmk_coeffi = [float(self.width)/images_gt.shape[1],float(self.width)/images_gt.shape[0]]
        #images_gt = Image.fromarray(images_gt)
        images_gt = resize(images_gt,(self.width,self.width))
        self.images_gt = np.array(images_gt)*255.0

    
    def get_img_lmk(self):
        gt_lmk_coor = np.floor(self.lmk_gt.detach().cpu().numpy()[0]).astype(np.int32)
        pred_lmk_coor = np.floor(self.lmk_2D.detach().cpu().numpy()[0]).astype(np.int32)

        #print(images_gt.shape, gt_lmk_coor.shape, pred_lmk_coor.shape)
        gt_lmk_coor = np.clip(gt_lmk_coor,0,511)
        pred_lmk_coor = np.clip(pred_lmk_coor,0,511)
        self.images_gt[gt_lmk_coor[:,1], gt_lmk_coor[:,0], :3] = [255,0,0]
        self.images_gt[pred_lmk_coor[:,1], pred_lmk_coor[:,0], :3] = [0,0,255]

        return self.images_gt.astype(np.uint8)

    def forward(self):
        calibs = self.get_camera()
        self.lmk_2D = self.lmk_proj(calibs)

        #loss = self.lmk_loss(self.lmk_2D, self.lmk_gt)

        loss = torch.sum(torch.abs(self.lmk_2D - self.lmk_gt))

        return loss
