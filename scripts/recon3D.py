import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torchvision.transforms as transforms

from tqdm import tqdm
import imageio
from PIL import Image

from lib.options import BaseOptions
from lib.mesh_util import *
from lib.hair_util import *
from lib.model.recon3D import *

def load_hairstep(orien2d_path, depth_path, seg_path, load_size=512):
    raw_orien2d = Image.open(orien2d_path).convert('RGB')
    img_to_tensor = transforms.Compose([
            transforms.Resize(load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    orien2d = img_to_tensor(raw_orien2d).float()

    out_mask = ((np.array(imageio.imread(seg_path))/255)<0.5)
    if len(out_mask.shape)==3:
        out_mask = out_mask[:,:,0]
    depth = np.load(depth_path)
    depth = depth + out_mask*opt.depth_out_mask
    depth = torch.from_numpy(depth).float()

    hairstep = torch.cat([orien2d, depth.unsqueeze(0)], dim=0)
        
    return hairstep

def load_calib(calib_path, loadSize=1024):
    # loading calibration data
    param = np.load(calib_path, allow_pickle=True)
    # pixel unit / world unit
    ortho_ratio = param.item().get('ortho_ratio')
    # world unit / model unit
    scale = param.item().get('scale')
    # camera center world coordinate
    center = param.item().get('center')
    # model rotation
    R = param.item().get('R')

    translate = -np.matmul(R, center).reshape(3, 1)
    extrinsic = np.concatenate([R, translate], axis=1)
    extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
    # Match camera space to image pixel space
    scale_intrinsic = np.identity(4)
    scale_intrinsic[0, 0] = scale / ortho_ratio
    scale_intrinsic[1, 1] = -scale / ortho_ratio
    scale_intrinsic[2, 2] = scale / ortho_ratio
    # Match image pixel space to image uv space
    uv_intrinsic = np.identity(4)
    uv_intrinsic[0, 0] = 1.0 / float(loadSize // 2)
    uv_intrinsic[1, 1] = 1.0 / float(loadSize // 2)
    uv_intrinsic[2, 2] = 1.0 / float(loadSize // 2)
    # Transform under image pixel space
    trans_intrinsic = np.identity(4)
    intrinsic = np.matmul(trans_intrinsic, np.matmul(uv_intrinsic, scale_intrinsic))
    calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()

    return calib

def load_occNet(cuda, opt):
    # create net
    net_path = opt.checkpoint_hairstep2occ
    net = HGPIFuNet_orien(opt).to(device=cuda)

    # load checkpoints
    print('loading for occNet ...', net_path)
    net.load_state_dict(torch.load(net_path, map_location=cuda))
    net.eval()

    return net

def load_orienNet(cuda, opt):
    # create net
    net_path = opt.checkpoint_hairstep2orien
    net = HGPIFuNet_orien(opt, gen_orien=True).to(device=cuda)

    # load checkpoints
    print('loading for orienNet ...', net_path)
    net.load_state_dict(torch.load(net_path, map_location=cuda))
    net.eval()

    return net

def recon3D_from_hairstep(opt):
    # set cuda
    cuda = torch.device('cuda:%d' % opt.gpu_id)

    seg_dir = os.path.join(opt.root_real_imgs, 'seg')
    depth_dir = os.path.join(opt.root_real_imgs, 'depth_map')
    strand_dir = os.path.join(opt.root_real_imgs, 'strand_map')
    calib_dir = os.path.join(opt.root_real_imgs, 'param')

    output_mesh_dir = os.path.join(opt.root_real_imgs, 'mesh')
    output_hair3D_dir = os.path.join(opt.root_real_imgs, 'hair3D')

    os.makedirs(output_mesh_dir, exist_ok=True)
    os.makedirs(output_hair3D_dir, exist_ok=True)

    occ_net = load_occNet(cuda, opt)
    orien_net = load_orienNet(cuda, opt)

    items = os.listdir(strand_dir)

    with torch.no_grad():
        for item in tqdm(items):
            strand_path = os.path.join(strand_dir, item[:-3] + 'png')
            seg_path = os.path.join(seg_dir, item[:-3] + 'png')
            mesh_path = os.path.join(output_mesh_dir, item[:-3] + 'obj')
            hair3D_path = os.path.join(output_hair3D_dir, item[:-3] + 'ply')
            calib_path = os.path.join(calib_dir, item[:-3] + 'npy')
            depth_path = os.path.join(depth_dir, item[:-3] + 'npy')

            if os.path.exists(hair3D_path):
                continue

            calib = load_calib(calib_path)
            hairstep = load_hairstep(strand_path, depth_path, seg_path, opt.loadSize)

            test_data = {'hairstep': hairstep, 'calib':calib}

            gen_mesh_real(opt, occ_net, cuda, test_data, mesh_path)
            export_hair_real(orien_net, cuda, test_data, mesh_path, hair3D_path)

if __name__ == '__main__':
    opt = BaseOptions().parse()
    recon3D_from_hairstep(opt)