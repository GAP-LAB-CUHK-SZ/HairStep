import torch
import numpy as np
import imageio
from torch.autograd import Variable
import os
from tqdm import tqdm

from lib.options import BaseOptions
from lib.model.img2hairstep.hourglass import Model

import matplotlib.pyplot as plt


def depth2vis(mask, depth, path_output):
    masked_img = depth * mask + (1 - mask) * ((depth * mask - (1 - mask) * 100000).max())  # set the value of un-mask to the min-val in mask
    norm_masked_depth = masked_img / (np.nanmax(masked_img) - np.nanmin(masked_img))  # norm

    plt.imsave('temp.png', norm_masked_depth, cmap='jet')

    depth_map_vis = imageio.imread('temp.png')[..., 0:3] * np.repeat(mask[:,:,None], 3, axis=2)
    plt.imsave(path_output, depth_map_vis)

def img2depth(opt):
    print("convert image to depth map")
    resized_path = os.path.join(opt.root_real_imgs, 'resized_img')
    output_seg_path = os.path.join(opt.root_real_imgs, 'seg')
    output_depth_path = os.path.join(opt.root_real_imgs, 'depth_map')
    output_depth_vis_path = os.path.join(opt.root_real_imgs, 'depth_vis_map')

    os.makedirs(output_depth_path, exist_ok=True)
    os.makedirs(output_depth_vis_path, exist_ok=True)

    items = os.listdir(resized_path)

    model = Model().cuda()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(opt.checkpoint_img2depth))

    model.eval()

    for item in tqdm(items):
        rgb_img = imageio.imread(os.path.join(resized_path, item))[:,:, 0:3] / 255.
        mask = (imageio.imread(os.path.join(output_seg_path, item))/255.>0.5)

        rgb_img = rgb_img

        rgb_img = Variable(torch.from_numpy(rgb_img).permute(2, 0, 1).float().unsqueeze(0)).cuda()

        depth_pred = model(rgb_img)
        depth_pred = depth_pred.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

        depth_pred_masked = depth_pred[:, :, 0] * mask - (1 - mask) * (np.abs(np.nanmax(depth_pred)) + np.abs(np.nanmin(depth_pred)))
        max_val = np.nanmax(depth_pred_masked)
        min_val = np.nanmin(depth_pred_masked + 2 * (1 - mask) * (np.abs(np.nanmax(depth_pred)) + np.abs(np.nanmin(depth_pred))))
        depth_pred_norm = (depth_pred_masked - min_val) / (max_val - min_val)*mask
        depth_pred_norm = np.clip(depth_pred_norm, 0., 1.)

        np.save(os.path.join(output_depth_path, item[:-3]+'npy'), depth_pred_norm)

        depth2vis(mask, depth_pred_norm, os.path.join(output_depth_vis_path, item))

if __name__ == "__main__":
    opt = BaseOptions().parse()
    img2depth(opt)