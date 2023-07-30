import torch
import numpy as np
import imageio
from torch.autograd import Variable
import os
from tqdm import tqdm

from lib.options import BaseOptions
from lib.model.img2hairstep.UNet import Model

def img2strand(opt):
    print("convert image to strand map")
    resized_path = os.path.join(opt.root_real_imgs, 'resized_img')
    output_seg_path = os.path.join(opt.root_real_imgs, 'seg')
    output_body_path = os.path.join(opt.root_real_imgs, 'body_img')
    output_strand_path = os.path.join(opt.root_real_imgs, 'strand_map')

    os.makedirs(output_strand_path, exist_ok=True)

    items = os.listdir(resized_path)

    model = Model().cuda()
    model.load_state_dict(torch.load(opt.checkpoint_img2strand))

    model.eval()

    for item in tqdm(items):
        rgb_img = imageio.imread(os.path.join(resized_path, item))[:,:, 0:3] / 255.
        mask = (imageio.imread(os.path.join(output_seg_path, item))/255.>0.5)[:,:,None]
        body = ((imageio.imread(os.path.join(output_body_path, item))[:,:,0]/255.>0.5)[:,:,None])*(1-mask)

        rgb_img = rgb_img*mask

        rgb_img = Variable(torch.from_numpy(rgb_img).permute(2, 0, 1).float().unsqueeze(0)).cuda()

        strand_pred = model(rgb_img)
        strand_pred = np.clip(strand_pred.permute(0, 2, 3, 1)[0].cpu().detach().numpy(), 0., 1.)  # 512 * 512 *60

        strand_pred = np.concatenate([mask+body*0.5, strand_pred*mask], axis=-1)
        # strand_pred = np.concatenate([mask, strand_pred * mask], axis=-1)

        imageio.imwrite(os.path.join(output_strand_path, item), (strand_pred*255).astype(np.uint8))

if __name__ == "__main__":
    opt = BaseOptions().parse()
    img2strand(opt)