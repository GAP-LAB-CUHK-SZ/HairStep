import imageio
import os
from tqdm import tqdm
import torch

from lib.options import BaseOptions
from lib.util.opt_lmk import OptLandmark
from lib.train_util import adjust_learning_rate

def opt_cam(sample_info):
    #os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    if os.path.exists(sample_info[2]):
        return
    input_image = sample_info[0]
    target_lmk = sample_info[1]
    output_param = sample_info[2]
    output_img = sample_info[3]

    lr = 0.01
    num_epoch = 201
    schedule = [100,150]

    # load mesh and transform to camera space
    lmk_opt = OptLandmark(target_lmk, input_image).cuda()

    optimizer = torch.optim.Adam(lmk_opt.parameters(), lr, betas=(0.5, 0.99))

    #loop = tqdm.tqdm(list(range(0, num_epoch)))
    loop = list(range(0, num_epoch))

    for i in loop:
        loss = lmk_opt.forward()

        #loop.set_description('Loss: %.4f'% (loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr = adjust_learning_rate(optimizer, i, lr, schedule, 0.5)

        if i == (num_epoch-1):
            image = lmk_opt.get_img_lmk()
            lmk_opt.save_param(output_param)
            imageio.imwrite(output_img, image)
if __name__ == '__main__':
    opt = BaseOptions().parse()

    img_dir = os.path.join(opt.root_real_imgs, 'resized_img')
    lmk_dir = os.path.join(opt.root_real_imgs, 'lmk')
    param_dir = os.path.join(opt.root_real_imgs, 'param')
    lmk_proj_dir = os.path.join(opt.root_real_imgs, 'lmk_proj')


    os.makedirs(param_dir, exist_ok=True)
    os.makedirs(lmk_proj_dir, exist_ok=True)

    items = os.listdir(img_dir)

    sample_info = [(img_dir +'/'+ item, lmk_dir +'/'+ item[:-3]+'npy', 
                    param_dir +'/'+ item[:-3]+'npy', lmk_proj_dir +'/'+ item) for item in items]

    for sample in tqdm(sample_info):
        opt_cam(sample)
