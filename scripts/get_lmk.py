'''
Description: in: RGB image, out: pose, lms 
Run-Cmd: 2021/12/26
python Tools/get_pose_lms_v1.py /share_graphics_ai/limengtian/hair-growth-data-1209/data_crop/ /home/limoran/lmr_hair_vae/DATA/real_test_female_crop/randomsample3k_to_3k200_fname.txt /home/limoran/lmr_hair_vae/DATA/real_test_female_crop/randomsample3k_to_3k200_fname_pose_lms.pkl
python Tools/get_pose_lms_v1.py {} {} {}
'''
import os, sys, cv2, copy
import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'external/3DDFA_V2/'))
from TDDFA import TDDFA
from FaceBoxes import FaceBoxes
from utils.pose import calc_pose

# del sys.path[0]
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from lib.options import BaseOptions

def GetPoseLms2d(imgs):
    config_path = 'configs/mb1_120x120.yml'
    # config_path = './external/3DDFA_V2/configs/mb1_120x120.yml'
    cfg = yaml.load(open(config_path), Loader=yaml.SafeLoader)

    gpu_mode= True
    tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
    face_boxes = FaceBoxes()
    dense_flag = False
    lms2d_all = []
    pose_all = []
    for i in tqdm(range(len(imgs))):
        cur_img = imgs[i]
        boxes = face_boxes(cur_img)
        n = len(boxes)
        if n <=0:
            pose_all.append(-1.0 * np.ones([3]))
            lms2d_all.append(-1.0 * np.ones([68,3]))
            continue 
        param_lst, roi_box_lst = tddfa(cur_img, boxes)
        ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
        P, pose = calc_pose(param_lst[0]) 
        pts = ver_lst[0].transpose(1, 0)
        lms2d_all.append(pts)
        pose_all.append(pose)
    return np.asarray(lms2d_all), np.asarray(pose_all)      
  
def RecoverLandmarkToImage(lms, org_size, size):
    org_h, org_w = org_size
    ans_lms = copy.deepcopy(lms)
    ans_lms[:, 0] = ans_lms[:, 0]/size[0] * org_w
    ans_lms[:, 1] = ans_lms[:, 1]/size[1] * org_h
    return ans_lms
                           
if __name__ == '__main__':
    opt = BaseOptions().parse()

    img_dir = os.path.join(opt.root_real_imgs, 'resized_img')
    lmk_dir = os.path.join(opt.root_real_imgs, 'lmk')

    os.makedirs(lmk_dir, exist_ok=True)

    fname_list_full = os.listdir(img_dir)
    fname_list = []
    for i in range(len(fname_list_full)):
        if os.path.exists(os.path.join(lmk_dir, fname_list_full[i][:-3]+'npy')):
            continue
        fname_list.append(fname_list_full[i])

    img_list = []
    size = 256
    org_img_size = []
    print('load imgs')
    for i in tqdm(range(len(fname_list))):
        org_img = cv2.imread(os.path.join(img_dir, fname_list[i]))
        org_img_size.append(org_img.shape[:2])
        img = cv2.resize(org_img, (size, size))
        img_list.append(img)
    print('get lmks')


    os.chdir('./external/3DDFA_V2/')

    lms_pred, pose_pred = GetPoseLms2d(img_list)

    os.chdir('../../')
    
    print('save lmks')
    for i in tqdm(range(len(lms_pred))):
        cur_lms_pred = lms_pred[i]
        cur_lms_r = RecoverLandmarkToImage(lms_pred[i], org_img_size[i], (size, size))
        np.save(os.path.join(lmk_dir, fname_list[i][:-3]+'npy'), cur_lms_r)
