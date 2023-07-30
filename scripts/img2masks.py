import os
import cv2
import numpy as np

from tqdm import tqdm
from skimage.transform import resize
from segment_anything import SamPredictor, sam_model_registry

from lib.options import BaseOptions


def pad_and_resize(img, width = 512):
    img = np.array(img)

    if img.shape[0] > img.shape[1]:
        img_pad = np.zeros((img.shape[0],int((img.shape[0]-img.shape[1])/2),img.shape[2]))
        padded_img = np.concatenate((img_pad, img ,img_pad),axis=1)
    else:
        img_pad = np.zeros((int((img.shape[1]-img.shape[0])/2),img.shape[1],img.shape[2]))
        padded_img = np.concatenate((img_pad, img ,img_pad),axis=0)

    padded_img = resize(padded_img,(width,width)).astype(np.uint8)

    padded_img = cv2.merge([padded_img[:,:,0], padded_img[:,:,1], padded_img[:,:,2]])

    return padded_img

def write_mask_to_folder(mask, filename):
    if mask.shape[0]==3:
        mask = mask.transpose(1,2,0)
        mask = ((mask[:,:,0]+mask[:,:,1]+mask[:,:,2])>0)[:,:,None]
        mask = np.concatenate([mask,mask,mask], axis=2)
    else:
        mask = mask.transpose(1,2,0)>0
    cv2.imwrite(filename, mask * 255)


def img2masks(opt):
    print("segment hair mask and body mask")
    sam = sam_model_registry[opt.model_type_sam](checkpoint=opt.checkpoint_sam)
    _ = sam.to(device=opt.device)

    input_path = os.path.join(opt.root_real_imgs, 'img')
    resized_path = os.path.join(opt.root_real_imgs, 'resized_img')
    output_seg_path = os.path.join(opt.root_real_imgs, 'seg')
    output_body_path = os.path.join(opt.root_real_imgs, 'body_img')

    os.makedirs(resized_path, exist_ok=True)
    os.makedirs(output_seg_path, exist_ok=True)
    os.makedirs(output_body_path, exist_ok=True)

    predictor = SamPredictor(sam)

    targets = os.listdir(input_path)

    for t in tqdm(targets):
        image = cv2.imread(os.path.join(input_path, t))

        if image is None:
            print(f"Could not load '{t}' as an image, skipping...")
            continue

        resized_image = pad_and_resize(image)
        cv2.imwrite(os.path.join(resized_path, t), resized_image)

        predictor.set_image(resized_image)

        body_masks, _, _ = predictor.predict(point_coords=np.array([[255,255]]), point_labels=np.array([1]), multimask_output=True)
        write_mask_to_folder(body_masks, os.path.join(output_body_path, t))

        hair_pt_x = np.where(np.sum(body_masks[:,:,255],axis=0)>0)[0][0]
        hair_pt = np.array([[255,hair_pt_x+5]])
        body_pt = np.array([[255,255]])
        pts = np.concatenate([hair_pt, body_pt], axis=0)
        hair_masks, _, _ = predictor.predict(point_coords=pts, point_labels=np.array([1,0]), multimask_output=False)
        write_mask_to_folder(hair_masks, os.path.join(output_seg_path, t))


if __name__ == "__main__":
    opt = BaseOptions().parse()
    img2masks(opt)