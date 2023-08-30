import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasePIFuNet import BasePIFuNet
from .SurfaceClassifier import SurfaceClassifier
from .HGFilters import *
from lib.net_util import init_net
from .Embedder import *
from lib.geometry import orthogonal
import copy

def positionalEncoder(cam_points, embedder, output_dim):
    cam_points = cam_points.permute(0, 2, 1)
    inputs_flat = torch.reshape(cam_points, [-1, cam_points.shape[-1]])
    embedded = embedder(inputs_flat)
    output = torch.reshape(embedded, [cam_points.shape[0], cam_points.shape[1], output_dim])
    return output.permute(0, 2, 1)

class HGPIFuNet_orien(BasePIFuNet):
    '''
    HG PIFu network uses Hourglass stacks as the image filter.
    It does the following:
        1. Compute image feature stacks and store it in self.im_feat_list
            self.im_feat_list[-1] is the last stack (output stack)
        2. Calculate calibration
        3. If training, it index on every intermediate stacks,
            If testing, it index on the last stack.
        4. Classification.
        5. During training, error is calculated on all stacks.
    '''

    def __init__(self,
                 opt,
                 error_term=nn.L1Loss(),
                 gen_orien=False,
                 ):
        super(HGPIFuNet_orien, self).__init__(
            error_term=error_term)

        self.name = 'hgpifu'

        self.opt = opt
        self.gen_orien = gen_orien

        self.image_filter = HGFilter(opt, first_channel=4)

        self.orign_embedder = None
        self.embedder_outDim = None
        mlp_dim = copy.copy(self.opt.mlp_dim)

        self.orign_embedder, self.embedder_outDim = get_embedder(opt.freq)
        self.embedder = positionalEncoder
        mlp_dim[0] = mlp_dim[0] + self.embedder_outDim

        if self.gen_orien:
            mlp_dim[-1] = 3
            self.surface_classifier = SurfaceClassifier(
                filter_channels=mlp_dim,
                no_residual=self.opt.no_residual,
                last_op=None)
        else:
            self.surface_classifier = SurfaceClassifier(
                filter_channels=mlp_dim,
                no_residual=self.opt.no_residual,
                last_op=nn.Sigmoid())

        # This is a list of [B x Feat_i x H x W] features
        self.im_feat_list = []
        self.tmpx = None
        self.normx = None

        self.intermediate_preds_list = []

        init_net(self)

    def filter(self, images):
        '''
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        '''
        self.im_feat_list, self.tmpx, self.normx = self.image_filter(images)
        # If it is not in training, only produce the last im_feat
        if not self.training:
            self.im_feat_list = [self.im_feat_list[-1]]


    def query(self, points, calibs, depth_maps=None, transforms=None, labels=None):
        '''
        Given 3D points, query the network predictions for each point.
        Image features should be pre-computed before this call.
        store all intermediate features.
        query() function may behave differently during training/testing.
        :param points: [B, 3, N] world space coordinates of points
        :param calibs: [B, 3, 4] calibration matrices for each image
        :param transforms: Optional [B, 2, 3] image space coordinate transforms
        :param labels: Optional [B, Res, N] gt labeling
        :return: [B, Res, N] predictions for each point
        '''
        if labels is not None:
            self.labels = labels

        xyz = orthogonal(points, calibs, transforms)
        xy = xyz[:, :2, :]
        z = xyz[:, 2:3, :]

        if self.opt.vis_loss:
            depth_proj = self.index(depth_maps.unsqueeze(1), xy)
            vis_ones = torch.ones_like(depth_proj).to(device=depth_proj.device)
            vis_bool = torch.abs(depth_proj - z) < self.opt.depth_vis
            out_mask_bool = (depth_proj<=self.opt.depth_out_mask)
            self.vis_weight = torch.where(vis_bool, vis_ones*10, vis_ones).float()
            self.vis_weight = torch.where(out_mask_bool, vis_ones, self.vis_weight).float()

        in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)

        position_feat = self.embedder(points, self.orign_embedder, self.embedder_outDim)

        self.intermediate_preds_list = []
        for idx in range(len(self.im_feat_list)):
            im_feat = self.im_feat_list[idx]
            point_local_feat_list = [self.index(im_feat, xy), position_feat]
            point_local_feat = torch.cat(point_local_feat_list, 1)

            # out of image plane is always set to 0
            pred = in_img[:,None].float() * self.surface_classifier(point_local_feat)
            self.intermediate_preds_list.append(pred)

        self.preds = self.intermediate_preds_list[-1]

        return self.preds

    def get_im_feat(self):
        '''
        Get the image filter
        :return: [B, C_feat, H, W] image feature after filtering
        '''
        return self.im_feat_list[-1]

    def get_error(self):
        '''
        Hourglass has its own intermediate supervision scheme
        '''
        error = 0
        for preds in self.intermediate_preds_list:
            if self.opt.vis_loss:
                error += self.error_term(preds*self.vis_weight, self.labels*self.vis_weight)
            else:
                error += self.error_term(preds, self.labels)
        error /= len(self.intermediate_preds_list)
        
        return error

    def forward(self, images, points, calibs, depth_maps=None, transforms=None, labels=None):
        self.filter(images)

        self.query(points=points, calibs=calibs, depth_maps=depth_maps, transforms=transforms, labels=labels)

        res = self.get_preds()

        error = self.get_error()

        return res, error