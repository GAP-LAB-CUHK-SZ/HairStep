import argparse
import os


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # Datasets related
        parser.add_argument("--root_real_imgs", type=str, default='./results/real_imgs/', 
                            help="Path to a folder of images.")

        parser.add_argument("--model_type_sam", type=str, default='vit_h',
                            help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']")

        parser.add_argument("--checkpoint_sam", type=str, default='./checkpoints/SAM-models/sam_vit_h_4b8939.pth',
                            help="The path to the SAM checkpoint to use for mask generation.")
        parser.add_argument("--checkpoint_img2strand", type=str, default='./checkpoints/img2hairstep/img2strand.pth',
                            help="The path to the checkpoint of img2strand.")
        parser.add_argument("--checkpoint_img2depth", type=str, default='./checkpoints/img2hairstep/img2depth.pth',
                            help="The path to the checkpoint of img2depth.")
        parser.add_argument("--checkpoint_hairstep2occ", type=str, default='./checkpoints/recon3D/occNet',
                            help="The path to the checkpoint of hairstep2occ.")
        parser.add_argument("--checkpoint_hairstep2orien", type=str, default='./checkpoints/recon3D/orienNet',
                            help="The path to the checkpoint of hairstep2orien.")

        parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")
        parser.add_argument('--loadSize', type=int, default=512, help='load size of input image')
        
        # Training related
        g_train = parser.add_argument_group('Training')
        g_train.add_argument('--gpu_id', type=int, default=0, help='gpu id for cuda')
        g_train.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2, -1 for CPU mode')

        g_train.add_argument('--num_threads', default=8, type=int, help='# sthreads for loading data')
        g_train.add_argument('--serial_batches', action='store_true',
                             help='if true, takes images in order to make batches, otherwise takes them randomly')
        g_train.add_argument('--pin_memory', action='store_true', help='pin_memory')
        
        g_train.add_argument('--batch_size', type=int, default=1, help='input batch size')
        g_train.add_argument('--learning_rate', type=float, default=1e-4, help='adam learning rate')
        g_train.add_argument('--num_epoch', type=int, default=1001, help='num epoch to train')

        g_train.add_argument('--freq_plot', type=int, default=10, help='freqency of the error plot')
        g_train.add_argument('--freq_save', type=int, default=50, help='freqency of the save_checkpoints')
        g_train.add_argument('--freq_save_ply', type=int, default=100, help='freqency of the save ply')
        g_train.add_argument('--freq_test', type=int, default=100, help='freqency of testing')

        g_train.add_argument('--no_gen_mesh', action='store_true')
        g_train.add_argument('--no_num_eval', action='store_true')
        g_train.add_argument('--vis_loss', action='store_true')
        
        #different input
        g_train.add_argument('--depth_out_mask', type=float, default=-3.0)
        g_train.add_argument('--depth_vis', type=float, default=0.1)

        g_train.add_argument('--freq', type=int, default=3, help='frequency of positional encoding')
        
        g_train.add_argument('--resume_epoch', type=int, default=-1, help='epoch resuming the training')
        g_train.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')

        # Testing related
        g_test = parser.add_argument_group('Testing')
        g_test.add_argument('--resolution', type=int, default=256, help='# of grid in mesh reconstruction')

        # Sampling related
        g_sample = parser.add_argument_group('Sampling')
        g_sample.add_argument('--sigma', type=float, default=5.0, help='perturbation standard deviation for positions')
        g_sample.add_argument('--num_sample_inout', type=int, default=5000, help='# of sampling points')

        # Model related
        g_model = parser.add_argument_group('Model')
        # General
        g_model.add_argument('--norm', type=str, default='group',
                             help='instance normalization or batch normalization or group normalization')
        # hg filter specify
        g_model.add_argument('--num_stack', type=int, default=4, help='# of hourglass')
        g_model.add_argument('--num_hourglass', type=int, default=2, help='# of stacked layer of hourglass')
        g_model.add_argument('--skip_hourglass', action='store_true', help='skip connection in hourglass')
        g_model.add_argument('--hg_down', type=str, default='ave_pool', help='ave pool || conv64 || conv128')
        g_model.add_argument('--hourglass_dim', type=int, default='256', help='256 | 512')

        # Classification General
        g_model.add_argument('--mlp_dim', nargs='+', default=[256, 1024, 512, 256, 128, 1], type=int,
                             help='# of dimensions of mlp')
        g_model.add_argument('--use_tanh', action='store_true',
                             help='using tanh after last conv of image_filter network')

        # for train
        parser.add_argument('--finetune', action='store_true', help='if generate orientation')
        parser.add_argument('--no_residual', action='store_true', help='no skip connection in mlp')
        parser.add_argument('--schedule', type=int, nargs='+', default=[800, 1500],
                            help='Decrease learning rate at these epochs.')
        parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')

        # special tasks
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        opt = self.gather_options()
        return opt
