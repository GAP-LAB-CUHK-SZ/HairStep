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
                            help="The path to the SAM checkpoint to use for mask generation.")
        parser.add_argument("--checkpoint_img2depth", type=str, default='./checkpoints/img2hairstep/img2depth.pth',
                            help="The path to the SAM checkpoint to use for mask generation.")

        parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

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
