from lib.options import BaseOptions
from scripts.img2masks import img2masks
from scripts.img2strand import img2strand
from scripts.img2depth import img2depth

if __name__ == "__main__":
    opt = BaseOptions().parse()
    img2masks(opt)
    img2strand(opt)
    img2depth(opt)