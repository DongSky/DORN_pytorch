import torch
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as multiprocessing
import numpy as np
from PIL import Image
import os


def depth_loader(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    depth_png = np.array(Image.open(filename), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)
    depth = depth_png.astype(np.float) / 256.
    depth[depth_png == 0] = -1.
    return depth

def img_loader(filename):
    img = np.array(Image.open(filename), dtype=float)
    img = img.transpose(2, 0, 1)
    return img

def path_convert_to_raw(filename):
    piece = filename.split("/")
    #["2011_xx_xx_drive_xxxx_sync", "proj_depth", "groundtruth",
    # "image_02 or image_03", "0000000005.png"]
    raw_path = os.path.join((piece[0], piece[3], "data", piece[4]))
    return raw_path

class KITTI_Depth_Dataset(Dataset):
    def __init__(self, raw_root_folder, depth_root_folder, source_path):
        super(KITTI_Depth_Dataset, self).__init__()
        self.raw_root_folder = raw_root_folder
        self.depth_root_folder = depth_root_folder
        self.source_path = source_path
        self.load_metas()

    def load_metas(self):
        with open(self.source_path, "r") as f:
            lines = f.readlines()
        self.num = len(lines)
        manager = multiprocessing.Manager()
        self.metas = manager.list()
        for i, line in enumerate(lines):
            self.metas.append((path_convert_to_raw(line), line)
        print("read metas done")
