from functools import lru_cache
from typing import Dict

import tqdm
import json
import os.path as osp
import cv2
import numpy as np
from skimage.filters import gaussian
from yacs.config import CfgNode
import torch
import pandas as pd

from hamer.datasets.utils import (convert_cvimg_to_tensor,
                    expand_to_aspect_ratio,
                    generate_image_patch_cv2)

DEFAULT_MEAN = 255. * np.array([0.485, 0.456, 0.406])
DEFAULT_STD = 255. * np.array([0.229, 0.224, 0.225])


EPIC_HOA_SIZE = (1920, 1080)

class GripDatasetHOV1(torch.utils.data.Dataset):
    """ EPIC Grasp dataset using HOS v1 boxes
    batch infer in per-vid basis
    """

    def __init__(self,
                 cfg: CfgNode,
                 hoa_cache_path='/media/barry/DATA/Zhifan/epic_hor_data/cache/hoa_hbox.pth',
                 step_size=1,
                 rescale_factor=2.5,
                 train: bool = False,
                 **kwargs):
        super().__init__()
        self.cfg = cfg
        self.step_size = step_size  # 60FPS / 5 = 12FPS
        self.rgb_root='/media/skynet/DATA/Datasets/epic-100/rgb'
        self.csv_path = 'data/epichor_round3_2447valid_nonempty.csv'
        self.image_format = osp.join(self.rgb_root, '%s/%s/frame_%010d.jpg')

        self.hoa_hbox = torch.load(hoa_cache_path)
        self.data_infos = self.preprocess_data_infos()

        assert train == False, "ViTDetDataset is only for inference"
        self.train = train
        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)

        self.rescale_factor = rescale_factor

    def preprocess_data_infos(self):
        """ returns: list of (vid, frame, box=(4,), right=0/1)
        return box is xyxy on 456x256 image
        """
        data_infos = []

        df = pd.read_csv(self.csv_path)
        box_scale = np.float32([456, 256] * 2) / np.float32([1920, 1080] * 2)
        for i, row in df.iterrows():
            mapping = self.hoa_hbox[row['mp4_name']]
            vid = row['vid']
            right = 'right' in row.handside
            for frame, box_xywh in mapping.items():
                x1, y1, w, h = box_xywh
                box_xyxy = np.array([x1, y1, x1+w, y1+h]) * box_scale
                info = (vid, frame, box_xyxy, right)
                data_infos.append(info)

        return data_infos

    def __len__(self) -> int:
        return len(self.data_infos)

    def __getitem__(self, idx: int) -> Dict[str, np.array]:
        vid, frame, box, right = self.data_infos[idx]
        center = (box[2:4] + box[0:2]) / 2.0
        scale = self.rescale_factor * (box[2:4] - box[0:2]) / 200.0
        center_x = center[0]
        center_y = center[1]

        img_path = self.image_format % (vid[:3], vid, frame)
        cvimg = cv2.imread(img_path)

        BBOX_SHAPE = self.cfg.MODEL.get('BBOX_SHAPE', None)
        bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=BBOX_SHAPE).max()

        patch_width = patch_height = self.img_size

        flip = right == 0

        # 3. generate image patch
        # if use_skimage_antialias:
        if True:
            # Blur image to avoid aliasing artifacts
            downsampling_factor = ((bbox_size*1.0) / patch_width)
            # print(f'{downsampling_factor=}')
            downsampling_factor = downsampling_factor / 2.0
            if downsampling_factor > 1.1:
                cvimg  = gaussian(cvimg, sigma=(downsampling_factor-1)/2, channel_axis=2, preserve_range=True)

        img_patch_cv, trans = generate_image_patch_cv2(cvimg,
                                                    center_x, center_y,
                                                    bbox_size, bbox_size,
                                                    patch_width, patch_height,
                                                    flip, 1.0, 0,
                                                    border_mode=cv2.BORDER_CONSTANT)
        img_patch_cv = img_patch_cv[:, :, ::-1]
        img_patch = convert_cvimg_to_tensor(img_patch_cv)

        # apply normalization
        for n_c in range(min(cvimg.shape[2], 3)):
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]

        item = {
            'vid': vid,
            'frame': frame,
            'img': img_patch,
        }
        item['box_center'] = center
        item['box_size'] = bbox_size
        item['img_size'] = 1.0 * np.array([cvimg.shape[1], cvimg.shape[0]])
        item['right'] = right
        return item

if __name__ == '__main__':
        # parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
    model, model_cfg = load_hamer(DEFAULT_CHECKPOINT)

    dataset = GripDatasetHOV1(model_cfg, rescale_factor=2.0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    # for batch in dataloader:
    #     pass
    elem = next(iter(dataloader))
    print(elem)
