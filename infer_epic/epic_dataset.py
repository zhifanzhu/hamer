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


class EPICDataset(torch.utils.data.Dataset):
    """ EPIC Grasp dataset using HOS v2 boxes
    batch infer in per-vid basis
    """

    def __init__(self,
                 cfg: CfgNode,
                 vid: str,
                 start_frame: int,
                 end_frame: int,
                 step_size=1,
                 rescale_factor=2.5,
                 train: bool = False,
                 **kwargs):
        super().__init__()
        self.cfg = cfg
        self.vid = vid
        self.step_size = step_size  # 60FPS / 5 = 12FPS
        self.hos_root='/media/barry/DATA/Zhifan/Sid-hand-objects/'
        self.rgb_root='/media/skynet/DATA/Datasets/epic-100/rgb'
        self.image_format = osp.join(self.rgb_root, '%s/%s/frame_%010d.jpg')  # resolution is 456x256
        self.data_infos = self.preprocess_data_infos(start_frame, end_frame)

        assert train == False, "ViTDetDataset is only for inference"
        self.train = train
        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)

        self.rescale_factor = rescale_factor

    def preprocess_data_infos(self, start_frame, end_frame):
        """ returns: list of (frame, box=(4,), right=0/1)
        box is xyxy on 456x256 image
        if no hos hand box in that frame, skip
        """
        HAND_BBOX = 'hand_bbox'
        HAND_SIDE = 'hand_side'

        data_infos = []
        hos = self.get_hos(self.vid)
        epic_is_right = {'left hand': 0, 'right hand': 1}
        hos_is_right = {'left_hand': 0, 'right_hand': 1}
        for frame in range(start_frame, end_frame+1, self.step_size):
            if frame not in hos:
                continue

            preds = hos[frame]
            for side in ('left hand', 'right hand'):
                right = epic_is_right[side]
                for pred in preds:
                    if hos_is_right[pred[HAND_SIDE]] != right:
                        continue
                    data_infos.append((frame, np.asarray(pred[HAND_BBOX], dtype=np.float32), right))
        return data_infos

    @lru_cache(maxsize=8)
    def get_hos(self, vid):
        def hos_mapping(raw_json: dict) -> dict:
            """ return: {frame[int]: preds[list]} """
            import re
            get_int = lambda s : int(re.search('\d{10}', s).group(0))
            return {get_int(blob['file_name']): blob['predictions'] for blob in raw_json['images'] }
        hos_path = osp.join(self.hos_root, f'{vid}.json')
        with open(hos_path, 'r') as fp:
            hos_dict = hos_mapping(json.load(fp))
        return hos_dict

    def __len__(self) -> int:
        return len(self.data_infos)

    def __getitem__(self, idx: int) -> Dict[str, np.array]:
        vid = self.vid
        frame, box, right = self.data_infos[idx]
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
    from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
    model, model_cfg = load_hamer(DEFAULT_CHECKPOINT)

    dataset = EPICDataset(
        model_cfg, vid='P01_01', start_frame=28802, end_frame=28802, rescale_factor=2.0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    elem = next(iter(dataloader))
    print(elem)
