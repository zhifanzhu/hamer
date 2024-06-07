import tqdm
import torch
import argparse
import os
import cv2
import numpy as np
import pandas as pd

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.datasets.vitdet_dataset import DEFAULT_MEAN, DEFAULT_STD
from hamer.utils import recursive_to
from hamer.utils.renderer import Renderer, cam_crop_to_full

from skimage.filters import gaussian
from hamer.datasets.utils import (convert_cvimg_to_tensor,
                    expand_to_aspect_ratio,
                    generate_image_patch_cv2)

from aria.fork_slam_reader import ForkRawSLAMReader

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)


""" Usage
python aria/run.py \
    --vid P01-20240202-110250 \
    --frame-start 1 \
    --frame-end 9999 \
    --bbox-file data/aria/ \
    --viz
"""


def parse_args():
    parser = argparse.ArgumentParser(description='HaMeR + ho_detector + VRS reading')
    parser.add_argument('--vid', type=str, required=True, help='video-id of the VRS file')
    parser.add_argument('--frame-start', type=int, required=True)
    parser.add_argument('--frame-end', type=int, required=True)
    parser.add_argument('--storage-dir', type=str, default='/media/eve/SCRATCH/Zhifan/eka_vista_kit/eka_storage' )
    parser.add_argument('--bbox-file', type=str, required=True)

    parser.add_argument('--viz', action='store_true', help='Visualize the output')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for dataloader')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--out_folder', type=str, default='data/aria', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    return parser.parse_args()


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 vid, 
                 storage_dir,
                 frame_start, 
                 frame_end):
        super().__init__()
        self.reader = ForkRawSLAMReader(
            vid, storage_dir, 
            load_pts=False, load_frame_traj=False, load_gaze=False)
        self.frame_start = frame_start
        self.frame_end = min(frame_end, self.reader.num_rgb_frames)
        self.dets_df = pd.read_csv(f'data/aria/{vid}.csv')

        self.data_infos = self.get_data_infos()
    
    def get_data_infos(self):
        data_infos = []
        for f in range(self.frame_start, self.frame_end):
            dets = self.get_det(f)
            for box, lr in dets:
                data_infos.append((f, box, lr))
        return data_infos

    def get_det(self, frame):
        ents = self.dets_df[self.dets_df.frame == frame]
        dets = []
        for i, (_frame, x0, y0, x1, y1, lr) in ents.iterrows():
            box = np.asarray([x0, y0, x1, y1])
            dets.append((box, int(lr)))
        return dets
    
    def __len__(self):
        return len(self.data_infos)
    
    def __getitem__(self, idx):
        frame, box, right = self.data_infos[idx]
        img = self.reader.read_rgb_frame(
            frame, ret_timestamp_ns=False, ret_record=False, undistort=True)
        item = self.get_data_item(img, box, right)
        item['frame'] = frame
        return item 

    def get_data_item(self,
                      img,
                      box,
                      right,
                      rescale_factor=2.5):
        IMAGE_SIZE=256
        IMAGE_MEAN = [0.485, 0.456, 0.406]
        IMAGE_STD = [0.229, 0.224, 0.225]
        img_size = IMAGE_SIZE
        mean = 255. * np.array(IMAGE_MEAN)
        std = 255. * np.array(IMAGE_STD)

        center = (box[2:4] + box[0:2]) / 2.0
        scale = rescale_factor * (box[2:4] - box[0:2]) / 200.0
        center_x = center[0]
        center_y = center[1]

        cvimg = img

        BBOX_SHAPE = [192, 256]   # self.cfg.MODEL.get('BBOX_SHAPE', None)
        bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=BBOX_SHAPE).max()

        patch_width = patch_height = img_size
        flip = right == 0

        if True:
            # Blur image to avoid aliasing artifacts
            downsampling_factor = ((bbox_size*1.0) / patch_width)
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
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - mean[n_c]) / std[n_c]

        item = {
            'img': img_patch,
        }
        item['box_center'] = center
        item['box_size'] = bbox_size
        item['img_size'] = 1.0 * np.array([cvimg.shape[1], cvimg.shape[0]])
        item['right'] = right
        return item


def main():
    args = parse_args()

    # Download and load checkpoints
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    renderer = Renderer(model_cfg, faces=model.mano.faces)

    dataset = TinyDataset(
        vid=args.vid, storage_dir=args.storage_dir, frame_start=args.frame_start, 
        frame_end=args.frame_end, )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Make output directory if it does not exist
    vid_outdir = os.path.join(args.out_folder, args.vid)
    os.makedirs(vid_outdir, exist_ok=True)

    for _i, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)

        multiplier = (2*batch['right']-1)
        pred_cam = out['pred_cam']
        pred_cam[:,1] = multiplier*pred_cam[:,1]
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        multiplier = (2*batch['right']-1)
        scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
        pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

        # Render the result
        batch_size = batch['img'].shape[0]
        frames = batch['frame']
        pred_cam_cpu = out['pred_cam'].cpu()
        pred_mano_params = out['pred_mano_params']
        global_orient = pred_mano_params['global_orient'].cpu()
        hand_pose = pred_mano_params['hand_pose'].cpu()
        betas = pred_mano_params['betas'].cpu()
        for n in range(batch_size):
            # Dumping
            params = {
                'pred_cam': pred_cam_cpu[n],
                'global_orient': global_orient[n],
                'hand_pose': hand_pose[n],
                'betas': betas[n],
            }
            save_path = os.path.join(vid_outdir, f'frame_{frames[n]:010d}.pt')
            all_params = torch.load(save_path) if os.path.exists(save_path) else {}  # in case there is another hand
            key = 'left' if batch['right'][n] == 0 else 'right'
            all_params[key] = params
            torch.save(all_params, save_path)

            if not args.viz:
                continue

            # Get filename from path img_path
            img_fn = f'frame_{frames[n]:010d}'
            person_id = 0
            # person_id = int(batch['personid'][n])
            white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
            input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
            input_patch = input_patch.permute(1,2,0).numpy()

            regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                    out['pred_cam_t'][n].detach().cpu().numpy(),
                                    batch['img'][n],
                                    mesh_base_color=LIGHT_BLUE,
                                    scene_bg_color=(1, 1, 1),
                                    )

            if args.side_view:
                side_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                        out['pred_cam_t'][n].detach().cpu().numpy(),
                                        white_img,
                                        mesh_base_color=LIGHT_BLUE,
                                        scene_bg_color=(1, 1, 1),
                                        side_view=True)
                final_img = np.concatenate([input_patch, regression_img, side_img], axis=1)
            else:
                final_img = np.concatenate([input_patch, regression_img], axis=1)

            cv2.imwrite(os.path.join(vid_outdir, f'{img_fn}_{person_id}.png'), 255*final_img[:, :, ::-1])


if __name__ == '__main__':
    main()
