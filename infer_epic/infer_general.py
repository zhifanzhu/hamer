import torch
import argparse
import os
import cv2
import numpy as np
import tqdm

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full
from infer_epic.epic_dataset import EPICDataset

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)


def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--vid', type=str, help='Video ID to run inference on')
    parser.add_argument('--start-frame', type=int, required=True)
    parser.add_argument('--end-frame', type=int, required=True)
    parser.add_argument('--step_size', type=int, default=1, help='Step size for frame extraction')
    parser.add_argument('--dump_dir', type=str, default='data/hamer_hov2', help='Directory to dump pose outputs')
    parser.add_argument('--viz', action='store_true', help='Visualize the output')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for dataloader')

    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')

    args = parser.parse_args()

    # Download and load checkpoints
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # Make output directory if it does not exist
    if args.viz:
        os.makedirs(args.out_folder, exist_ok=True)

    # Run reconstruction on all detected hands
    vid = args.vid
    dataset = EPICDataset(
        model_cfg, vid=vid, start_frame=args.start_frame, end_frame=args.end_frame,
        rescale_factor=args.rescale_factor, step_size=args.step_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    vid_outdir = os.path.join(args.dump_dir, vid)
    os.makedirs(vid_outdir, exist_ok=True)

    for _i, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        # print("Progress: [{}/{}]".format(_i, len(dataloader)))
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
            # img_fn, _ = os.path.splitext(os.path.basename(img_path))
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

            cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_{person_id}.png'), 255*final_img[:, :, ::-1])

            # Add all verts and cams to list
            verts = out['pred_vertices'][n].detach().cpu().numpy()
            is_right = batch['right'][n].cpu().numpy()
            verts[:,0] = (2*is_right-1)*verts[:,0]
            cam_t = pred_cam_t_full[n]

            # Save all meshes to disk
            if args.save_mesh:
                camera_translation = cam_t.copy()
                tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE, is_right=is_right)
                tmesh.export(os.path.join(args.out_folder, f'{img_fn}_{person_id}.obj'))


if __name__ == '__main__':
    main()
