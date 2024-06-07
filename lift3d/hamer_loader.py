""" Load hamer from disk """

import os
import numpy as np
import torch
from manopth.manolayer import ManoLayer
from trimesh import Trimesh
from hamer.datasets.utils import expand_to_aspect_ratio
# from epic_reader_lib.reader import EpicImageReader
from lift3d.hos_getter import HOSGetter
from lift3d.rot_cvt import matrix_to_axis_angle


class HamerLoader:
    """ Load hamer from disk """

    def __init__(self, 
                 load_dir='./data/hamer_hov2',
                 mano_root='externals/mano',
                 img_reader=False):
        """
        Args:
            load_dir: e.g. <load_dir>/P01_01/frame_0000012345.pt
        """
        self.load_dir = load_dir
        self.hosgetter = HOSGetter()
        if img_reader:
            from epic_reader_lib.reader import EpicImageReader
            self.reader = EpicImageReader()

        self.left_mano = ManoLayer(
            flat_hand_mean=True, ncomps=45, side='left',
            mano_root=mano_root, use_pca=False)
        self.right_mano = ManoLayer(
            flat_hand_mean=True, ncomps=45, side='right',
            mano_root=mano_root, use_pca=False)
    
    def load_frame_all_params(self, vid: str, frame: int) -> dict:
        """ Load both hand params from *.pt file """
        return torch.load(f'{self.load_dir}/{vid}/frame_{frame:010d}.pt')

    def avail_frames(self, vid: str) -> list:
        """ Return available frames for a video """
        return sorted([
            int(f.split('.')[0].split('_')[-1]) 
            for f in os.listdir(f'{self.load_dir}/{vid}')])
    
    def has_frame(self, vid: str, frame: int) -> bool:
        """ Check if a frame is available """
        return os.path.exists(f'{self.load_dir}/{vid}/frame_{frame:010d}.pt')
    
    def load_mhand(self, 
                   vid: str, 
                   frame: int, 
                   hand_side: str,
                   epic_focal=5000, 
                   out_h=256,
                   out_w=456,
                   target_volume=375.65):
        """
        Args:
            target_volume: Mean hand volume in cc.
        """
        if not self.has_frame(vid, frame):
            return None
        all_params = torch.load(f'{self.load_dir}/{vid}/frame_{frame:010d}.pt')
        if hand_side not in all_params:
            return None
        params = all_params[hand_side]
        lbox, rbox = self.hosgetter.get_frame_hbox(vid, frame)

        # glb_orient = rot_cvt.matrix_to_axis_angle(params['global_orient'])
        # thetas = rot_cvt.matrix_to_axis_angle(params['hand_pose']).view([1, 45])
        glb_orient = matrix_to_axis_angle(params['global_orient'])
        thetas = matrix_to_axis_angle(params['hand_pose']).view([1, 45])
        thetas = torch.cat([glb_orient, thetas], axis=1)
        betas = params['betas'].view([1, 10])

        if hand_side == 'left':
            mano = self.left_mano
            box = lbox
            # The following is to flip the hand (flipping y and z)
            thetas = thetas.view(16, 3)
            thetas[:, 1:] *= -1
            thetas = thetas.view([1, 48])
        elif hand_side == 'right':
            mano = self.right_mano
            box = rbox
        box_scale = np.float32([out_w, out_h, out_w, out_h]) / \
            np.float32([456, 256, 456, 256])
        box *= box_scale

        vh, jh = mano.forward(thetas, betas)
        vh /= 1000.
        jh /= 1000.
        # vh *= mag

        # Mimic hamer/datasets/utils.py to get bbox_processed
        rescale_factor = 2.0
        center = (box[2:4] + box[0:2]) / 2.0
        scale = rescale_factor * (box[2:4] - box[0:2]) / 200.0
        center_x = center[0]
        center_y = center[1]
        BBOX_SHAPE = [192, 256] 
        bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=BBOX_SHAPE).max()

        # Making global translation
        s, tx, ty = params['pred_cam']
        x0 = center_x - bbox_size/2
        y0 = center_y - bbox_size/2
        tx = tx + 1/s + (2*x0-out_w)/(s*bbox_size+1e-9)  # putting 1/s and xmin is equivalent to putting center_x
        ty = ty + 1/s + (2*y0-out_h)/(s*bbox_size+1e-9)
        tz = 2*epic_focal/(s*bbox_size+1e-9)
        global_transl = torch.Tensor([tx, ty, tz])
        # global_transl *= mag

        verts = vh + global_transl
        faces = mano.th_faces
        mesh = Trimesh(
            vertices=verts.squeeze().cpu().numpy(), 
            faces=faces.squeeze().cpu().numpy())
        # mesh = SimpleMesh(vh + global_transl, mano.th_faces)
        mag = np.power(
            (target_volume * 1e-6) / mesh.volume, 1/3)
        mesh.vertices = mesh.vertices * mag
        return mesh

    def visualise_frame(self, 
                        vid: str, frame: int, 
                        hand_side: str,
                        out_h=256, out_w=456,
                        epic_focal=5000, 
                        mag=1.0) -> np.ndarray:
        """
        Assumes each *.pt contains:
        dict(left=params, right=params)
        params = dict(
            pred_cam
                Tensor torch.Size([3])
            global_orient
                Tensor torch.Size([1, 3, 3])
            hand_pose
                Tensor torch.Size([15, 3, 3])
            betas
                Tensor torch.Size([10])
        )

        Args:
            hand_side: 'left' or 'right'
            mag: magnifying the vertices, hence also magnifying the translation_z
        """
        from libzhifan.geometry import projection, CameraManager
        mesh = self.load_mhand(vid, frame, hand_side, epic_focal, mag)
        if mesh is None:
            return None

        # Forge global camera
        global_cam = CameraManager(
            fx=epic_focal, fy=epic_focal, cx=out_w//2, cy=out_h//2, 
            img_h=out_h, img_w=out_w)

        img_pil = self.reader.read_image_pil(vid, frame).resize([out_w, out_h])
        rend = projection.perspective_projection_by_camera(
            mesh, global_cam, 
            method=dict(name='pytorch3d', coor_sys='nr', in_ndc=False), 
            image=np.asarray(img_pil))
        return rend
