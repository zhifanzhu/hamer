from typing import Union
from argparse import ArgumentParser
import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import cv2
import open3d as o3d
from open3d.visualization import rendering
from lift3d.hamer_loader import HamerLoader

from lift3d.colmap_type import ColmapModel, JsonColmapModel
from lift3d.hovering.helper import (
    Helper,
    get_cam_pos,
    get_trajectory, get_pretty_trajectory, set_offscreen_as_gui
)
from lift3d.visualise_open3d import get_c2w, get_frustum

from moviepy import editor
from PIL import ImageDraw, ImageFont


""" Usage
python tools/hovering/hover_open3d.py \
    --model outputs/demo/ \
    --pcd-path outputs/demo/fused.ply \
    --view-path outputs/demo/viewstatus.json \
    --out_dir outputs/demo/hovering/
"""


def read_image(vid: str, frame: int) -> np.ndarray:
    img_path = f'/home/skynet/Zhifan/data/epic/rgb_root/{vid[:3]}/{vid}/frame_{frame:010d}.jpg'
    return np.asarray(Image.open(img_path))


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model', help="path to direcctory containing images.bin", required=True)
    parser.add_argument('--pcd-path', help="path to fused.ply", default=None)
    parser.add_argument('--view-path', type=str, required=True,
                        help='path to the view file, copy-paste from open3d gui.')
    parser.add_argument('--scene-mag', type=float, required=True)

    parser.add_argument('--vid', type=str, required=True)
    parser.add_argument('--start-frame', type=int, required=True)
    parser.add_argument('--end-frame', type=int, required=True)
    parser.add_argument('--out-video', type=str, required=True) # default='outputs/hovering/')

    parser.add_argument('--frustum-size', type=float, default=0.1)
    parser.add_argument('--frustum-line-width', type=float, default=1)
    parser.add_argument('--trajectory-line-radius', type=float, default=0.01)
    args = parser.parse_args()
    return args


class HoverRunner:

    fov = None
    lookat = None
    front = None
    up = None

    background_color = [1, 1, 1, 1.0]

    def __init__(self, out_size: str = 'big'):
        if out_size == 'big':
            out_size = (1920, 1080)
        else:
            out_size = (640, 480)
        self.render = rendering.OffscreenRenderer(*out_size)

    def setup(self,
              model: Union[ColmapModel, JsonColmapModel],
              pcd_path: str,
              viewstatus_path: str,
              scene_mag: float,
              img_x0: int = 0,
              img_y0: int = 0,
              frustum_size: float = 0.2,
              frustum_line_width: float = 5,
              trajectory_line_radius: float = 0.01):
        """
        Args:
            model:
            viewstatus_path:
                path to viewstatus.json, CTRL-c output from Open3D gui
        """

        self.model = model
        self.scene_mag = scene_mag
        if pcd_path is not None:
            pcd = o3d.io.read_point_cloud(pcd_path)
        else:
            pcd_np = np.asarray([v.xyz for v in model.points.values()])
            pcd_rgb = np.asarray([v.rgb / 255 for v in model.points.values()])
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_np)
            pcd.colors = o3d.utility.Vector3dVector(pcd_rgb)
        # self.transformed_pcd = pcd
        self.transformed_pcd = pcd.scale(scene_mag, center=np.zeros(3))

        """ Camera poses queried by frame number """
        frame_to_c2w = {}
        for v in self.model.ordered_images:
            frame = int(v.name[6:16])
            c2w = get_c2w(list(v.qvec) + list(v.tvec))
            c2w[:3, -1] *= scene_mag
            frame_to_c2w[frame] = c2w
        self.frame_to_c2w = frame_to_c2w

        self.viewstatus_path = viewstatus_path

        # Render Layout params
        # img_x0/img_y0: int. The top-left corner of the display image
        self.img_x0 = img_x0
        self.img_y0 = img_y0
        self.rgb_monitor_height = 256
        self.rgb_monitor_width = 456
        self.frustum_size = frustum_size
        self.frustum_line_width = frustum_line_width
        self.traj_line_radius = trajectory_line_radius
        self.text_loc = (450, 1000)

    def setup_hand(self,
                   hamer_loader: HamerLoader,
                   vid: str,
                   focal_length: float):
        self.hamer_loader = hamer_loader
        self.vid = vid
        self.focal_length = focal_length

    def test_single_frame(self,
                          psize,
                          frame_idx:int,
                          clear_geometry: bool =True,
                          lay_rgb_img: bool =True,
                          sun_light: bool =False,
                          show_first_frustum: bool =True,
                          ):
        """
        Args:
            psize: point size,
                probing a good point size is a bit tricky but very important!
            img_index: int. I.e. Frame number
        """
        pcd = self.transformed_pcd

        if clear_geometry:
            self.render.scene.clear_geometry()

        # Get materials
        helper = Helper(point_size=psize)
        white = helper.material('white')
        red = helper.material('red', shader='unlitLine')
        red.line_width = self.frustum_line_width
        purple = helper.material('purple', shader='defaultLit')
        self.helper = helper

        # put on pcd
        self.render.scene.add_geometry('pcd', pcd, white)
        with open(self.viewstatus_path) as f:
            viewstatus = json.load(f)
        set_offscreen_as_gui(self.render, viewstatus)

        # now put frustum on canvas
        assert frame_idx is not None
        c2w = self.frame_to_c2w[frame_idx]
        frustum = get_frustum(
            c2w=c2w, sz=self.frustum_size,
            camera_height=self.rgb_monitor_height,
            camera_width=self.rgb_monitor_width)
        if show_first_frustum:
            self.render.scene.add_geometry('first_frustum', frustum, red)
        self.render.scene.set_background(self.background_color)

        """ Add hand(s) """
        for side in ('left', 'right'):
            mhand = self.hamer_loader.load_mhand(
                self.vid, frame_idx, side, epic_focal=self.focal_length)
            if mhand is None:
                continue
            mhand = mhand.as_open3d.transform(c2w)
            mhand.compute_vertex_normals()
            mhand.compute_triangle_normals()
            self.render.scene.add_geometry(side, mhand, purple)

        if sun_light:
            self.render.scene.scene.set_sun_light(
                [0.707, 0.0, -.707], [1.0, 1.0, 1.0], 75000)
            self.render.scene.scene.enable_sun_light(True)
        else:
            self.render.scene.set_lighting(
                rendering.Open3DScene.NO_SHADOWS, (0, 0, 0))
        self.render.scene.show_axes(False)

        img_buf = self.render.render_to_image()
        img = np.asarray(img_buf)
        # test_img = self.model.read_rgb_from_name(c_image.name)
        test_img = read_image(self.vid, frame_idx)
        test_img = cv2.resize(
            test_img, (self.rgb_monitor_width, self.rgb_monitor_height))
        if lay_rgb_img:
            img[:self.rgb_monitor_height,
                -self.rgb_monitor_width:] = test_img

            img_pil = Image.fromarray(img)
            I1 = ImageDraw.Draw(img_pil)
            myFont = ImageFont.truetype('FreeMono.ttf', 65)
            bbox = (
                img.shape[1] - self.rgb_monitor_width,
                0, # img.shape[0] - self.rgb_monitor_height,
                img.shape[1],
                self.rgb_monitor_height) # img.shape[0])
            # print(bbox)
            text = "Frame %d" % frame_idx
            I1.text(self.text_loc, text, font=myFont, fill =(0, 0, 0))
            I1.rectangle(bbox, outline='red', width=5)
            img = np.asarray(img_pil)
        
        for side in ('left', 'right'):
            self.render.scene.remove_geometry(side)
        return img

    def run_all(self, start_frame, end_frame, step, 
                out_video, traj_len=10):
        """
        Args:
            step: int. Render every `step` frames
            traj_len: int. Number of trajectory lines to show
        """
        render = self.render
        out_fmt = os.path.join('/tmp/hovering', 'frame_%010d.jpg')
        os.makedirs(os.path.dirname(out_video), exist_ok=True)
        os.makedirs(os.path.dirname(out_fmt), exist_ok=True)
        red_m = self.helper.material('red', shader='unlitLine')
        red_m.line_width = self.frustum_line_width
        white_m = self.helper.material('white')
        purple_m = self.helper.material('purple', shader='defaultLit')

        render.scene.remove_geometry('first_frustum')

        myFont = ImageFont.truetype('FreeMono.ttf', 65)

        jpg_names = []
        pos_history = []
        ranger = tqdm(range(start_frame, end_frame+1, step), total=(end_frame+1-start_frame)//step)
        for frame_idx in ranger:
            frame_rgb = read_image(self.vid, frame_idx)
            frame_rgb = cv2.resize(
                frame_rgb, (self.rgb_monitor_width, self.rgb_monitor_height))
            c2w = self.frame_to_c2w[frame_idx]
            frustum = get_frustum(
                c2w=c2w, sz=self.frustum_size,
                camera_height=self.rgb_monitor_height,
                camera_width=self.rgb_monitor_width)
            pos_history.append(get_cam_pos(c2w))

            """ Add hands """
            flag_text = ""
            for side in ('left', 'right'):
                mhand = self.hamer_loader.load_mhand(
                    self.vid, frame_idx, side, epic_focal=self.focal_length)
                if mhand is None:
                    flag_text += " NO "+side+"H"
                    continue
                mhand = mhand.as_open3d.transform(c2w)
                mhand.compute_vertex_normals()
                mhand.compute_triangle_normals()
                self.render.scene.add_geometry(side, mhand, purple_m)

            if len(pos_history) > 2:
                # lines = get_pretty_trajectory(
                traj = get_trajectory(
                    pos_history, num_line=traj_len,
                    line_radius=self.traj_line_radius)
                if render.scene.has_geometry('traj'):
                    render.scene.remove_geometry('traj')
                render.scene.add_geometry('traj', traj, white_m)
            render.scene.add_geometry('frustum', frustum, red_m)

            img = render.render_to_image()
            img = np.asarray(img)
            img[:self.rgb_monitor_height,
                -self.rgb_monitor_width:] = frame_rgb
            img_pil = Image.fromarray(img)

            I1 = ImageDraw.Draw(img_pil)
            text = ("Frame %d" % frame_idx) + flag_text
            I1.text(self.text_loc, text, font=myFont, fill =(0, 0, 0))
            bbox = (img.shape[1] - self.rgb_monitor_width, 
                    0,  # img.shape[0] - self.rgb_monitor_height,
                    img.shape[1], self.rgb_monitor_height)  # img.shape[0])
            I1.rectangle(bbox, outline='red', width=5)
            img_pil.save(out_fmt % frame_idx)
            jpg_names.append(out_fmt % frame_idx)

            render.scene.remove_geometry('frustum')
            for side in ('left', 'right'):
                render.scene.remove_geometry(side)

        # Gen output
        video_fps = 20
        print("Generating video...")
        # seq = sorted(glob.glob(os.path.join(self.out_dir, '*.jpg')))
        clip = editor.ImageSequenceClip(jpg_names, fps=video_fps)
        # clip.write_videofile(os.path.join(self.out_dir, 'out.mp4'))
        clip.write_videofile(out_video)


if __name__ == '__main__':
    args = parse_args()
    model = JsonColmapModel(args.model)
    loader = HamerLoader(
        # load_dir='./data/hamer_hov1/',
        load_dir='./data/epic/',
        mano_root='third-party/mano')
    runner = HoverRunner()
    runner.setup(
        model,
        pcd_path=args.pcd_path,
        viewstatus_path=args.view_path,
        scene_mag=args.scene_mag,
        frustum_size=args.frustum_size,
        frustum_line_width=args.frustum_line_width,
        trajectory_line_radius=args.trajectory_line_radius)
    runner.setup_hand(
        loader, args.vid, focal_length=240.0)

    runner.test_single_frame(0.1, frame_idx=args.start_frame)
    runner.run_all(
        args.start_frame, args.end_frame, step=1, traj_len=10,
        out_video=args.out_video)
