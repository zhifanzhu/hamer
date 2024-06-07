from argparse import ArgumentParser
from PIL import Image
from PIL import ImageDraw, ImageFont
import os.path as osp
import numpy as np
import tqdm
from moviepy import editor
import tempfile

from libzhifan import odlib
from aria.fork_slam_reader import ForkRawSLAMReader

odlib.setup('xyxy')


""" To run this, use a pytorch3d-compatible environment.
Hamer's environment doesn't work with pytorch3d.
"""


def parse_args():
    parser = ArgumentParser(description='HaMeR + ho_detector + VRS reading')
    parser.add_argument('--vid', type=str, required=True,
                        help='E.g. P01-20240202-110250')

    parser.add_argument('--detection-dir', type=str, default='./data/aria/')
    parser.add_argument('--hamer-dir', type=str, default='./data/aria/')
    parser.add_argument('--fps', type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    dets_csv = osp.join(args.detection_dir, args.vid + '.csv')

    storage_dir = '/home/eve/Zhifan/eka_vista_kit/eka_storage/'
    reader = ForkRawSLAMReader(
        args.vid, storage_dir=storage_dir,
        load_pts=False,
        load_obs=False,
        load_frame_traj=False,
        load_gaze=False, load_hamer=True, hamer_dir=args.hamer_dir,
        bbox_file=dets_csv)


    out_w, out_h = 512, 512
    box_scale = np.float32([out_w, out_h, out_w, out_h]) / \
        np.float32([1408, 1408, 1408, 1408])
    text_loc = (40, 480) # (300, 1300) for 1408
    myFont = ImageFont.truetype('FreeMono.ttf', 24)
    tmpdir = tempfile.mkdtemp(dir='/media/eve/SCRATCH/Zhifan/unused/')
    print("Saving frames to tmpdir:", tmpdir)
    fnames = []
    # for f in tqdm.trange(250, 270):
    for f in tqdm.trange(1, reader.num_rgb_frames):
        img, ts_ns = reader.visualise_frame(
            f, focal_length=5000, ret_timestamp_ns=True,
            out_h=out_h, out_w=out_w)
        bboxes = reader.get_detections(f)[..., :4]
        if len(bboxes) > 0:
            bboxes *= box_scale
            img_pil = odlib.draw_bboxes_image_array(img*255., bboxes)
        else:
            img_pil = Image.fromarray((img * 255.).astype(np.uint8)).convert('RGB')
        fname = osp.join(tmpdir, f'{f:010d}.png')

        text = "Frame %d at time %.3fs" % (f, ts_ns/1e9)
        I1 = ImageDraw.Draw(img_pil)
        I1.text(text_loc, text, font=myFont, fill =(255, 255, 255))
        fnames.append(fname)
        img_pil.save(fname)

    clip = editor.ImageSequenceClip(fnames, fps=args.fps)
    clip.write_videofile(f'outputs/{args.vid}.mp4')


if __name__ == '__main__':
    main()
