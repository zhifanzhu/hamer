import tqdm
from ekavista.io.slam_reader_basic import RawSLAMReaderBasic
from ekavista.masks.functions import load_box_from_mask
import pandas as pd

vid = 'P01-20240202-161948'
reader = RawSLAMReaderBasic(
    vid, frame_type='mp4', 
    storage_dir='/media/eve/DATA/Zhifan/eka_storage', 
    load_pts=False, load_frame_traj=False)
out_path = f'data/{vid}_undistorted_dets.csv'

columns = columns={'0': 'frame', '1': 'x0', '2': 'y0', '3': 'x1', '4': 'y1', '5': 'right'}


rows = []
for f in tqdm.trange(0, reader.num_rgb_frames):
    lbox, rbox, _ = load_box_from_mask(
        vid, f, reader, undistort=True, ret_mask=True)
    
    if lbox is not None:
        row = [f] + lbox.tolist() + [0]
        rows.append(row)
    if rbox is not None:
        row = [f] + rbox.tolist() + [1]
        rows.append(row)
    
    if len(rows) % 500 == 0:
        df = pd.DataFrame(rows)
        df = df.rename(columns=columns)
        df.to_csv(out_path, index=None)

df = pd.DataFrame(rows)
df = df.rename(columns=columns)
df.to_csv(out_path, index=None)