""" Mostly copied from eka_vista_kit """
import os
import torch
from PIL import Image
from typing import List
from pathlib import Path
import json
import numpy as np
import gzip
import tqdm
import pandas as pd
import open3d as o3d
import projectaria_tools.core.mps as mps
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.mps.utils import get_nearest_pose, get_nearest_eye_gaze
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.mps.utils import EyeGaze, get_eyegaze_point_at_depth, get_gaze_vector_reprojection
from _core_pybinds.mps import (
    ClosedLoopTrajectoryPose,
    EyeGaze,
    get_eyegaze_point_at_depth,
    GlobalPointPosition,
    hand_tracking,
)

from manopth.manolayer import ManoLayer
from lift3d.rot_cvt import matrix_to_axis_angle

# STORAGE_DIR = './eka_storage'

DEVICE_SLAM_L = '1201-1'
DEVICE_SLAM_R = '1201-2'
DEVICE_RGB = '214-1'


UNDEFINED_ROW = 'UNDEFINED_ROW'
    

def make_frame_trajectory(vrs_file: str, close_loop_trajectory: str,
                          show_tqdm=False) -> pd.DataFrame:
    """ This extract the rows corresponding(nearest) to rgb frames. 
    As input, this stores the **slam_left** camera pose.
    In the output, this will have `num_rgb_frames` rows.
    The frames without corresponding trajectory will have string NONE set in every `graph_uid` field.
    """
    # For faithfulness, we will use the same get_nearest_pose function. Hence this preparation
    traj = mps.read_closed_loop_trajectory(close_loop_trajectory)
    traj_df = pd.read_csv(close_loop_trajectory)
    timed_rows = []
    for i, row in traj_df.iterrows():
        row.tracking_timestamp = traj[i].tracking_timestamp
        timed_rows.append(row)

    vrs_provider = data_provider.create_vrs_data_provider(str(vrs_file))
    rgb_stream_id = StreamId(DEVICE_RGB)

    nan_row = traj_df.iloc[0].copy()
    nan_row.graph_uid = UNDEFINED_ROW  # flag for non-existence
    nan_row[['qx_world_device', 'qy_world_device', 'qz_world_device', 'qw_world_device']] = [
        np.nan, np.nan, np.nan, np.nan]  # Need some obvious bugs so that user won't miss it
    num_frames = vrs_provider.get_num_data(rgb_stream_id)
    rows = []
    ranger = tqdm.tqdm(range(num_frames), total=num_frames) if show_tqdm else range(num_frames)
    for f in ranger:
        _, img_record = vrs_provider.get_image_data_by_index(rgb_stream_id, f)
        ts_ns = img_record.capture_timestamp_ns
        row = get_nearest_pose(timed_rows, ts_ns)
        if row is None:
            rows.append(nan_row)
        else:
            rows.append(row)
    frame_traj_df = pd.DataFrame(rows)
    return frame_traj_df

def make_eye_gaze(vrs_file: str, general_eye_gaze: str,
                          show_tqdm=False) -> pd.DataFrame:
    """ This extracts the closest gaze reading for each frame. 
    As input, this stores the **camera-et** eye tracks.
    In the output, this will have `num_rgb_frames` rows.
    The frames without corresponding trajectory will have string NONE set in every `graph_uid` field.
    """
    # For faithfulness, we will use the same get_nearest_pose function. Hence this preparation
    gaze = mps.read_eyegaze(general_eye_gaze)
    gaze_df = pd.read_csv(general_eye_gaze)
    max_gaze_ts = gaze_df['tracking_timestamp_us'].max()
    min_gaze_ts = gaze_df['tracking_timestamp_us'].min()
    timed_rows = []
    for i, row in gaze_df.iterrows():
        row.tracking_timestamp = gaze[i].tracking_timestamp
        timed_rows.append(row)

    vrs_provider = data_provider.create_vrs_data_provider(str(vrs_file))
    rgb_stream_id = StreamId(DEVICE_RGB)

    nan_row = gaze_df.iloc[0].copy()
    for column in gaze_df.columns:
        if column != 'tracking_timestamp_us' and column != 'session_uid':
            nan_row[[column]] = np.nan
    num_frames = vrs_provider.get_num_data(rgb_stream_id)
    rows = []
    ranger = tqdm.tqdm(range(num_frames), total=num_frames) if show_tqdm else range(num_frames)
    for f in ranger:
        _, img_record = vrs_provider.get_image_data_by_index(rgb_stream_id, f)
        ts_ns = img_record.capture_timestamp_ns
        row = get_nearest_eye_gaze(timed_rows, ts_ns)
        if row is None: 
            if ts_ns >= (max_gaze_ts * 1e3):
                nan_row['tracking_timestamp_us'] = max_gaze_ts
            else:
                nan_row['tracking_timestamp_us'] = min_gaze_ts
            rows.append(nan_row.copy())
        else:
            rows.append(row)
    frame_gaze_df = pd.DataFrame(rows)
    return frame_gaze_df


class ForkRawSLAMReader:
    """ A SLAM + VRS reader 

    The first time instantiating this class, 
    will create directory 'intermediate_data/frame_trajectory/'
    and create a file 'intermediate_data/frame_trajectory/<vid>.csv'
    The benefit of a new frame_trajectory file are:
    1. Most of the time, we only need the trajectory corresponding to rgb frames, not full trajectory of IMU.
    2. Reduce the size of trajectory, hence faster to load. 
    3. O(1) hashing is always faster than O(lgn) bin search.
    The disadvantage is, of course, losing the denser trajectory and initial creation cost.
    
    """

    def __init__(self, 
                 vid: str, 
                 storage_dir='./eka_storage',
                 slam_id=None,
                 load_pts=True,
                 load_frame_traj=True,
                 load_gaze=True,
                 load_vrs=True,
                 load_hamer=False,
                 pts_path=None,
                 traj_path=None,
                 gaze_path=None,
                 frame_traj_path=None,
                 vrs_path=None,
                 load_raw_traj=False,
                 write_T_rgb_to_slamL_cache=False,
                 load_obs=False,
                 obs_path=None,
                 hamer_dir=None,
                 bbox_file=None,
                 ):
        """
        Args:
            vid: str. E.g. 'P01-20240202-161948'
            traj_path/pts_path/vrs_path: str. If provided, use these instead of default
        """
        pid = vid.split('-')[0]
        multi_dir = Path(storage_dir)/f'eye_gaze_n_slam/{pid}/multi'
        self.vid = vid

        if slam_id is None:
            # Get slam id
            with open(multi_dir/"vrs_to_multi_slam.json", 'r') as fp:
                vrs_to_slam = json.load(fp)
                slam_id = vrs_to_slam[f'{vid}.vrs']
        self.slam_dir = multi_dir/f'{slam_id}/slam/'

        self.semi_pts_path = self.slam_dir/'semidense_points.csv.gz' \
            if pts_path is None else pts_path 
        if load_pts:
            with open(self.semi_pts_path, 'rb') as fp:
                gzip_fp = gzip.GzipFile(fileobj=fp)
                self.semi_pts = pd.read_csv(gzip_fp)  # csv
        
        default_vrs_path = Path(storage_dir)/f'raw_data/{pid}/{vid}.vrs'
        self.vrs_path = default_vrs_path \
            if vrs_path is None else vrs_path
        if not self.vrs_path.exists() and load_vrs:
            print(f"VRS {self.vrs_path} file not found.")
        if load_vrs:
            vrs_provider = data_provider.create_vrs_data_provider(str(self.vrs_path))
            rgb_stream_id = StreamId(DEVICE_RGB)
            rgb_stream_label = vrs_provider.get_label_from_stream_id(rgb_stream_id)
            self.device_calibration = vrs_provider.get_device_calibration()
            self.num_rgb_frames = vrs_provider.get_num_data(rgb_stream_id)
            self.provider = vrs_provider
            self.rgb_stream_id = rgb_stream_id

            rgb_calib = self.device_calibration.get_camera_calib(rgb_stream_label)
            self.rgb_camera_calibration = rgb_calib
            PINHOLE_RARIO = 3.0139  # Sugguested by Ahmad
            img_w, img_h = rgb_calib.get_image_size()
            self.pinhole_params = calibration.get_linear_camera_calibration(
                img_w, img_h, focal_length=img_w / PINHOLE_RARIO,
                T_Device_Camera=rgb_calib.get_transform_device_camera())
            self.pinhole_cw90 = calibration.rotate_camera_calib_cw90deg(self.pinhole_params)

            self.T_rgb_to_slamL = \
                rgb_calib.get_transform_device_camera().to_matrix() # Transform from rgb to slam_l 
            self.T_pinholecw90_to_slamL = \
                self.pinhole_cw90.get_transform_device_camera().to_matrix() # Transform from rgb to slam_l
            self.T_cpf_to_slamL = \
                self.device_calibration.get_transform_device_cpf().to_matrix()

            if write_T_rgb_to_slamL_cache:
                rgb_to_slamL_path = Path('./intermediate_data/rgb_to_slamL')/f'{self.vid}.json'
                rgb_to_slamL_path.parent.mkdir(parents=True, exist_ok=True)
                print("Writing T_rgb_to_slamL to", rgb_to_slamL_path)
                with open(rgb_to_slamL_path, 'w') as fp:
                    json.dump(self.T_rgb_to_slamL.tolist(), fp)

                cpf_to_slamL_path = Path('./intermediate_data/cpf_to_slamL')/f'{self.vid}.json'
                cpf_to_slamL_path.parent.mkdir(parents=True, exist_ok=True)
                print("Writing cpf_to_slamL to", cpf_to_slamL_path)
                with open(cpf_to_slamL_path, 'w') as fp:
                    json.dump(self.T_cpf_to_slamL.tolist(), fp)

        if load_frame_traj:
            if frame_traj_path is not None:
                self.frame_traj_path = frame_traj_path
                self.frame_traj = mps.read_closed_loop_trajectory(frame_traj_path)
            else:
                self.frame_traj_path = Path('./intermediate_data/frame_trajectory')/f'{self.vid}.csv'
                self.frame_traj_path.parent.mkdir(parents=True, exist_ok=True)
                traj_path = self.slam_dir/'closed_loop_trajectory.csv' \
                    if traj_path is None else traj_path
                self.frame_traj = self._load_frame_trajectory(self.vrs_path, traj_path)
            # Make None pose explicit
            self.frame_traj = [
                pose_info if pose_info.graph_uid != UNDEFINED_ROW else None
                for pose_info in self.frame_traj
            ]

        if load_raw_traj:
            self.raw_traj = mps.read_closed_loop_trajectory(
                  str(self.traj_path))
            self.raw_traj_csv = pd.read_csv(self.traj_path)

        if load_obs:
            assert os.path.isfile(obs_path), f"Observation file {obs_path} not found."
            self.obs = pd.read_csv(obs_path)

        self.load_obs = load_obs

        if load_gaze:
            if gaze_path is not None:
                self.gaze_path = gaze_path
                self.eye_gaze = mps.read_eyegaze(gaze_path)
            else:
                self.gaze_path = Path('./intermediate_data/eye_gaze')/f'{self.vid}.csv'
                self.gaze_path.parent.mkdir(parents=True, exist_ok=True)
                gaze_path = Path(storage_dir)/f'updated_eye_gaze/{pid}/mps_{vid}_vrs/eye_gaze/general_eye_gaze.csv' \
                    if gaze_path is None else gaze_path
                self.eye_gaze = self._load_eye_gaze(self.vrs_path, gaze_path)

        self.load_pts = load_pts
        self.load_frame_traj = load_frame_traj
        self.load_vrs = load_vrs
        self.load_raw_traj = load_raw_traj
        self.load_gaze = load_gaze
        self.load_hamer = load_hamer
        self.hamer_dir = hamer_dir
        if self.load_hamer:
            mano_root = './third-party/mano'
            self.left_mano = ManoLayer(
                flat_hand_mean=True, ncomps=45, side='left',
                mano_root=mano_root, use_pca=False)
            self.right_mano = ManoLayer(
                flat_hand_mean=True, ncomps=45, side='right',
                mano_root=mano_root, use_pca=False)
            self.dets_df = pd.read_csv(bbox_file)
    
    def _load_frame_trajectory(self, vrs_path, traj_path):
        frame_traj_path = self.frame_traj_path
        if not frame_traj_path.exists():
            print("Creating frame trajectory, writing to", self.frame_traj_path)
            frame_traj_df = make_frame_trajectory(str(vrs_path), str(traj_path), show_tqdm=True)
            frame_traj_df.to_csv(frame_traj_path, index=False)
        return mps.read_closed_loop_trajectory(str(frame_traj_path))
    
    def _load_eye_gaze(self, vrs_path, gaze_path):
        save_gaze_path = self.gaze_path
        if not save_gaze_path.exists():
            print("Creating eye gaze, writing to", self.gaze_path)
            frame_traj_df = make_eye_gaze(str(vrs_path), str(gaze_path), show_tqdm=True)
            frame_traj_df.to_csv(save_gaze_path, index=False)
        return mps.read_eyegaze(str(save_gaze_path))

    def load_all_points(self, as_open3d=False) -> np.ndarray:
        """ returns: (N, 3) """
        assert self.load_pts, "Points not loaded"
        pts = np.asarray(self.semi_pts[['px_world', 'py_world', 'pz_world']])
        if as_open3d:
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
            return pcd
        return pts

    def load_filtered_points(self, 
                             threshold_invdep: float = 0.001,
                             threshold_dep: float = 0.05,
                             as_open3d=False) -> np.ndarray:
        """ returns: (N, 3) 
        The default thresholds are copied from projectaria_tools. 

        Args:
            threshold_invdep: float. Larger threshold means more noise included.
            threshold_dep: float. 
        """
        assert self.load_pts, "Points not loaded"
        INV_DIST_STD = 'inv_dist_std'
        DIST_STD = 'dist_std'
        sel_pts = self.semi_pts[
            (self.semi_pts[INV_DIST_STD] < threshold_invdep) & (self.semi_pts[DIST_STD] < threshold_dep)]
        pts = np.asarray(sel_pts[['px_world', 'py_world', 'pz_world']])
        if as_open3d:
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
            return pcd
        return pts

    def load_observations(self) -> pd.DataFrame:
        assert self.load_obs, "Observations not loaded"
        return self.obs

    def load_observations_at_timestamp(
            self,
            timestamp_ns: int,
            threshold: bool = True,
            threshold_invdep: float = 0.001,
            threshold_dep: float = 0.05,
            as_open3d=False
            ) -> np.ndarray:
        """ Get observed points at a particular timestamp.

        Parameters:
        - timestamp_ns (int): The timestamp in nanoseconds for which to retrieve the observed points.
        - threshold (bool, optional): Whether to apply thresholding on the observed points. Defaults to True.
        - threshold_invdep (float, optional): The inverse depth threshold for thresholding. Defaults to 0.001.
        - threshold_dep (float, optional): The depth threshold for thresholding. Defaults to 0.05.
        - as_open3d (bool, optional): Whether to return the observed points as an Open3D PointCloud object. 
                                        Defaults to False.

        Returns:
        - np.ndarray or o3d.geometry.PointCloud: The observed points at the specified timestamp. 
                                                    If as_open3d is True, returns an Open3D PointCloud object.
                                                    Otherwise, returns a numpy array of shape (N, 3) where N is 
                                                    the number of observed points.

        Raises:
        - AssertionError: If observed points are not loaded.
        - AssertionError: If the number of unique points in observations and semi_pts are not the same.

        Note:
        - The observed points should be loaded before calling this function.
        - The observed points and semi-dense points should be loaded from the same recording session.
        - A usecase of this function is in tools/visualise_dynamic_pt.py
        """
        assert self.load_obs, "Observed points not loaded"
        assert self.obs['uid'].nunique() == self.semi_pts['uid'].nunique(), \
            '''
            Number of unique points in observations and semi_pts should be same. Maybe you're loading 
            observation points and semi-dense points from different recording sessions.
            '''
        selected_uids = self.obs[self.obs['frame_tracking_timestamp_us'] == timestamp_ns]['uid'].values
        selected_points = self.semi_pts[self.semi_pts['uid'].isin(selected_uids)]
        if threshold:
            selected_points = selected_points[
                (selected_points['inv_dist_std'] < threshold_invdep) & (selected_points['dist_std'] < threshold_dep)]
        selected_points = np.asarray(selected_points[['px_world', 'py_world', 'pz_world']])
        if as_open3d:
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(selected_points))
            return pcd
        return selected_points

    def load_slaml_trajectory(self, ret_mat=True) -> List:
        """ The slam_l poses at rgb_frames, in world coor.

        Args:
            ret_mat: bool. If True, return the 4x4 matrix.
                otherwise, return the Sophus SE3 object.

        Returns: 
            poses: list of (4, 4), slamL-to-world
                if pose at a frame is not available, pose=None.
        """
        assert self.load_frame_traj, "Frame trajectory not loaded"
        poses = [None for _ in range(self.num_rgb_frames)]
        for i, f in enumerate(range(self.num_rgb_frames)):
            pose_info = self.frame_traj[i]
            if pose_info is None:
                continue
            pose = pose_info.transform_world_device
            if ret_mat:
                pose = pose.to_matrix()
            poses[i] = pose
        return poses

    def load_rgb_trajectory(self, ret_mat=True) -> List:
        """ The rgb poses at rgb_frames

        Args:
            ret_mat: bool. If True, return the 4x4 matrix.
                otherwise, return the Sophus SE3 object.

        Returns: 
            poses: (N, 4, 4), apply-to-col. Can be None
                where N is number of rgb frames
        """
        slaml_poses = self.load_slaml_trajectory(ret_mat=ret_mat)
        T_rgb_to_slamL = self.T_rgb_to_slamL
        poses = []
        for pose in slaml_poses:
            if pose is None:
                poses.append(None)
            else:
                poses.append(pose @ T_rgb_to_slamL)

        return poses
    
    def load_gaze_vectors(self) -> List:
        """ The eye gaze vectors at rgb_frames, in slamL coor.
            (Open3D will transform to world space later)

        Returns: 
            gazes: list of (4,), cpf-to-slamL
                if gaze at a frame is not available, gaze=(0, 0, 0, 0).
        """
        assert self.load_gaze, "Eye Gaze not loaded"
        gazes = [np.zeros((4,)) for _ in range(self.num_rgb_frames)] # Dummy gaze points
        T_cpf_to_slamL = self.T_cpf_to_slamL
        for i, f in enumerate(range(self.num_rgb_frames)):
            gaze_cpf = self.get_gaze_vector(self.eye_gaze[i]) # [X, Y, Z, 1.0] (CPF coords)
            if np.isnan(gaze_cpf[0]):
                continue
            gazes[i] = T_cpf_to_slamL @ gaze_cpf # [X, Y, Z, 1.0] (SlamL coords)

        return gazes
    
    def read_rgb_frame(self, frame: int, 
                       ret_timestamp_ns=False,
                       ret_record=False,
                       undistort=False) -> np.ndarray:
        """ 
        Warning: the img_record also have a field 'frame_number', but it can be diffrent.
        E.g. 
            `self.read_rgb_frame(frame=0, ret_record=True).frame_number` might print 5, instead of 0.

        Returns:
            img: (H, W, 3)
            [Optional] timestamp_ns: int. 
                The corresponding timestamp of the frame in nanosecond.
        """
        assert self.load_vrs, "VRS not loaded"
        img_data, img_record = self.provider.get_image_data_by_index(self.rgb_stream_id, frame)
        img_arr = img_data.to_numpy_array()

        if self.load_gaze:
            gaze_at_frame = self.eye_gaze[frame]
            if not np.isnan(gaze_at_frame.yaw):
                gc = get_gaze_vector_reprojection(
                    eye_gaze=gaze_at_frame,
                    stream_id_label=self.provider.get_label_from_stream_id(self.rgb_stream_id),
                    device_calibration=self.device_calibration,
                    camera_calibration=self.rgb_camera_calibration,
                    depth_m=gaze_at_frame.depth
                )
                gc = list(map(int, gc))

                # Draw cross at gaze point before rotation
                cross_colour = np.array([0, 255, 255])
                img_arr[max(0, gc[1]-25):min(gc[1]+25, img_arr.shape[0]), max(0, gc[0]-5):min(gc[0]+5, img_arr.shape[1])] = cross_colour
                img_arr[max(0, gc[1]-5):min(gc[1]+5, img_arr.shape[0]), max(0, gc[0]-25):min(gc[0]+25, img_arr.shape[1])] = cross_colour

        if undistort:
            # See https://facebookresearch.github.io/projectaria_tools/docs/data_utilities/advanced_code_snippets/image_utilities#rotated-image-clockwise-90-degrees
            img_arr = calibration.distort_by_calibration(
                img_arr, self.pinhole_params, self.rgb_camera_calibration)
        img_arr = np.rot90(img_arr, k=-1)
        if ret_timestamp_ns:
            ts_ns = img_record.capture_timestamp_ns
            return img_arr, ts_ns
        if ret_record:
            return img_arr, img_record
    
        return img_arr
    
    def get_rgb_timestamp(self, frame: int) -> int:
        """ Returns: timestamp_ns: int """
        assert self.load_vrs, "VRS not loaded"
        _, img_record = self.provider.get_image_data_by_index(self.rgb_stream_id, frame)
        return img_record.capture_timestamp_ns
    
    def get_gaze_vector(self, eye_gaze: EyeGaze) -> np.ndarray:
        """
        Returns 4D Gaze point in CPF frame
        """
        gaze_center_in_cpf = get_eyegaze_point_at_depth(
            eye_gaze.yaw, eye_gaze.pitch, eye_gaze.depth
        )
        return np.concatenate([gaze_center_in_cpf, [1.0]], axis=0)

    """ Detections: 3D Hand from HAMER, Bounding box """
    def get_detections(self, frame:int) -> np.ndarray:
        """ returns: [N, 5] of (x0, y0, x1, y1, right)"""
        ents = self.dets_df[self.dets_df.frame == frame]
        dets = []
        for i, (frame, x0, y0, x1, y1, lr) in ents.iterrows():
            det = np.asarray([x0, y0, x1, y1, lr])
            dets.append(det)
        return np.asarray(dets)

    def load_mhand(self, 
                   frame: int, 
                   hand_loc: np.ndarray,
                   hand_side: str,
                   focal_length=5000, 
                   out_h=1408,
                   out_w=1408,
                   verbose=False):
        """
        Args:
            target_volume: Mean hand volume in cc.
            hand_loc: the GT/predicted hand loc,
                NB. since we alter the depth of the hand, 
                we will also alter the hand volume
        """
        vid = self.vid
        hamer_params_path = f'{self.hamer_dir}/{vid}/frame_{frame:010d}.pt'
        has_frame = os.path.exists(hamer_params_path)
        if not has_frame:
            if verbose:
                print(f"Frame {frame} not found in {hamer_params_path}")
            return None
        all_params = torch.load(hamer_params_path)
        if hand_side not in all_params:
            if verbose:
                print(f"Hand side {hand_side} not found in {hamer_params_path}")
            return None
        params = all_params[hand_side]
        # lbox, rbox = self.hosgetter.get_frame_hbox(vid, frame)
        """ Get boxes """
        lbox, rbox = None, None
        entries = self.dets_df[self.dets_df['frame'] == frame]
        box_scale = np.float32([out_w, out_h, out_w, out_h]) / \
            np.float32([1408, 1408, 1408, 1408])
        for i, (frame, x0, y0, x1, y1, lr) in entries.iterrows():
            right = int(lr)
            if right:
                rbox = np.asarray([x0, y0, x1, y1])
                rbox *= box_scale
            else:
                lbox = np.asarray([x0, y0, x1, y1])
                lbox *= box_scale

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

        vh, jh = mano.forward(thetas, betas)
        vh /= 1000.
        jh /= 1000.

        # Mimic hamer/datasets/utils.py to get bbox_processed
        rescale_factor = 2.5
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
        tz = 2*focal_length/(s*bbox_size+1e-9)
        global_transl = torch.Tensor([tx, ty, tz])

        mag = hand_loc / global_transl
        verts = (vh + global_transl) * mag
        faces = mano.th_faces

        # mesh = Trimesh(
        #     vertices=verts.squeeze().cpu().numpy(), 
        #     faces=faces.squeeze().cpu().numpy())
        from libzhifan.geometry import SimpleMesh
        mesh = SimpleMesh(verts, faces)
        return mesh

    def visualise_frame(self, 
                        frame: int, 
                        out_h=1408, out_w=1408,
                        focal_length=5000,
                        ret_timestamp_ns=False) -> np.ndarray:
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
        img_arr, ts_ns = self.read_rgb_frame(
            frame, ret_timestamp_ns=True, undistort=True)
        img_pil = Image.fromarray(img_arr).resize([out_w, out_h])
        # Forge global camera
        global_cam = CameraManager(
            fx=focal_length, fy=focal_length, cx=out_w//2, cy=out_h//2, 
            img_h=out_h, img_w=out_w)

        meshes = []
        for hand_side in ('left', 'right'):
            mesh = self.load_mhand(
                frame, hand_side, focal_length,
                out_h=out_h, out_w=out_w)
            if mesh is None:
                continue
            meshes.append(mesh)

        if len(meshes) == 0:
            rend = np.asarray(img_pil) / 255.
        else:
            rend = projection.perspective_projection_by_camera(
                meshes, global_cam, 
                method=dict(name='pytorch3d', coor_sys='nr', in_ndc=False), 
                image=np.asarray(img_pil))
        if ret_timestamp_ns:
            return rend, ts_ns
        return rend


""" Utils """
def expand_to_aspect_ratio(input_shape, target_aspect_ratio=None):
    """Increase the size of the bounding box to match the target shape."""
    if target_aspect_ratio is None:
        return input_shape

    try:
        w , h = input_shape
    except (ValueError, TypeError):
        return input_shape

    w_t, h_t = target_aspect_ratio
    if h / w < h_t / w_t:
        h_new = max(w * h_t / w_t, h)
        w_new = w
    else:
        h_new = h
        w_new = max(h * w_t / h_t, w)
    if h_new < h or w_new < w:
        breakpoint()
    return np.array([w_new, h_new])