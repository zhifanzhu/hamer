import open3d as o3d
import numpy as np
from argparse import ArgumentParser
import json


""" Visualize poses and point-cloud stored in json file. """

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--json-data', help='path to json data', required=True)
    parser.add_argument('--pcd-path', help='path to alternative point cloud data', default='./geom_data/P28_101.ply')
    parser.add_argument('--show-mesh-frame', default=True)
    parser.add_argument('--scene-mag', type=float, default=0.4,
        help='scale the scene by this factor. 0.4 for P28_101')
    parser.add_argument('--specify-frame-name', default='frame_0000002370.jpg')
    parser.add_argument('--num-display-poses', type=int, default=500,
        help='randomly display num-display-poses to avoid creating too many poses')
    parser.add_argument('--frustum-size', type=float, default=0.2)
    parser.add_argument('--view-path', help='path to view data', default=None)
    return parser.parse_args()


""" Source: See COLMAP """
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def get_c2w(img_data: list) -> np.ndarray:
    """
    Args:
        img_data: list, [qvec, tvec] of w2c

    Returns:
        c2w: np.ndarray, 4x4 camera-to-world matrix
    """
    w2c = np.eye(4)
    w2c[:3, :3] = qvec2rotmat(img_data[:4])
    w2c[:3, -1] = img_data[4:7]
    c2w = np.linalg.inv(w2c)
    return c2w


def get_frustum(c2w: np.ndarray,
                sz=0.2,
                camera_height=None,
                camera_width=None,
                frustum_color=[1, 0, 0]) -> o3d.geometry.LineSet:
    """
    Args:
        c2w: np.ndarray, 4x4 camera-to-world matrix
        sz: float, size (width) of the frustum
    Returns:
        frustum: o3d.geometry.TriangleMesh
    """
    cen = [0, 0, 0]
    wid = sz
    if camera_height is not None and camera_width is not None:
        hei = wid * camera_height / camera_width
    else:
        hei = wid
    tl = [wid, hei, sz]
    tr = [-wid, hei, sz]
    br = [-wid, -hei, sz]
    bl = [wid, -hei, sz]
    points = np.float32([cen, tl, tr, br, bl])
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1],]
    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(points)
    frustum.lines = o3d.utility.Vector2iVector(lines)
    frustum.colors = o3d.utility.Vector3dVector([np.asarray([1, 0, 0])])
    frustum.paint_uniform_color(frustum_color)

    frustum = frustum.transform(c2w)
    return frustum


if __name__ == "__main__":
    args = parse_args()
    frustum_size = args.frustum_size

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0, origin=[0, 0, 0])

    with open(args.json_data, 'r') as f:
        model = json.load(f)

    # scene_mag = 0.4
    scene_mag = args.scene_mag

    if args.pcd_path is not None:
        pcd_path = args.pcd_path
        assert pcd_path.endswith('.ply')
        pcd = o3d.io.read_point_cloud(pcd_path)
    else:
        points = model['points']
        pcd_np = [v[:3] for v in points]
        pcd_rgb = [np.asarray(v[3:6]) / 255 for v in points]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_np)
        pcd.colors = o3d.utility.Vector3dVector(pcd_rgb)
    pcd.scale(scene_mag, center=np.zeros(3))
    vis.add_geometry(pcd, reset_bounding_box=True)
    if args.show_mesh_frame:
        vis.add_geometry(mesh_frame, reset_bounding_box=True)

    camera = model['camera']
    cam_h, cam_w = camera['height'], camera['width']
    c2w_list = [get_c2w(img) for img in model['images'].values()]
    c2w_sel_inds = np.random.choice(
        len(c2w_list), min(len(c2w_list), args.num_display_poses), replace=False)
    c2w_sel = [c2w_list[i] for i in c2w_sel_inds]
    if args.specify_frame_name is None:
        frustums = [
            get_frustum(c2w, sz=frustum_size, camera_height=cam_h, camera_width=cam_w)
            for c2w in c2w_sel
        ]
        for frustum in frustums:
            vis.add_geometry(frustum, reset_bounding_box=True)
    else:
        frame_name = args.specify_frame_name
        c2w = get_c2w(model['images'][frame_name])
        c2w[:3, -1] *= scene_mag
        frustum = get_frustum(c2w, sz=frustum_size, camera_height=cam_h, camera_width=cam_w)
        vis.add_geometry(frustum, reset_bounding_box=True)

        # import pickle
        # mhand = pickle.load(open('mhand.pkl', 'rb'))
        # mhand = mhand.as_open3d.transform(c2w)
        # mhand.compute_vertex_normals()
        # mhand.compute_triangle_normals()
        # vis.add_geometry(mhand, reset_bounding_box=True)

    control = vis.get_view_control()
    if args.view_path is not None:
        with open(args.view_path, 'r') as f:
            status = json.load(f)
        cam_info = status['trajectory'][0]
        fov = cam_info['field_of_view']
        lookat = np.asarray(cam_info['lookat'])
        front = np.asarray(cam_info['front'])
        front = front / np.linalg.norm(front)
        up = np.asarray(cam_info['up'])
        zoom = cam_info['zoom']
        control.set_front(front)
        control.set_lookat(lookat)
        control.set_up(up)
        control.set_zoom(zoom)
    else:
        control.set_front([1, 1, 1])
        control.set_lookat([0, 0, 0])
        control.set_up([0, 0, 1])
        control.set_zoom(1)

    vis.run()
    vis.destroy_window()
