import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import numpy.typing as nptype
from tqdm import tqdm
import time
import dataclasses
from typing import Tuple

from kornia.filters import canny
from kornia.morphology import dilation
import skimage.morphology

import viser
import viser.transforms as tf
from kornia.geometry.depth import depth_to_3d_v2
from colmap_loader import read_extrinsics_text, read_intrinsics_text, fov2focal, readColmapCameras

def compute_edge(depth: torch.Tensor):
    magnitude, edges = canny(depth[None, None, :, :], low_threshold=0.4, high_threshold=0.5)
    magnitude = magnitude[0, 0]
    edges = edges[0, 0]
    return edges > 0

def dilation_mask(mask: torch.Tensor):
    mask = mask.float()
    # kernel = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).to(mask.device)
    mask = dilation(mask[None, None, :, :], torch.ones((3,3), device=mask.device))
    return mask[0, 0] > 0.5

def getWorld2View(Rs, ts):
    Rt = torch.cat([Rs.transpose(1, 2), ts.unsqueeze(-1)], dim=-1)
    Rt = torch.cat([Rt, torch.tensor([0, 0, 0, 1], dtype=Rt.dtype, device=Rt.device).reshape(1, 4).repeat(Rt.shape[0], 1, 1)], dim=1)
    return Rt

@dataclasses.dataclass
class Record3dFrame:
    """A single frame from a Record3D capture."""

    K: nptype.NDArray[np.float32]
    rgb: nptype.NDArray[np.uint8]
    depth: nptype.NDArray[np.float32]
    mask: nptype.NDArray[np.bool_]
    T_world_camera: nptype.NDArray[np.float32]

    def get_point_cloud(self) -> Tuple[nptype.NDArray[np.float32], nptype.NDArray[np.uint8]]:
        rgb = self.rgb
        depth = self.depth
        mask = self.mask
        assert depth.shape == rgb.shape[:2]
        assert mask.shape == rgb.shape[:2]

        K = self.K
        pts3d = depth_to_3d_v2(
            torch.from_numpy(depth),
            torch.from_numpy(K),
            normalize_points=False
        ).reshape(-1, 3)
        if mask is not None:
            pts3d = pts3d[torch.from_numpy(mask).reshape(-1)]
        points = pts3d.cpu().numpy().astype(np.float32)
        point_colors = rgb[mask]

        return points, point_colors

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sfm_dir', required=True, type=str)
    parser.add_argument('--path_id', default=0, type=int)
    parser.add_argument("--indices", type=int, default=None, nargs='+', help='only load static parts of these frames for visualization')
    parser.add_argument("--sample_num", type=int, default=None, help='only sample several frames for static background visualization')
    parser.add_argument("--downsample_ratio", type=int, default=1, help='downsample ratio')
    parser.add_argument("--point_size", type=float, default=0.002, help='point size')  
    parser.add_argument("--scale_factor", type=float, default=1.0, help='point cloud scale factor for visualization')
    parser.add_argument("--port", type=int, default=7891, help='port')
    args = parser.parse_args()

    cameras_extrinsic_file = os.path.join(args.sfm_dir, "sparse", str(args.path_id), "images.txt")
    cameras_intrinsic_file = os.path.join(args.sfm_dir, "sparse", str(args.path_id), "cameras.txt")
    cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, 
        cam_intrinsics=cam_intrinsics, 
        images_folder=os.path.join(args.sfm_dir, "images"), sample_indices=np.array(list(range(len(cam_extrinsics)))))
    
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    
    FovXs = [cam_info.FovX for cam_info in cam_infos]
    FovYs = [cam_info.FovY for cam_info in cam_infos]
    Rs = torch.from_numpy(np.stack([cam_info.R for cam_info in cam_infos], axis=0)).float().cuda()
    Ts = torch.from_numpy(np.stack([cam_info.T for cam_info in cam_infos], axis=0)).float().cuda()
    view_mats = getWorld2View(Rs, Ts).inverse()

    depths = [torch.from_numpy(cam_info.depth).cuda() for cam_info in cam_infos]
    images = [torch.from_numpy(np.array(cam_info.image)).cuda() / 255.0 for cam_info in cam_infos]
    valid_masks = np.stack([skimage.morphology.binary_erosion(np.array(cam_info.valid)[:, :, 0], skimage.morphology.disk(3)) for cam_info in cam_infos], axis=0)
    valid_masks = torch.from_numpy(valid_masks).bool().cuda()

    H, W = images[0].shape[0:2]
    if args.downsample_ratio > 1:
        H, W = H // args.downsample_ratio, W // args.downsample_ratio
        depths = [F.interpolate(d[None, None], (H, W), mode='nearest-exact')[0, 0] for d in depths]
        images = [F.interpolate(i[None].permute(0,3,1,2), (H, W), mode='bilinear').permute(0,2,3,1)[0] for i in images]
        valid_masks = [F.interpolate(i[None, None].float(), (H, W), mode='bilinear')[0, 0] > 0.5 for i in valid_masks]

    if args.indices:
        indices = np.array(args.indices, dtype=np.int32)
    elif args.sample_num:
        indices = np.linspace(0, len(cam_extrinsics)-1, args.sample_num)
        indices = np.round(indices).astype(np.int32)
    else:
        indices = np.array(list(range(len(cam_extrinsics))))

    frames = []
    for i in tqdm(indices):
        edge_mask = compute_edge(depths[i])
        edge_mask = dilation_mask(edge_mask)
        
        K = np.array([
            [fov2focal(FovXs[i], W),0, W / 2],
            [0, fov2focal(FovYs[i], H), H / 2],
            [0, 0, 1],
        ]).astype(np.float32)
        T_world_camera = view_mats[i].cpu().numpy()

        frames.append(Record3dFrame(
            K=K,
            rgb=(images[i]*255).cpu().numpy().astype(np.uint8),
            depth=depths[i].cpu().numpy(),
            mask=((depths[i] > 1e-3)&(~edge_mask)&valid_masks[i]).cpu().numpy(),
            T_world_camera=T_world_camera
        ))

    server = viser.ViserServer(port=args.port)
    server.request_share_url()
    num_frames = len(frames)

    # Add playback UI.
    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
            disabled=False,
        )
        gui_next_frame = server.gui.add_button("Next Frame", disabled=False)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=False)
        gui_playing = server.gui.add_checkbox("Playing", False)
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=1, initial_value=20
        )
        gui_framerate_options = server.gui.add_button_group(
            "FPS options", ("10", "20", "30", "60")
        )

    # Frame step buttons.
    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

     # Disable frame controls when we're playing.
    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value

    # Set the framerate when we click one of the options.
    @gui_framerate_options.on_click
    def _(_) -> None:
        gui_framerate.value = int(gui_framerate_options.value)

    prev_timestep = gui_timestep.value

    # Toggle frame visibility when the timestep slider changes.
    @gui_timestep.on_update
    def _(_) -> None:
        global prev_timestep
        current_timestep = gui_timestep.value
        with server.atomic():

            # Toggle visibility.
            frame_nodes[current_timestep].visible = True
            frame_nodes[prev_timestep].visible = False
        prev_timestep = current_timestep
        server.flush()  # Optional!

    # Load in frames.
    server.scene.add_frame(
        "/frames",
        wxyz=tf.SO3.exp(np.array([np.pi / 2.0, 0.0, 0.0])).wxyz,
        position=(0, 0, 0),
        show_axes=False,
    )


    frame_nodes: list[viser.FrameHandle] = []
    for i in tqdm(range(num_frames)):
        # pcd, color = pcds[i], colors[i]

        frame = frames[i]
        position, color = frame.get_point_cloud()

        # Add base frame.
        frame_nodes.append(server.scene.add_frame(
            f"/frames/t{i}", 
            show_axes=False, 
            wxyz=tf.SO3.exp(np.array([0.0, 0.0, np.pi])).wxyz
        ))

        # Place the point cloud in the frame.
        server.scene.add_point_cloud(
            name=f"/frames/t{i}/point_cloud",
            points=position * args.scale_factor,
            colors=color,
            point_size=args.point_size * args.scale_factor,
            point_shape="rounded",
            wxyz=tf.SO3.from_matrix(frame.T_world_camera[:3, :3]).wxyz,
            position=frame.T_world_camera[:3, 3],
        )

        server.scene.add_frame(
            f"/frames/t{i}/frustum/axes",
            axes_length=0.02 * args.scale_factor,
            axes_radius=0.001 * args.scale_factor,
            wxyz=tf.SO3.from_matrix(frame.T_world_camera[:3, :3]).wxyz,
            position=frame.T_world_camera[:3, 3],
        )

    # Hide all but the current frame.
    for i, frame_node in enumerate(frame_nodes):
        frame_node.visible = i == gui_timestep.value

    # Playback update loop.
    prev_timestep = gui_timestep.value
    while True:
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames

        time.sleep(1.0 / gui_framerate.value)