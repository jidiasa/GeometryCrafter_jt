import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pyquaternion import Quaternion
import math
from typing import NamedTuple
from PIL import Image
import cv2
from kornia.utils import create_meshgrid
from decord import cpu, VideoReader

def draw_track_on_image(image, track, track_vis):
    image = np.array(image) # RGB H,W,3 uint8
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for i in range(len(track)):
        x,y = int(track[i][0]), int(track[i][1])
        if track_vis[i]:
            image = cv2.circle(
                image,
                (x,y),
                int(4 * 2),
                (0, 0, 255),
                thickness=-1 if track_vis[i] else 2,
            )
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: float
    FovX: float
    width: int
    height: int

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def scale_inv_loss(input, target):
    return torch.abs(input / (target+1e-6) - 1) + torch.abs(target / (input+1e-6) - 1)


class CameraModel(nn.Module):
    
    def __init__(self, cam_infos, H, W):
        super().__init__()
        self.zfar = 100.0
        self.znear = 0.01
        
        rots = []
        trans = []
        FovXs = []
        FovYs = []
        for cam_info in cam_infos:
            q = Quaternion(matrix=cam_info.R, rtol=1e-03, atol=1e-04)
            rot = torch.from_numpy(q.elements).float().cuda()
            tran = torch.from_numpy(cam_info.T).float().cuda()  
            rots.append(rot)
            trans.append(tran)
            FovXs.append(cam_info.FovX)
            FovYs.append(cam_info.FovY)
            
        rots = torch.stack(rots, dim=0)
        trans = torch.stack(trans, dim=0)
        FovXs = torch.tensor(FovXs).cuda()
        FovYs = torch.tensor(FovYs).cuda()
        self.rots = nn.Parameter(rots.requires_grad_(True))
        self.trans = nn.Parameter(trans.requires_grad_(True))
        self.FovXs = nn.Parameter(FovXs.requires_grad_(True))
        self.FovYs = nn.Parameter(FovYs.requires_grad_(True))
        self.optimizer = None

        self.image_height = H
        self.image_width = W
        self.cnt = 0
        
    def setup_optim(self, cfg):
        
        l = [
            {'params': [self.rots], 'lr': cfg.camera_rotate_lr, "name": "cam_rots"},
            {'params': [self.trans], 'lr': cfg.camera_translate_lr, "name": "cam_trans"},
            {'params': [self.FovXs, self.FovYs], 'lr': cfg.camera_fov_lr, "name": "cam_fovs"},
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.proj_loss_w = cfg.proj_loss_w
        self.z_loss_w = cfg.z_loss_w

    def setup_init_track(self, tracks):
        self.tracks = []
        self.tracks_st_idx = dict()
        self.tracks_ed_idx = dict()
        cur_idx = 0
        # self.tracks_rgb = []
        for (st_frame, ed_frame, track, track_vis, track_query_time) in tracks:
            # self.tracks[st_frame] = track
            # img = images[st_frame]

            # uv = track[0, :, 0:2].unsqueeze(0).unsqueeze(1) # 1, 1, N, 2
            # rgb = torch.from_numpy(np.array(img)).to(uv.device).permute(2, 0, 1).unsqueeze(0).float() # 1, 3, H, W
            # rgb = F.grid_sample(rgb, uv, mode='bilinear') # 1, 3, 1, N
            # rgb = rgb.squeeze().permute(1, 0) # N, 3
            # self.tracks_rgb.append(rgb)
            Rs = self.build_rotation(self.rots[st_frame:st_frame+1])
            # 1, 3, 3
            Ts = self.trans[st_frame:st_frame+1]
            # 1, 3
            tanHalfFovXs = self.tanHalfFovXs[st_frame:st_frame+1]
            tanHalfFovYs = self.tanHalfFovYs[st_frame:st_frame+1]
            # 1, 4, 4
            view_mats = self.getWorld2View(Rs, Ts)
            view_mats_inv = torch.linalg.inv(view_mats)

            mask = (track_query_time == 0)

            T, N, _ = track[0:1, mask].shape
            z = track[0:1, mask, 2:3]
            # T, N, 1
            x = track[0:1, mask, 0:1] * tanHalfFovXs.reshape(T, 1, 1).repeat(1, N, 1) * z
            y = track[0:1, mask, 1:2] * tanHalfFovYs.reshape(T, 1, 1).repeat(1, N, 1) * z

            xyzw_world = torch.cat([x,y,z, torch.ones_like(x)], dim=-1) @ view_mats_inv.transpose(-2, -1)
            self.tracks.append(xyzw_world[..., :3].squeeze(0))
            
            self.tracks_st_idx[st_frame] = cur_idx
            cur_idx += N

            Rs = self.build_rotation(self.rots[ed_frame-1:ed_frame])
            # 1, 3, 3
            Ts = self.trans[ed_frame-1:ed_frame]
            # 1, 3
            tanHalfFovXs = self.tanHalfFovXs[ed_frame-1:ed_frame]
            tanHalfFovYs = self.tanHalfFovYs[ed_frame-1:ed_frame]
            # 1, 4, 4
            view_mats = self.getWorld2View(Rs, Ts)
            view_mats_inv = torch.linalg.inv(view_mats)

            mask = (track_query_time == ed_frame-st_frame-1)

            T, N, _ = track[-1:, mask].shape
            z = track[-1:, mask, 2:3]
            # T, N, 1
            x = track[-1:, mask, 0:1] * tanHalfFovXs.reshape(T, 1, 1).repeat(1, N, 1) * z
            y = track[-1:, mask, 1:2] * tanHalfFovYs.reshape(T, 1, 1).repeat(1, N, 1) * z

            xyzw_world = torch.cat([x,y,z, torch.ones_like(x)], dim=-1) @ view_mats_inv.transpose(-2, -1)
            self.tracks.append(xyzw_world[..., :3].squeeze(0))
            cur_idx += N

            self.tracks_ed_idx[st_frame] = cur_idx
            
        self.tracks = torch.concat(self.tracks, dim=0).requires_grad_(True)
        # self.tracks_rgb = torch.stack(self.tracks_rgb)
        # print(self.tracks.shape)
        self.tracks = nn.Parameter(self.tracks)

    def setup_optim_refine(self, cfg):
        l = [
            {'params': [self.rots], 'lr': cfg.camera_rotate_lr, "name": "cam_rots"},
            {'params': [self.trans], 'lr': cfg.camera_translate_lr, "name": "cam_trans"},
            {'params': [self.tracks], 'lr': cfg.track_xyz_lr, "name": "tracks"},
            # {'params': [self.FovXs, self.FovYs], 'lr': cfg.camera_fov_lr, "name": "cam_fovs"},
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.proj_loss_w = cfg.proj_loss_w

    def build_rotation(self, r):
        norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

        q = r / norm[:, None]

        R = torch.zeros((q.size(0), 3, 3), device='cuda')

        r = q[:, 0]
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]

        R[:, 0, 0] = 1 - 2 * (y*y + z*z)
        R[:, 0, 1] = 2 * (x*y - r*z)
        R[:, 0, 2] = 2 * (x*z + r*y)
        R[:, 1, 0] = 2 * (x*y + r*z)
        R[:, 1, 1] = 1 - 2 * (x*x + z*z)
        R[:, 1, 2] = 2 * (y*z - r*x)
        R[:, 2, 0] = 2 * (x*z - r*y)
        R[:, 2, 1] = 2 * (y*z + r*x)
        R[:, 2, 2] = 1 - 2 * (x*x + y*y)
        return R


    def getWorld2View(self, Rs, ts):
        Rt = torch.cat([Rs.transpose(1, 2), ts.unsqueeze(-1)], dim=-1)
        Rt = torch.cat([Rt, torch.tensor([0, 0, 0, 1], dtype=Rt.dtype, device=Rt.device).reshape(1, 4).repeat(Rt.shape[0], 1, 1)], dim=1)
        return Rt

    @property
    def tanHalfFovXs(self):
        return torch.tan(self.FovXs / 2)
    
    @property
    def tanHalfFovYs(self):
        return torch.tan(self.FovYs / 2)
    
    @property
    def proj_mats(self):
        tanHalfFovY = self.tanHalfFovYs.reshape(-1, 1)
        tanHalfFovX = self.tanHalfFovXs.reshape(-1, 1)

        device = tanHalfFovX.device
        zero = torch.zeros_like(tanHalfFovX)
        one = torch.ones_like(tanHalfFovX)

        top = tanHalfFovY * self.znear
        bottom = -top
        right = tanHalfFovX * self.znear
        left = -right
        z_sign = 1.0
        P = torch.cat([
            2.0 * self.znear / (right - left),
            zero,
            (right + left) / (right - left),
            zero,
            zero,
            2.0 * self.znear / (top - bottom),
            (top + bottom) / (top - bottom),
            zero,
            zero,
            zero,
            z_sign * self.zfar / (self.zfar - self.znear) * one,
            -(self.zfar * self.znear) / (self.zfar - self.znear) * one,
            zero,
            zero,
            z_sign*one,
            zero
        ], dim=-1).reshape(-1, 4, 4).float()        
        return P
        

    def train_one_step(self, st_frame, ed_frame, track, track_vis):
        
        # self.optimizer.zero_grad()

        Rs = self.build_rotation(self.rots[st_frame:ed_frame])
        # T, 3, 3
        Ts = self.trans[st_frame:ed_frame]
        # T, 3
        tanHalfFovXs = self.tanHalfFovXs[st_frame:ed_frame]
        tanHalfFovYs = self.tanHalfFovYs[st_frame:ed_frame]
        proj_mats = self.proj_mats[st_frame:ed_frame]
        # T, 4, 4
        view_mats = self.getWorld2View(Rs, Ts)
        view_mats_inv = torch.linalg.inv(view_mats)
        full_proj_mats = proj_mats @ view_mats
        # T, 4, 4

        T, N, _ = track.shape
        z = track[:, :, 2:3]
        # T, N, 1
        x = track[:, :, 0:1] * tanHalfFovXs.reshape(T, 1, 1).repeat(1, N, 1) * z
        y = track[:, :, 1:2] * tanHalfFovYs.reshape(T, 1, 1).repeat(1, N, 1) * z

        xyzw_world = torch.cat([x,y,z, torch.ones_like(x)], dim=-1) @ view_mats_inv.transpose(-2, -1)
        # T, N, 4
        
        # track_vis T, N
        track_vis_list = track_vis.float().unsqueeze(0).repeat(T, 1, 1) * track_vis.float().unsqueeze(1).repeat(1, T, 1)

        full_proj_mats_list = full_proj_mats.unsqueeze(1).repeat(1, T, 1, 1)
        # T(j), T(i), 4, 4
        # proj i-th frame track point to j-th frame
        xyzw_world_list = xyzw_world.unsqueeze(0).repeat(T, 1, 1, 1)
        # T(j), T(i), N, 4
        xyzw_proj_list = (xyzw_world_list.reshape(-1, N, 4) @ full_proj_mats_list.reshape(-1, 4, 4).permute(0, 2, 1)).reshape(T, T, N, 4)
        proj_z_list = xyzw_proj_list[..., 3:4]
        # T(j), T(i), N, 1
        # print(proj_z_list.min(), proj_z_list.max())
        track_vis_list = torch.logical_and(track_vis_list, proj_z_list.squeeze(-1) > 0.1)
        proj_z_list = torch.clamp_min(proj_z_list, 0.1)
        
        proj_xy_list = xyzw_proj_list[..., 0:2] / proj_z_list
        # T(j), T(i), N, 2
        proj_xy_gt = track[..., :2].unsqueeze(1).repeat(1, T, 1, 1) 

        # if images is not None:
        #     track_img = draw_track_on_image(images[29], ((1+proj_xy_list[29][0]) / 2).cpu().detach().numpy() * np.array([self.image_width, self.image_height]), track_vis_list[29][0].cpu().detach().numpy())
        #     track_img.save('tmpdd_%d.png' % self.cnt)
        #     self.cnt += 1

        xy_loss = F.mse_loss(proj_xy_list, proj_xy_gt, reduction='none').sum(dim=-1)
        xy_loss = (xy_loss * track_vis_list).sum() / track_vis_list.sum()
        proj_z_gt = track[..., 2:3].unsqueeze(1).repeat(1, T, 1, 1)
        track_vis_list = torch.logical_and(track_vis_list, proj_z_gt.squeeze(-1) > 0.1)
        # T(j), T(i), N, 2
        # z_loss = scale_inv_loss(proj_z_list, proj_z_gt).squeeze(-1)
        # z_loss = (z_loss * track_vis_list).sum() / track_vis_list.sum()

        loss = xy_loss * self.proj_loss_w #+ z_loss * self.z_loss_w
        # loss_value = loss.item()
        # loss.backward()
        # self.optimizer.step()
        return loss, dict(loss=loss.item())

    def train_one_step_refine(self, idx, st_frame, ed_frame, track_gt, track_vis):
        
        Rs = self.build_rotation(self.rots[st_frame:ed_frame])
        # T, 3, 3
        Ts = self.trans[st_frame:ed_frame]
        proj_mats = self.proj_mats[st_frame:ed_frame]
        # T, 4, 4
        view_mats = self.getWorld2View(Rs, Ts)
        full_proj_mats = proj_mats @ view_mats
        # T, 4, 4

        T, N = track_vis.shape
        st_idx = self.tracks_st_idx[st_frame]
        ed_idx = self.tracks_ed_idx[st_frame]
        track = self.tracks[st_idx:ed_idx].unsqueeze(0).repeat(T, 1, 1)
        # T, N, 3
        # print(track.shape, st_idx, ed_idx)
        xyzw_world = torch.cat([track, torch.ones_like(track[..., 0:1])], dim=-1)
        # T, N, 4

        # T(i), 4, 4
        xyzw_proj = (xyzw_world @ full_proj_mats.permute(0, 2, 1))
        proj_z = xyzw_proj[..., 3:4]
        track_vis = torch.logical_and(track_vis, proj_z.squeeze(-1) > 0.1)
        proj_z = torch.clamp_min(proj_z, 0.05)
        
        proj_xy = xyzw_proj[..., 0:2] / proj_z
        # T(i), N, 2
        proj_xy_gt = track_gt[..., :2]
        proj_z_gt = torch.clamp_min(track_gt[..., 2:3], 0.05) 

        # if images is not None:
        #     track_img = draw_track_on_image(images[29], ((1+proj_xy[29]) / 2).cpu().detach().numpy() * np.array([self.image_width, self.image_height]), track_vis[29].cpu().detach().numpy())
        #     track_img.save('tmpdx_%d.png' % self.cnt)
        #     self.cnt += 1

        xy_loss = F.mse_loss(proj_xy, proj_xy_gt, reduction='none').sum(dim=-1)
        xy_loss = (xy_loss * track_vis).sum() / track_vis.sum()
        z_loss = F.mse_loss(proj_z/proj_z_gt, torch.ones_like(proj_z_gt), reduction='none').sum(dim=-1)
        z_loss = (z_loss * track_vis).sum() / track_vis.sum()
        
        loss = xy_loss * self.proj_loss_w + z_loss * self.z_loss_w 
        return loss, dict(
            xy_loss=xy_loss,
            z_loss=z_loss
        )

def save_cameras(focals, principal_points, sparse_path, imgs_shape):
    # Save cameras.txt
    os.makedirs(sparse_path, exist_ok=True)
    cameras_file = os.path.join(sparse_path, 'cameras.txt')
    with open(cameras_file, 'w') as cameras_file:
        cameras_file.write("# Camera list with one line of data per camera:\n")
        cameras_file.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i, (focal, pp) in enumerate(zip(focals, principal_points)):
            cameras_file.write(f"{i} PINHOLE {imgs_shape[1]} {imgs_shape[0]} {focal[0]} {focal[1]} {pp[0]} {pp[1]}\n")



def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def save_imagestxt(rots, trans, sparse_path):
     # Save images.txt
    images_file = os.path.join(sparse_path, 'images.txt')
    # Generate images.txt content
    with open(images_file, 'w') as images_file:
        images_file.write("# Image list with two lines of data per image:\n")
        images_file.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        images_file.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i in range(len(rots)):
            # Convert rotation matrix to quaternion
            rotation_matrix = rots[i]
            qw, qx, qy, qz = rotmat2qvec(rotation_matrix)
            tx, ty, tz = trans[i]
            images_file.write(f"{i} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {i} {str(i).zfill(5)}.jpg\n")
            images_file.write("\n") # Placeholder for points, assuming no points are associated with images here


def point_map_xy2intrinsic(point_map_xy):
    # *,h,w,2
    height, width = point_map_xy.shape[-3], point_map_xy.shape[-2]
    assert height % 2 == 0
    assert width % 2 == 0
    mesh_grid = create_meshgrid(
        height=height,
        width=width,
        normalized_coordinates=True,
        device=point_map_xy.device,
        dtype=point_map_xy.dtype
    )[0] # h,w,2
    assert mesh_grid.abs().min() > 1e-4
    # *,h,w,2
    mesh_grid = mesh_grid.expand_as(point_map_xy)
    nc = point_map_xy.mean(dim=-2).mean(dim=-2) # *, 2
    nc_map = nc[..., None, None, :]
    nf = ((point_map_xy - nc_map) / mesh_grid).mean(dim=-2).mean(dim=-2)
    nf_map = nf[..., None, None, :]
    # *, 1, 1, 2
    # print((mesh_grid * nf_map + nc_map - point_map_xy).abs().max())

    return torch.cat([nc_map, nf_map], dim=-1)[..., 0, 0, :] # *, 4 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', required=True, type=str)
    parser.add_argument('--track_dir', required=True, type=str)
    parser.add_argument('--mask_path', required=True, type=str)
    parser.add_argument('--point_map_path', required=True, type=str)
    parser.add_argument('--out_dir', required=True, type=str)
    parser.add_argument('--path_id', default=0, type=int)
    parser.add_argument('--num_iterations', type=int, default=1000)
    parser.add_argument('--proj_loss_w', type=float, default=1.0)
    parser.add_argument('--z_loss_w', type=float, default=0.0)
    parser.add_argument('--camera_rotate_lr', type=float, default=1e-3)
    parser.add_argument('--camera_translate_lr', type=float, default=1e-2)
    parser.add_argument('--track_xyz_lr', type=float, default=1e-2)
    parser.add_argument('--camera_fov_lr', type=float, default=1e-4)
    parser.add_argument('--num_points_per_frame', type=int, default=10000)
    parser.add_argument('--use_ori_res', action='store_true')
    parser.add_argument('--use_refine', action='store_true')
    args = parser.parse_args()

    vid = VideoReader(args.video_path, ctx=cpu(0))
    images = vid.get_batch(list(range(0, len(vid)))).asnumpy()    
    seq_len = images.shape[0]

    vid = VideoReader(args.mask_path, ctx=cpu(0))
    masks = vid.get_batch(list(range(0, len(vid)))).asnumpy() 

    data = np.load(args.point_map_path)
    point_map = torch.from_numpy(data['point_map']).float()
    valid_masks = torch.from_numpy(data['mask']).bool()
    intr = point_map_xy2intrinsic(point_map[..., 0:2] / (point_map[..., 2:3] + 1e-6))
    depths = point_map[..., 2]

    if args.use_ori_res:
        H, W = images.shape[1], images.shape[2]
        valid_masks = F.interpolate(valid_masks.float().unsqueeze(1), size=(H,W), mode='nearest-exact').squeeze(1).bool()
        depths = F.interpolate(depths.unsqueeze(1), size=(H,W), mode='nearest-exact').squeeze(1)
    else:
        H, W = depths.shape[1:3]
    
    os.makedirs(os.path.join(args.out_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'masks'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'valid_masks'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'depths'), exist_ok=True)
    
    for i in range(seq_len):
        img = images[i]
        if not args.use_ori_res:
            img = cv2.resize(img, (W, H))
        mask = masks[i]
        if not args.use_ori_res:
            mask = cv2.resize(mask, (W, H))
        Image.fromarray(img).save(os.path.join(args.out_dir, 'images', str(i).zfill(5) + ".jpg"))                
        mask = (255 - mask)[..., 0]
        Image.fromarray(mask, 'L').save(os.path.join(args.out_dir, 'masks', str(i).zfill(5) + ".png"))
        v_mask = (valid_masks[i].cpu().float().numpy() * 255).astype(np.uint8)
        Image.fromarray(v_mask, 'L').save(os.path.join(args.out_dir, 'valid_masks', str(i).zfill(5) + ".png"))
        np.save(os.path.join(args.out_dir, 'depths', str(i).zfill(5) + ".npy"), depths[i])

    cam_infos = []
    for i in range(seq_len):
        # intr[i][2] ~ W/(2fx)
        # fov ~ 2*math.atan(W/(2*fx))
        cam_info = CameraInfo(
            uid=0, 
            R=np.eye(3), 
            T=np.zeros(3,), 
            FovX=2*math.atan(intr[i][2]),
            FovY=2*math.atan(intr[i][3]),
            width=W,
            height=H
        )
        cam_infos.append(cam_info)
    
    camera_model = CameraModel(cam_infos, H, W)
    camera_model.setup_optim(args)

    tracks = []

    track_list = sorted(filter(lambda x:x.endswith(".npz") ,os.listdir(args.track_dir)))
    track_list = [os.path.join(args.track_dir, path) for path in track_list]
    
    # track_list = track_list[:1]

    depths = depths.float().cuda()
    # b, h, w

    for i in range(len(track_list)):
        track_path = track_list[i]
        track_name = os.path.basename(track_path)
        st_frame = int(track_name[:-4])
        data = np.load(track_path)
        track = data['tracks']
        track_vis = data['visible']
        track_query_time = data['query_time']
        ed_frame = st_frame + len(track)
        track = torch.from_numpy(track).float().cuda()
        # T, N, 2
        track_vis = torch.from_numpy(track_vis).cuda()
        # T, N
        track_depth = depths[st_frame:ed_frame]
        # T, H, W
        
        grid = track.clone()
        grid[:, :, 0] = (grid[:, :, 0] / W) * 2 - 1
        grid[:, :, 1] = (grid[:, :, 1] / H) * 2 - 1
        track_vis = torch.logical_and(track_vis, grid[:, :, 0] > -1.0)
        track_vis = torch.logical_and(track_vis, grid[:, :, 0] < 1.0)
        track_vis = torch.logical_and(track_vis, grid[:, :, 1] > -1.0)
        track_vis = torch.logical_and(track_vis, grid[:, :, 1] < 1.0)
        grid = torch.clamp(grid, -1, 1)
        grid = grid.unsqueeze(1) # T, 1, N, 2
        grid_depth = F.grid_sample(track_depth.unsqueeze(1), grid, mode='nearest', align_corners=False)
        # T, 1, 1, N
        track = torch.cat([grid.squeeze(), grid_depth.squeeze().unsqueeze(-1)], dim=-1).detach()
        # T, N, 3
        tracks.append((st_frame, ed_frame, track, track_vis, track_query_time))
    
    ema_loss_for_log = None
    progress_bar = tqdm(range(0, args.num_iterations))


    for iter in range(0, args.num_iterations):
        loss = 0.0
        camera_model.optimizer.zero_grad()
        for (st_frame, ed_frame, track, track_vis, track_query_time) in tracks:
            loss_track, loss_info = camera_model.train_one_step(st_frame, ed_frame, track, track_vis)
            loss += loss_track
        
        loss.backward()
        camera_model.optimizer.step()

        if ema_loss_for_log is None:
            ema_loss_for_log = {k:0 for k in loss_info}
        for k in loss_info:
            ema_loss_for_log[k] = 0.4 * loss_info[k] + 0.6 * ema_loss_for_log[k]
        if iter % 20 == 0:
            progress_bar.set_postfix({
                k: f"{ema_loss_for_log[k]:.{6}f}" for k in ema_loss_for_log  
            })
            progress_bar.update(20)
    else:
        progress_bar.close()

    if args.use_refine:
        ################ refine optimization ###################
        camera_model.setup_init_track(tracks)
        camera_model.setup_optim_refine(args)

        ema_loss_for_log = None
        progress_bar = tqdm(range(0, args.num_iterations))


        for iter in range(0, args.num_iterations):
            loss = 0.0
            camera_model.optimizer.zero_grad()
            for idx, (st_frame, ed_frame, track, track_vis, track_query_time) in enumerate(tracks):
                loss_track, loss_info = camera_model.train_one_step_refine(idx, st_frame, ed_frame, track, track_vis)
                loss += loss_track
            
            loss.backward()
            camera_model.optimizer.step()

            if ema_loss_for_log is None:
                ema_loss_for_log = {k:0 for k in loss_info}
            for k in loss_info:
                ema_loss_for_log[k] = 0.4 * loss_info[k] + 0.6 * ema_loss_for_log[k]
            if iter % 20 == 0:
                progress_bar.set_postfix({
                    k: f"{ema_loss_for_log[k]:.{6}f}" for k in ema_loss_for_log  
                })
                progress_bar.update(20)
        else:
            progress_bar.close()

    save_cameras(
        [(fov2focal(camera_model.FovXs[idx], cam.width), fov2focal(camera_model.FovYs[idx], cam.height)) for idx, cam in enumerate(cam_infos)],
        [(cam.width / 2, cam.height / 2) for cam in cam_infos],
        os.path.join(args.out_dir, "sparse", str(args.path_id)),
        (cam_infos[0].height, cam_infos[0].width))


    Rs = camera_model.build_rotation(camera_model.rots).detach().cpu().numpy()
    Ts = camera_model.trans.detach().cpu().numpy()

    save_imagestxt(
        [R.T for R in Rs],
        [T for T in Ts], 
        os.path.join(args.out_dir, "sparse", str(args.path_id)))

