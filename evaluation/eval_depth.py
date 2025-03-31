import numpy as np
import os
import torch
from metrics import *
import argparse
from tqdm import tqdm
import json
import h5py


device = 'cuda'
eval_metrics = [
    "depth_abs_relative_difference",
    "depth_delta1_acc",
]

def recover_scale_and_shift(points, points_gt, mask=None, weight=None):
    """
    Recover the scale factor for a point map with a target point map by minimizing the mse loss.
    points * scale + shift ~ points_gt

    ### Parameters:
    - `points: np.array` of shape (T, H, W, 3/1)
    - `points_gt: np.array` of shape (T, H, W, 3/1)
    - `mask: np.array` of shape (T, H, W) Optional.
    - `weight: np.array` of shape (T, H, W) Optional.
    
    ### Returns:
    - `scale`: the estimated scale factor
    - `shift`: the estimated shift factor
    """
    ndim = points.shape[-1]
    points = points.reshape(-1, ndim)
    points_gt = points_gt.reshape(-1, ndim)
    mask = None if mask is None else mask.reshape(-1)
    weight = None if weight is None else weight.reshape(-1)
    if mask is not None:
        points = points[mask]
        points_gt = points_gt[mask]
        weight = None if weight is None else weight[mask]
    tensor = np.zeros_like(points)
    tensor[:, -1] = 1
    # print(tensor.shape, tensor)
    # min_x ||Ax-b||_2^2
    A = np.stack([points, tensor], axis=-1).reshape(-1, 2)
    b = points_gt.reshape(-1, 1)
    if weight is not None:
        weight = np.tile(weight.reshape(-1, 1), (1, ndim)).reshape(-1, 1)
        A = A * weight
        b = b * weight
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    return x[0, 0], x[1, 0]

def eval_single_ssi_depth(
    pred_path, 
    gt_path, 
    use_weight=False,
    dataset_max_depth=70
):
    pred_data = np.load(pred_path)
    if 'disparity' in pred_data:
        pred_disp = pred_data['disparity'].astype(np.float32)
        pred_disp = np.clip(pred_disp, a_min=1e-3, a_max=None) 
        
        with h5py.File(gt_path, "r") as file:
            gt_mask = file['valid_mask'][:].astype(np.bool_)
            gt_pmap = file['point_map'][:].astype(np.float32)


        gt_disp = 1.0 / (gt_pmap[..., 2] + 1e-7)
        gt_depth = gt_pmap[..., 2]
        gt_mask = gt_mask & (gt_depth > 1e-3) & (gt_depth < dataset_max_depth)
        if gt_mask.sum() < 10:
            return [0.0, 0.0]
        
        assert pred_disp.shape == gt_disp.shape # t,h,w float32
        assert gt_mask.shape == gt_disp.shape # t,h,w float32
        
        
        scale, shift = recover_scale_and_shift(pred_disp[..., np.newaxis], gt_disp[..., np.newaxis], mask=gt_mask)

        aligned_pred = pred_disp * scale + shift
        aligned_pred = np.clip(aligned_pred, a_min=1e-3, a_max=None)    
        pred_depth = 1.0 / aligned_pred
        pred_depth = np.clip(
            pred_depth, a_min=1e-3, a_max=dataset_max_depth
        )
    elif 'point_map' in pred_data:
        pred_pmap = pred_data['point_map'].astype(np.float32)
        pred_mask = pred_data['mask'].astype(np.bool_)
        
        with h5py.File(gt_path, "r") as file:
            gt_mask = file['valid_mask'][:].astype(np.bool_)
            gt_pmap = file['point_map'][:].astype(np.float32)


        gt_depth = gt_pmap[..., 2]
        gt_mask = gt_mask & (gt_depth > 1e-3) & (gt_depth < dataset_max_depth)
        if gt_mask.sum() < 10:
            return [0.0, 0.0]

        assert pred_pmap.shape == gt_pmap.shape # t,h,w,3 float32
        assert pred_mask.shape == gt_mask.shape # t,h,w bool        

        scale, shift = recover_scale_and_shift(
            pred_pmap[..., 2:3], gt_pmap[..., 2:3], 
            mask=gt_mask,
            weight=1.0 / (gt_pmap[..., 2] + 1e-6) if use_weight else None)
        
        aligned_pred = pred_pmap[..., 2] * scale + shift
        pred_depth = np.clip(
            aligned_pred, a_min=1e-3, a_max=dataset_max_depth
        )
    else:
        pred_depth = pred_data['depth'].astype(np.float32)
        with h5py.File(gt_path, "r") as file:
            gt_mask = file['valid_mask'][:].astype(np.bool_)
            gt_pmap = file['point_map'][:].astype(np.float32)
        

        gt_depth = gt_pmap[..., 2]
        gt_mask = gt_mask & (gt_depth > 1e-3) & (gt_depth < dataset_max_depth)
        if gt_mask.sum() < 10:
            return [0.0, 0.0]

        assert pred_depth.shape == gt_depth.shape # t,h,w float32
        assert gt_mask.shape == gt_depth.shape # t,h,w float32
        
        
        scale, shift = recover_scale_and_shift(pred_depth[..., np.newaxis], gt_depth[..., np.newaxis], mask=gt_mask)

        aligned_pred = pred_depth * scale + shift
        pred_depth = np.clip(
            aligned_pred, a_min=1e-3, a_max=dataset_max_depth
        )

    # filter illegal frame
    valid_frame = gt_mask.sum((-1, -2)) > 0 
    pred_depth = pred_depth[valid_frame]
    gt_depth = gt_depth[valid_frame]
    gt_mask = gt_mask[valid_frame]

    d_rel_err = depth_rel_error(
        torch.from_numpy(pred_depth).to(device), 
        torch.from_numpy(gt_depth).to(device), 
        torch.from_numpy(gt_mask).to(device)
    ).item()
    d_in_percent = depth_inlier_percent(
        torch.from_numpy(pred_depth).to(device), 
        torch.from_numpy(gt_depth).to(device), 
        torch.from_numpy(gt_mask).to(device)
    ).item()

    return [d_rel_err, d_in_percent]



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_max_depth",
        type=int,
        default=70,
        help="Dataset max depth clip."
    )

    parser.add_argument(
        "--pred_data_dir",
        type=str,
        required=True,
        help="Predicted output directory."
    )

    parser.add_argument(
        "--gt_data_dir",
        type=str,
        required=True,
        help="data directory with GT."
    )

    parser.add_argument(
        "--use_weight",
        action="store_true",
        help="use weight map during alignment"
    )

    parser.add_argument(
        "--save_file_name",
        type=str,
        default="metrics_depth.json",
        help="save file name"
    )


    args = parser.parse_args()

    meta_file_path = os.path.join(args.gt_data_dir, 'filename_list.txt')
    assert os.path.exists(meta_file_path), meta_file_path

    samples = []        
    with open(meta_file_path, "r") as f:
        for line in f.readlines():
            video_path, data_path = line.split()
            samples.append(dict(
                video_path=video_path,
                data_path=data_path
            ))

    # iterate all cases
    results_all = []
    for i, sample in enumerate(tqdm(samples)):
        gt_path = os.path.join(args.gt_data_dir, sample['data_path'])
        pred_path = os.path.join(args.pred_data_dir,sample['video_path'][:-4]+'.npz')
        results_single = eval_single_ssi_depth(pred_path, gt_path, args.use_weight, args.dataset_max_depth)
        results_all.append(results_single)

    # avarage
    final_results =  np.array(results_all)
    final_results_mean = np.mean(final_results, axis=0)
    print("")

    # save mean to json
    result_dict = {}
    for i, m in enumerate(eval_metrics):
        result_dict[m] = final_results_mean[i]
        print(f"{m}: {final_results_mean[i]:04f}")

    # save each case to json
    for i, results in enumerate(results_all):
        result_dict[samples[i]['video_path']] = results

    # write json
    save_json_path = os.path.join(args.pred_data_dir, args.save_file_name)
    with open(save_json_path, 'w') as f:
        json.dump(result_dict, f, indent=4)
    print("")
    print(f"Evaluation results json are saved to {save_json_path}")
    
