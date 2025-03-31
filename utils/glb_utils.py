import trimesh
import numpy as np

def pmap_to_glb(point_map, valid_mask, frame) -> trimesh.Scene:


    pts_3d = point_map[valid_mask] * np.array([-1, -1, 1])
    pts_rgb = frame[valid_mask] 

    # Initialize a 3D scene
    scene_3d = trimesh.Scene()

    # Add point cloud data to the scene
    point_cloud_data = trimesh.PointCloud(
        vertices=pts_3d, colors=pts_rgb
    )
    
    scene_3d.add_geometry(point_cloud_data)
    return scene_3d
