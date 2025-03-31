import torch
from matplotlib import cm

def robust_min_max(tensor, quantile=0.99):
    T, H, W = tensor.shape
    min_vals = []
    max_vals = []
    for i in range(T):
        min_vals.append(torch.quantile(tensor[i], q=1-quantile, interpolation='nearest').item())
        max_vals.append(torch.quantile(tensor[i], q=quantile, interpolation='nearest').item())
    return min(min_vals), max(max_vals) 


class ColorMapper:
    def __init__(self, colormap: str = "inferno"):
        self.colormap = torch.tensor(cm.get_cmap(colormap).colors)

    def apply(self, image: torch.Tensor, v_min=None, v_max=None):
        # assert len(image.shape) == 2
        if v_min is None:
            v_min = image.min()
        if v_max is None:
            v_max = image.max()
        image = (image - v_min) / (v_max - v_min)
        image = (image * 255).long()
        colormap = self.colormap.to(image.device)
        image = colormap[image]
        return image

def color_video_disp(disp):
    visualizer = ColorMapper()
    disp_img = visualizer.apply(disp, v_min=0, v_max=1)
    return disp_img     

def pmap_to_disp(point_maps, valid_masks):
    disp_map = 1.0 / (point_maps[..., 2] + 1e-4)
    min_disparity, max_disparity = robust_min_max(disp_map)
    disp_map = torch.clamp((disp_map - min_disparity) / (max_disparity - min_disparity+1e-4), 0, 1)
    
    disp_map = color_video_disp(disp_map)
    disp_map[~valid_masks] = 0
    return disp_map
    # imageio.mimsave(os.path.join(args.save_dir, os.path.basename(args.data[:-4])+'_disp.mp4'), disp, fps=24, quality=9, macro_block_size=1)
