import gc
import napari
import numpy as np
from Seg_pos import *
from load_func import *
from stardist import Rays_GoldenSpiral
from stardist.models import Config3D, StarDist3D, StarDistData3D
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from qtpy.QtCore import QTimer



def load_annotations_h5_t_idx(file_name,t_idx):
    '''
    Load annotation at all time 
    '''
    with h5py.File(file_name, 'r') as hdf: 
        dfs = []
        for key in hdf.keys():
            dataset = hdf[key]
            df = pd.DataFrame(dataset[:],columns=[key])
            dfs.append(df)

        combined_df = pd.concat(dfs, axis = 1)
    hdf.close()
    combined_df_t_idx = combined_df[combined_df['t_idx'] == t_idx]
    return combined_df_t_idx

def get_abs_pos(combined_df_t_idx,labels_shape):
    '''
    Obtain the absolutely coordinates annotations at the time t_idx with the segmented size of labels_shape from the normalized coordinates
    !!! Important notice: the absolute coordinates starting from 0, then it should be multiplied by (labels_shape-1)
    '''
    pos = np.array(combined_df_t_idx[['z','y','x']])
    abs_pos = (np.round(pos * (np.array(labels_shape)-1))).astype(int)
    return abs_pos


def load_model_2D():
    # if X[0].ndim == 3 else X[0].shape[-1]
    n_channel = 1 
    # anisotropy=(5.333333333333333, 1.1428571428571428, 1.0)
    anisotropy=(5.0, 1.0, 1.0)
    n_rays = 96
    # Use OpenCL-based computations for data generator during training (requires 'gputools')
    use_gpu = False and gputools_available()
    use_gpu = True
    print("use_gpu: ", use_gpu)
    # Predict on subsampled grid for increased efficiency and larger field of view
    grid = tuple(1 if a > 1.5 else 2 for a in anisotropy)
    rays = Rays_GoldenSpiral(n_rays, anisotropy=anisotropy)

    conf = Config3D (
        rays             = rays,
        grid             = grid,
        anisotropy       = anisotropy,
        use_gpu          = use_gpu,
        n_channel_in     = n_channel,
        # adjust for your data below (make patch size as large as possible)
        train_patch_size = (8,128,128),
        train_batch_size = 6,
    )

    StarDist2D.from_pretrained()
    model_2D = StarDist2D.from_pretrained('2D_versatile_fluo')
    return model_2D


def combine_all_size(gray_volume):
    [z,y,x]= gray_volume.shape
    new = np.zeros((np.max([y,x]), y+z, x+z))
    new[0:z,0:y,0:x] = gray_volume
    new[0:x,0:y,x:x+z] = np.transpose(gray_volume,(2,1,0))
    new[0:y,y:y+z,0:x] = np.transpose(gray_volume,(1,0,2))
    return new




isotropy_scale = (5, 1, 1)
normalize_lim = (3, 99.5)
zoom_factor = 1


model_2D = load_model_2D()


folder_path =Path('ZM9624')


for t_idx in tqdm(range(1060)):

    
    img_original, _ = get_volume_at_frame(folder_path/'data.h5',t_idx)
    combined_df_t_idx = load_annotations_h5_t_idx(folder_path/('coordinates.h5'),t_idx)
    abs_pos = get_abs_pos(combined_df_t_idx,img_original.shape[-3:])
    gray_volume = img_original[0,1]
    apply_NucleiSegmentation = NucleiSegmentationAnnotation(gray_volume/np.max(gray_volume) *255, model_2D, isotropy_scale,  normalize_lim, zoom_factor)
    seg_0 = apply_NucleiSegmentation.run_segmentation(combined_df_t_idx)
    seg = post_process_worldline_seg(combined_df_t_idx, abs_pos, seg_0)
    new = combine_all_size(gray_volume)
    seg_new = combine_all_size(seg)



    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(new, name='gray', colormap='gray')
    viewer.add_labels(seg_new.astype(np.int16))
    viewer.dims.ndisplay = 3

    def capture_and_close():
        img = viewer.screenshot(canvas_only=True)
        Image.fromarray(img).save('track/' + str(t_idx) + '.png', dpi=(300, 300))
        viewer.close()  

    QTimer.singleShot(100, capture_and_close)  # 1000 ms delay to allow rendering
    napari.run()

    del img_original, combined_df_t_idx, seg, seg_new, gray_volume, new
   
    # napari.close()






