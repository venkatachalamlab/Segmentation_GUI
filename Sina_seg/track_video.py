





import numpy as np  
import matplotlib.pyplot as plt
import pandas as pd
# import napari
from load_func import *
from Seg_pos import *
from Seg2graph import *
import time
import torch
from parameters import *
from tqdm import tqdm

import os
import napari
import numpy as np
from PIL import Image

import napari
import numpy as np
from PIL import Image
from qtpy.QtCore import QTimer
import gc

import os
import subprocess


folder_path = Path('ZM9624')





def combine_all_size(gray_volume):
    right = np.transpose(gray_volume,(2,1,0))
    bottom = np.transpose(gray_volume,(1,0,2))
    [z,y,x] = gray_volume.shape
    new = np.zeros([np.max([y,x]), y+z, x+z])
    new[0:z,0:y,0:x] =gray_volume
    new[0:x,0:y,x:x+z] = right
    new[0:y,y:y+z,0:x ] = bottom
    return new

def load_h5_variable(file_path,variable):
    f = h5py.File(file_path, "r") 
    data = f[variable][:]
    f.close()
    return data

model_2D = StarDist2D.from_pretrained('2D_versatile_fluo')


folder = 'track'
os.makedirs(folder, exist_ok=True)

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

    def capture_and_close(viewer, t_idx):
        img = viewer.screenshot(canvas_only=True)
        Image.fromarray(img).save(f'track/{t_idx}.png', dpi=(300, 300))
        viewer.close()
        del viewer
        gc.collect()

    QTimer.singleShot(100, lambda: capture_and_close(viewer, t_idx))
    napari.run()




    del img_original, combined_df_t_idx, seg, seg_new, gray_volume, new
   







files = sorted([f for f in os.listdir(folder) if f.endswith('.png')],
               key=lambda x: int(os.path.splitext(x)[0]))
for i, fname in enumerate(files):
    new_name = f"{i:04d}.png"
    os.rename(os.path.join(folder, fname), os.path.join(folder, new_name))


subprocess.run([
    'ffmpeg',
    '-framerate', '10',
    '-i', 'track/%04d.png',
    '-vf', 'scale=1920:1080',
    '-c:v', 'libx264',
    '-crf', '18',
    '-preset', 'slow',
    '-pix_fmt', 'yuv420p',
    'track.mp4'
])

