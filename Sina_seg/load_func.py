from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import os
import time
import re
import copy


def find_global_worldline_map(root_dir):
    train_path = find_paths_with_worldlines(root_dir,'worldlines.h5')
    worldline_all_df = pd.DataFrame()
    for folder_path in train_path:
        worldline_df = load_worldline_h5(folder_path/('worldlines.h5'))
        worldline_all_df = pd.concat([worldline_all_df,worldline_df],axis=0)
    global_worline_map = list(np.unique(worldline_all_df['name']))
    # global_worline_map.append('test') ## need to be modified
    # global_worline_map.insert(0,'noname')
    global_worline_map_dict = {name: i for i, name in enumerate(global_worline_map)}
    return global_worline_map_dict


def extract_symmetric_pair_name(global_worline_map_dict):
    """
    Extract all the left and right symmetric pairs of label name
    """

    symmetric_pairs = []
    keys = [k for k in global_worline_map_dict.keys() if k]  # Use byte strings directly

    for key in keys:
        if key[-1:] in b'LR':  # Check if the key ends with 'L' or 'R'
            opposite = key[:-1] + (b'R' if key[-1:] == b'L' else b'L')
            if opposite in keys:
                pair = [key, opposite]
                if pair not in symmetric_pairs and pair[::-1] not in symmetric_pairs:  # Avoid duplicates
                    symmetric_pairs.append(pair)
    return symmetric_pairs

def extract_sysmmetric_pair_coords(folder_path,combined_df_t_idx,norm_scale,symmetric_pairs):

    worldline_df = load_worldline_h5(folder_path/'worldlines.h5')
    worldline_df.rename(columns = {'id':'worldline_id'},inplace = True)
    merged_df = pd.merge(worldline_df, combined_df_t_idx, on='worldline_id', how='inner')
    coords_pair = []
    for L_name, R_name in symmetric_pairs:
        filtered_df = merged_df[merged_df['name'].isin([L_name, R_name])]
        if filtered_df.shape[0] == 2:
            pos = np.array(filtered_df[['z','y','x']])
            pair_pos = (np.round(pos * (np.array(norm_scale)))).astype(int)
            coords_pair.append(pair_pos)
        elif filtered_df.shape[0] == 1:
            print('one side missing')
        
        else:
            print("missing", [L_name, R_name])
            # pass
    return np.array(coords_pair)

def find_paths_with_worldlines(root_dir,search_file):
    '''
    Find the paths with the worldlines.h5 file
    '''
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        if search_file in filenames:
            paths.append(Path(dirpath))
    return paths



def load_seg_h5(seg_path):
    with h5py.File(seg_path, 'r') as hdf:
        seg = hdf['label'][:]
    hdf.close()
    return seg

def save_seg_h5(seg_path,seg):
    with h5py.File(seg_path, 'w') as hdf:
        hdf.create_dataset('label', data=seg)
    hdf.close()


def get_volume_at_frame(file_name,t_idx):
    '''
    Get the 3D original volume at frame t_idx from the h5 file_name
    '''
    with h5py.File(file_name, 'r') as f:
        img_original = np.array(f['data'][t_idx:t_idx+1])
        mask = None
    f.close()
    return img_original,mask 


def load_worldline_h5(file_name):
    '''
    Load worldline from the h5 file
    '''
    with h5py.File(file_name, 'r') as hdf: 
        dfs = []
        for key in hdf.keys():
            dataset = hdf[key]
            df = pd.DataFrame(dataset[:],columns=[key])
            dfs.append(df)

        combined_df = pd.concat(dfs, axis = 1)
    hdf.close()
    return combined_df


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



def extract_time_paris(file_path):
    pattern = r'Frame #\d+\s+Parent #\d+\s+Reference #\d+\s+Distance to parent: d=\d+\.\d+'
    matching_lines = []
    with open(file_path, 'r') as file:
        for line in file:
            # Search for lines that match the pattern
            if re.search(pattern, line):
                matching_lines.append(line.strip())

    extracted_pairs = []

    # Regular expression to match frame and parent numbers
    pattern_t = r'Frame #(\d+)\s+Parent #(\d+)'


    for line in matching_lines:
        match = re.search(pattern_t, line)
        if match:
            # Append the extracted numbers as [parent, frame] to the list
            parent = int(match.group(2))  # Parent number
            frame = int(match.group(1))   # Frame number
            extracted_pairs.append([parent, frame])
    return extracted_pairs


def extract_time_paris_reference(file_path):
    pattern = r'Frame #\d+\s+Parent #\d+\s+Reference #\d+\s+Distance to parent: d=\d+\.\d+'
    matching_lines = []
    with open(file_path, 'r') as file:
        for line in file:
            # Search for lines that match the pattern
            if re.search(pattern, line):
                matching_lines.append(line.strip())

    extracted_pairs = []

    # Regular expression to match frame and parent numbers
    pattern_t = r'Frame #(\d+)\s+Parent #(\d+)\s+Reference #(\d+)'


    for line in matching_lines:
        match = re.search(pattern_t, line)
        if match:
            # Append the extracted numbers as [parent, frame] to the list
            parent = int(match.group(3))  # Parent number
            frame = int(match.group(1))   # Frame number
            extracted_pairs.append([parent, frame])
    return extracted_pairs  
    




def get_annotation_file_df(dataset: Path, file_name: str) -> pd.DataFrame:
    """Load and return annotations as an ordered pandas dataframe.
    This should contain the following:
    - t_idx: time index of each annotation
    - x: x-coordinate as a float between (0, 1)
    - y: y-coordinate as a float between (0, 1)
    - z: z-coordinate as a float between (0, 1)
    - worldline_id: track or worldline ID as an integer
    - provenance: scorer or creator of the annotation as a byte string
    """

    with h5py.File(dataset / file_name, 'r') as f:
        data = pd.DataFrame()
        for k in f:
            data[k] = f[k]
    return data



def get_annotation_df(dataset: Path) -> pd.DataFrame:
    """Load and return annotations as an ordered pandas dataframe.
    This should contain the following:
    - t_idx: time index of each annotation
    - x: x-coordinate as a float between (0, 1)
    - y: y-coordinate as a float between (0, 1)
    - z: z-coordinate as a float between (0, 1)
    - worldline_id: track or worldline ID as an integer
    - provenance: scorer or creator of the annotation as a byte string
    """
    with h5py.File(dataset / 'annotations.h5', 'r') as f:
        data = pd.DataFrame()
        for k in f:
            data[k] = f[k]
    return data


def save_pandas_h5(save_h5_path, df):
    with h5py.File(save_h5_path, 'w') as hdf:
        for column in df.columns:
            data = df[column].to_numpy()
            if data.dtype == object:
                data = data.astype(h5py.string_dtype())
            hdf.create_dataset(column, data=data)
    hdf.close()





############################################################################################################

def get_all_datasets_foldername(folder_path):
    folder_list = []
    for root, dirs, files in os.walk(folder_path):
        for dir in dirs:
            folder_list.append(Path(os.path.join(root, dir) + '/'))
    return folder_list



def img_orig_rgb_color(img_orig):
    # ["mneptune", "cyofp", "bfp", "tagrfp", "gcamp", "wf"]
    # Red = channel 0 (mneptune)
    # Green = channel 1 - 0.1 * channel 4 (cyofp - 0.1*gcamp)
    # Blue = channel 2 (bfp)
    img_R = copy.deepcopy(img_orig[0])
    img_G = np.clip(copy.deepcopy(img_orig[1]) - 0.1*copy.deepcopy(img_orig[4]), 0,255)
    img_B = img_orig[2]
    # img_trans = np.transpose(np.array([img_R,img_G,img_B]),(2,3,1,0))
    img_trans = np.transpose(np.array([img_R,img_G,img_B]),(1,2,3,0))
    return img_trans







