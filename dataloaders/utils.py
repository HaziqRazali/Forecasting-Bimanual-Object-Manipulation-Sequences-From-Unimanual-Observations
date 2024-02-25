import numpy as np
from collections.abc import Iterable

def one_hot(labels, max_label, set_negatives_to_zero=False, labels_shape=None):

    # copy so i dont overwrite
    labels_copy = np.copy(labels)
    
    # if labels_copy is empty
    if labels_copy.size == 0:
        one_hot_labels = np.zeros(labels_shape)
        one_hot_labels = one_hot_labels.astype(np.int64)
        return one_hot_labels
    
    # else do normally
    if set_negatives_to_zero == True:
        labels_copy[labels_copy < 0] = 0

    one_hot_labels = np.zeros((labels_copy.size, max_label))
    one_hot_labels[np.arange(labels_copy.size),labels_copy] = 1
    one_hot_labels = one_hot_labels.astype(np.int64)
    return one_hot_labels

# for each frame, we get the time_segmentation it lies in and extract the corresponding segment_data
# - len(frames) 
# - len(segment_data) = num_segments
# - len(time_segmentations) = num_segments
def extract_data(frames, time_segmentations, segment_data, mode="frame"):
                                      
    extracted_list = []    
    if mode == "frame":
        for frame in frames:
            idx = np.where((frame >= time_segmentations[:,0]) & (frame < time_segmentations[:,1]))
            idx = idx[0]
            if len(idx) == 0:
                idx = segment_data[-1] # frame has exceeded time_segmentations so take the final entry
            else:
                idx = idx[0]
                extracted_entry = segment_data[idx]
            extracted_list.append(extracted_entry)
            
    elif mode == "segment":
        for i,time_segmentation in enumerate(time_segmentations):
            #print("frames[0], frames[-1], time_segmentation", frames[0], frames[-1], time_segmentation)
            if time_segmentation[0] <= frames[-1] and frames[0] <= time_segmentation[1]:
                extracted_list.append(segment_data[i])
                
    else:
        print("Unknown mode", mode)
        sys.exit()
    return extracted_list

# for each timestep, we convert the id to idx given the reference
# - len(ids) = frames
def get_idxs(ids, reference):
                        
    reference = reference
    idxs = []
    for id_t in ids:
        idx = np.where(reference == id_t)[0]
        idx = -1 if len(idx) == 0 else idx[0]
        idxs.append(idx)
    return idxs

def sort_actions(sub_action_list, full_action_list):
    
    full_action_dict = {x:i for i,x in enumerate(full_action_list)}
    
    sub_action_key = sub_action_list
    sub_action_value = [full_action_dict[key] for key in sub_action_key]
    sub_action_value_sorted_idx = np.argsort(sub_action_value)
    sub_action_key_sorted = [sub_action_key[i] for i in sub_action_value_sorted_idx]
        
    return sub_action_key_sorted

def flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x

def get_dct_matrix(N):

    """
    N = sequence length certainly impacts the values since its a denominator
    """

    dct_m = np.eye(N)
    
    # k'th row
    for k in np.arange(N):
    
        # i'th column 
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m
    
def detach(adj, idx):
    
    i = np.array([i for i in range(adj.shape[0])])
    j = np.array([idx])    
    x, y = np.meshgrid(i, j, indexing='xy')
    adj[x,y] = 0 
    x, y = np.meshgrid(j, i, indexing='xy')
    adj[x,y] = 0
    
    return adj

def dict_arr_to_list(d):
    for k, v in d.items():
        if isinstance(v, dict):
            dict_arr_to_list(v)
        else:
            if type(v) == type(np.array(0)):
                v = v.tolist()
            d.update({k: v})
    return d

def dict_list_to_arr(d,skip=[]):
    for k, v in d.items():
        
        if any([k == x for x in skip]):
            continue    

        if isinstance(v, dict):
            dict_list_to_arr(v,skip=skip)

        else:
            if type(v) == type([]):
                v = np.array(v)
            d.update({k: v})

    return d
 
# compute the rotation matrix given theta in radians and axes
def compute_rotation_matrix(theta, axis):

    assert axis == "x" or axis == "y" or axis == "z"
    
    # form n 3x3 identity arrays
    n = theta.shape[0] if type(theta) == type(np.array(1)) else 1
    r = np.zeros((n,3,3),dtype=np.float32)
    r[:,0,0] = 1
    r[:,1,1] = 1
    r[:,2,2] = 1

    if axis == "x":
        #r = np.array([[1, 0,              0],
        #              [0, np.cos(theta), -np.sin(theta)],
        #              [0, np.sin(theta),  np.cos(theta)]])
        r[:,1,1] =  np.cos(theta)
        r[:,1,2] = -np.sin(theta)
        r[:,2,1] =  np.sin(theta)
        r[:,2,2] =  np.cos(theta)
                     
    if axis == "y":
        #r = np.array([[ np.cos(theta), 0,  np.sin(theta)],
        #              [ 0,             1,  0],
        #              [-np.sin(theta), 0,  np.cos(theta)]])
        r[:,0,0] =  np.cos(theta)
        r[:,0,2] =  np.sin(theta)
        r[:,2,0] = -np.sin(theta)
        r[:,2,2] =  np.cos(theta)

    if axis == "z":
        #r = np.array([[np.cos(theta), -np.sin(theta), 0],
        #              [np.sin(theta),  np.cos(theta), 0],
        #              [0,              0,             1]])
        r[:,0,0] =  np.cos(theta)
        r[:,0,1] = -np.sin(theta)
        r[:,1,0] =  np.sin(theta)
        r[:,1,1] =  np.cos(theta)
    
    return r
 
"""
# compute the rotation matrix given theta in radians and axes
def compute_rotation_matrix(theta, axis):

    assert axis == "x" or axis == "y" or axis == "z"
    
    # form n 3x3 identity arrays
    #n = theta.shape[0] if type(theta) == type(np.array(1)) else 1
    #r = np.zeros((n,3,3),dtype=np.float32)
    #r[:,0,0] = 1
    #r[:,1,1] = 1
    #r[:,2,2] = 1

    if axis == "x":
        r = np.array([[1, 0,              0],
                      [0, np.cos(theta), -np.sin(theta)],
                      [0, np.sin(theta),  np.cos(theta)]])
        #r[:,1,1] =  np.cos(theta)
        #r[:,1,2] = -np.sin(theta)
        #r[:,2,1] =  np.sin(theta)
        #r[:,2,2] =  np.cos(theta)
                     
    if axis == "y":
        r = np.array([[ np.cos(theta), 0,  np.sin(theta)],
                      [ 0,             1,  0],
                      [-np.sin(theta), 0,  np.cos(theta)]])
        #r[:,0,0] =  np.cos(theta)
        #r[:,0,2] =  np.sin(theta)
        #r[:,2,0] = -np.sin(theta)
        #r[:,2,2] =  np.cos(theta)

    if axis == "z":
        r = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta),  np.cos(theta), 0],
                      [0,              0,             1]])
        #r[:,0,0] =  np.cos(theta)
        #r[:,0,1] = -np.sin(theta)
        #r[:,1,0] =  np.sin(theta)
        #r[:,1,1] =  np.cos(theta)
    
    return r
"""