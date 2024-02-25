import numpy as np
from scipy import interpolate
from itertools import groupby

from collections.abc import Iterable

def interpolate(x1, x2, num_points):
        
    # compute length
    length = x2 - x1
    
    # initialize entry
    x = np.zeros(shape=[num_points]+list(x1.shape))
    x[0]  = x1
    x[-1] = x2
    
    # interpolate
    for i in range(1,num_points-1):
        x[i] = x1 + length * float(i)/float(num_points-1)
        #print(x[i])
        
    return x

def mean(data):
    return np.mean(list(flatten(data)))

def flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x
     
def object_aggregator_vis(true_obj_xyz, pred_obj_xyz, prob, inp_length, out_length):

    #print(true_obj_xyz.shape)  #[num_obj, inp+out length, 4, 3]
    #print(pred_obj_xyz.shape)  #[num_obj, inp+out length, 4, 3]
    #print(prob.shape)          #[num_obj, inp+out length]
    
    # process input
    for i in range(inp_length + out_length):
        pred_obj_xyz[:,i] = prob[:,i] * pred_obj_xyz[:,i] + (1 - prob[:,i]) * true_obj_xyz[:,i]
        
    ## process output           
    ## compute for t0
    #pred_obj_xyz[:,inp_length] = prob[:,inp_length] * pred_obj_xyz[:,inp_length] + (1 - prob[:,inp_length]) * true_obj_xyz[:,inp_length]
    ## compute for t1 onwards
    #for i in range(inp_length+1,inp_length+out_length):
    #    pred_obj_xyz[:,i] = prob[:,i] * pred_obj_xyz[:,i] + (1 - prob[:,i]) * pred_obj_xyz[:,i-1]
    
    return pred_obj_xyz
     
def object_aggregator(true_obj_xyz, pred_obj_xyz, prob, inp_length, out_length):

    #print(true_obj_xyz.shape)  #[num_obj, inp+out length, 4, 3]
    #print(pred_obj_xyz.shape)  #[num_obj, inp+out length, 4, 3]
    #print(prob.shape)          #[num_obj, inp+out length]
    
    # process input
    for i in range(inp_length):
        pred_obj_xyz[:,i] = prob[:,i] * pred_obj_xyz[:,i] + (1 - prob[:,i]) * true_obj_xyz[:,i]
        
    # process output           
    # compute for t0
    pred_obj_xyz[:,inp_length] = prob[:,inp_length] * pred_obj_xyz[:,inp_length] + (1 - prob[:,inp_length]) * true_obj_xyz[:,inp_length]
    # compute for t1 onwards
    for i in range(inp_length+1,inp_length+out_length):
        pred_obj_xyz[:,i] = prob[:,i] * pred_obj_xyz[:,i] + (1 - prob[:,i]) * pred_obj_xyz[:,i-1]
    
    return pred_obj_xyz
    
def one_hot(labels, max_label=None):

    one_hot_labels = np.zeros((labels.size, labels.max()+1)) if max_label is None else np.zeros((labels.size, max_label))
    one_hot_labels[np.arange(labels.size),labels] = 1
    
    return one_hot_labels

def decoder_object_aggregator(true_obj_xyz, pred_obj_xyz, prob):

    #print(pred_obj_xyz.shape)  #[num_objects, inp+out length, 4, 3]
    #print(prob.shape)          #[2, out length, obj_body_padded_length]
    num_objects, inp_out_length, _, _ = pred_obj_xyz.shape
    _, out_length, _ = prob.shape
    
    # https://www.includehelp.com/python/how-to-change-max-in-each-row-to-1-all-other-numbers-to-0-in-numpy-array.aspx
    max_prob = prob.max(axis=2).reshape(prob.shape[0], prob.shape[1], 1)
    prob = np.where(prob == max_prob, 1, 0)
    
    # remove padding
    prob = prob[:,:,:num_objects]       # [2, out_length, num_objects]
    prob = np.transpose(prob,[2,1,0])   # [num_objects, out_length, 2]
    
    # sum
    prob = np.sum(prob,axis=-1)         # [num_objects, out_length]
    prob[prob > 1] = 1
    prob = prob[:,:,None,None]
    
    # compute for t0
    pred_obj_xyz[:,10] = prob[:,0] * pred_obj_xyz[:,10] + (1 - prob[:,0]) * true_obj_xyz[:,10]
    # compute for t1 onwards
    for i in range(11,30):
        pred_obj_xyz[:,i] = prob[:,i-10] * pred_obj_xyz[:,i] + (1 - prob[:,i-10]) * pred_obj_xyz[:,i-1]
       
    return pred_obj_xyz

def split_list(data, sep):
    
    #https://stackoverflow.com/questions/72426671/how-can-i-split-my-list-into-a-list-of-lists-given-a-condition
    #sep  = str(sep)
    #data = [str(x) for x in data]
    
    return [list(group) for key, group in groupby(data, key=lambda x: x == sep) if not key]  

def custom_sort(data):
    
    order = ["Idle","Approach","Move","Hold","Place","Retreat"]
    sorted_data = [x for x in order if x in data]
    sorted_data = sorted_data + [x for x in data if x not in order]
    return sorted_data

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

# transform pos to object frame
def transform_frame(pos, rot, inverse):

    transformed_pos = []
    for pos_t,rot_t in zip(pos,rot):

        # rotation matrix to transform the origin coordinate system
        # to the object's coordinate system
        rx = compute_rotation_matrix(rot_t[0],"x")
        ry = compute_rotation_matrix(rot_t[1],"y")
        rz = compute_rotation_matrix(rot_t[2],"z")
        r = rz @ ry @ rx
        
        # rotation matrix to view the object position wrt 
        # to the object's coordinate system
        r = r.T if inverse == 0 else r
        
        # transform object pos to view it wrt 
        # to the object's coordinate system
        pos_t = r @ np.expand_dims(pos_t,1)
        transformed_pos.append(pos_t)

    transformed_pos = np.squeeze(np.stack(transformed_pos)) # [t, 3]
    return transformed_pos

# get the key frame given the segmentation timesteps and input frame
def get_key_frame_row(segmentation_timesteps, inp_frame):

    row = np.where(np.logical_and(segmentation_timesteps[:,0] <= inp_frame, segmentation_timesteps[:,1] >= inp_frame))[0]
    assert row.shape[0] == 1
    return row[0]
 
# concatenates the data along axis 
def concat(data1,data2,axis):
    
    data1 = np.array(data1)
    data2 = np.array(data2)
        
    data1 = np.expand_dims(data1,axis)
    return np.concatenate((data1,data2),axis)

# merges the kept and lost joints    
def merge(lost_data, kept_data, dimensions_to_lose):
    
    dimensions_to_lose.sort()
    print(lost_data.shape, kept_data.shape, dimensions_to_lose, len(dimensions_to_lose))
    if len(dimensions_to_lose) == 0:
        return kept_data
    
    #if max(dimensions_to_lose_bool) == 1:
    #    # convert from [0, 0, 1, 1, ...] to [2, 3, ...]
    #    dimensions_to_lose = np.array([i for i in range(len(dimensions_to_lose_bool))])
    #    dimensions_to_lose = [x for x,y in zip(dimensions_to_lose,dimensions_to_lose_bool) if y == 1]
    #else:
    #    dimensions_to_lose = dimensions_to_lose_bool
        
    # insert data back in
    for i,dim in enumerate(dimensions_to_lose):
        kept_data = np.insert(kept_data,int(dim),lost_data[:,i],1)
        
    return kept_data

# unprocesses the pose
# remember to inverse the order done in kit.py which was
# 1. subtract table_center
# 2. subtract reference_object_center
# 3. scale
def unprocess_pose(data, table_center, scale):

    # unscale
    data = data / scale
        
    data = data + table_center if table_center is not None else data
    
    """# center
    #if reference_object_center is not None:
    #    data = data # + reference_object_center
    if table_center is not None:
        if len(data.shape) == 3:
            data = data + np.expand_dims(table_center,1) if table_center is not None else data
        elif len(data.shape) == 2:
            data = data + table_center
        else:
            print("New shape")
            sys.exit()"""        
    return data
    
# unprocesses the object
# remember to inverse the order done in kit.py which was
# 1. subtract table_pos
# 3. scale
def unprocess_obj(pos, table_pos, scale):
    
    if pos is None:
        return None
    
    # pos [n, t, 3]
    # xyz [n, t, num_markers, 3]
        
    #if len(pos.shape) == 4:
    #    table_pos = np.expand_dims(table_pos,1)
        
    # ================ #
    # process position #
    # ================ #
    
    # scale
    pos = pos / scale
    
    # center
    #pos = pos + reference_object_center# if reference_object_center is not None else pos
    pos = pos + table_pos if table_pos is not None else pos
            
    return pos