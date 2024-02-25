import torch
from itertools import groupby

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

def unstandardize_data(ref_pos, ref_rot, ref_dim, tgt_xyz):
        
    # scale tgt_obj wrt ref_dim
    tgt_xyz = tgt_xyz * ref_dim # [pose_unpadded_length, 4, 3]
    
    # compute rotation matrix
    rx = compute_rotation_matrix(ref_rot[:,0], "x")    # [pose_unpadded_length, 3, 3]
    ry = compute_rotation_matrix(ref_rot[:,1], "y")    # [pose_unpadded_length, 3, 3]
    rz = compute_rotation_matrix(ref_rot[:,2], "z")    # [pose_unpadded_length, 3, 3]
    r  = rz @ ry @ rx                                  # [pose_unpadded_length, 3, 3]
    
    # project tgt obj xyz wrt ref
    tgt_xyz = r @ torch.permute(tgt_xyz,[0,2,1])    # [pose_unpadded_length, 4, 3]
    tgt_xyz = torch.permute(tgt_xyz,[0,2,1])        # [pose_unpadded_length, 4, 3]
    
    # center
    tgt_xyz = tgt_xyz + ref_pos[:,None,:]   # [pose_unpadded_length, 4, 3]
    #print(tgt_xyz)
    #print("================")
    #print(ref_pos)
    #print()
    
    return tgt_xyz

def interpolate(x1, x2, num_points):
    
    current_device = torch.cuda.current_device() if x1.get_device() != -1 else "cpu"
    
    # compute length
    length = x2 - x1
    
    # initialize entry
    x = torch.zeros(size=[num_points]+list(x1.shape)).to(device=current_device)
    x[0]  = x1
    x[-1] = x2
    
    # interpolate
    for i in range(1,num_points-1):
        x[i] = x1 + length * float(i)/float(num_points-1)
        #print(x[i])
        
    return x

def masked_sum(data1, data2, idxs):

    # data1 [batch, inp_length, 53, 3]
    # data2 [batch, inp_length, 8,  3]
    # idxs  [batch, 8]

    data1_clone = torch.clone(data1)
    data2_clone = torch.clone(data2)
    batch_size = idxs.shape[0]
    for i in range(batch_size):
        
        # data1_clone[i,:,idxs_i] # [inp_length, 8, 3]
        # data2_clone[i,:]        # [inp_length, 8, 3]
        # idxs_i            # [8]
        
        idxs_i = idxs[i]
        data1_clone[i,:,idxs_i] = data1_clone[i,:,idxs_i] + data2_clone[i,:]
    return data1_clone

#def split_list(data, sep):
    
    #sep  = str(sep)
    #data = [str(x) for x in data]
    
#    return [list(group) for key, group in groupby(data, key=lambda x: x == sep) if not key]    

# scale center of data
def scale(obj, t0, scale):
    
    # Wrong to scale it down directly. Read kit_mocap   
    # Should instead subtract by t0 before scaling down
    #obj_center         = torch.mean(obj,dim=0)
    #scaled_obj_center  = obj_center / scale
    #difference         = scaled_obj_center - obj_center
    #obj                = obj - difference
        
    # should be right
    obj_center                  = torch.mean(obj,dim=0)
    obj_center_minus_t0         = obj_center - t0
    scaled_obj_center_minus_t0  = obj_center_minus_t0 / scale
    difference                  = obj_center_minus_t0 - scaled_obj_center_minus_t0
    obj                         = obj - difference
    return obj

def reform_data(inp_data, out_data, inp_frame, key_frame):
    
    inp  = inp_data[inp_frame:inp_frame+1].clone()
    pred = out_data[0:key_frame-1].clone()                    
    zero = inp_data[key_frame:].clone()  

    return torch.cat([inp,pred,zero])

def reform_obj(inp_data, out_data, key_frame):
    
    # dim 0 = time
    # dim 1 = objects
    
    inp  = inp_data[:,key_frame:].clone()
    pred = out_data[:,:key_frame].clone()
    return torch.cat([pred,inp],dim=1)

def detach(adj, idx):
    
    i = torch.Tensor([i for i in range(adj.shape[0])]).long()
    j = torch.Tensor([idx]).long()    
    x, y = torch.meshgrid(i, j, indexing='xy')
    adj[x,y] = 0 
    x, y = torch.meshgrid(j, i, indexing='xy')
    adj[x,y] = 0
    
    return adj

#def reverse_data(data, un

def zero_pad(data_i, key_frame):

    pred_data_i = data_i[:key_frame].clone()
    zero_data_i = torch.zeros(size=data_i[key_frame:].shape).to(device=torch.cuda.current_device())
    pred_data_i = torch.cat([pred_data_i,zero_data_i])
    
    return pred_data_i

def pad(data, pad_length):

    current_device = torch.cuda.current_device() if data.get_device() != -1 else "cpu"
    
    unpadded_length = data.shape[0]
    if pad_length < unpadded_length:
        print("pad_length too short !")
        print("Pad Length = ", pad_length)
        print("Unpadded Sequence Length =", unpadded_length)
        sys.exit()
        
    new_shape = [pad_length] + list(data.shape[1:])
    new_shape = tuple(new_shape)
    data_padded = torch.zeros(new_shape).to(device=current_device)
    data_padded[:unpadded_length] = data
    
    assert torch.equal(data_padded[:unpadded_length],data)
    assert torch.count_nonzero(data_padded[unpadded_length:]) == 0
    
    return data_padded

def pad_variable_lengths(data, unpadded_length, padded_length):
        
    # initialize padded data
    new_shape = tuple([len(data), padded_length, data[-1].shape[-1]])
    padded_data = torch.zeros(size=new_shape)
    
    #print(padded_data.shape)
    
    # fill up padded_data
    for i in range(len(data)):
        #print(data[i].shape, unpadded_length[i])
        padded_data[i,:unpadded_length[i]] = data[i]
    current_device = torch.cuda.current_device() if data[0].get_device() != -1 else "cpu"
    padded_data = padded_data.to(device=current_device)
    return padded_data
    
"""# pad data
def pad(data, pad_length, return_unpadded_length=0):

    #print(data.shape)
    # data must be [t, ...]
    
    unpadded_length = data.shape[0]
    if pad_length < unpadded_length:
        print("pad_length too short !")
        print("Pad Length = ", pad_length)
        print("Unpadded Sequence Length =", unpadded_length)
        sys.exit()

    new_shape = [pad_length] + list(data.shape[1:])
    new_shape = tuple(new_shape)
    data_padded = np.zeros(shape=new_shape)
    data_padded[:unpadded_length] = data    
    
    assert np.array_equal(data_padded[:unpadded_length],data)
    assert np.all(data_padded[unpadded_length:] == 0)    
    
    if return_unpadded_length:
        return unpadded_length, data_padded.astype(np.float32) 
    return data_padded.astype(np.float32)"""

def dense_to_sparse(adj):
    r"""Converts a dense adjacency matrix to a sparse adjacency matrix defined
    by edge indices and edge attributes.
    Args:
        adj (Tensor): The dense adjacency matrix.
     :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    assert adj.dim() >= 2 and adj.dim() <= 3
    assert adj.size(-1) == adj.size(-2)

    index = adj.nonzero(as_tuple=True)

    if len(index) == 3:
        batch = index[0] * adj.size(-1)
        index = (batch + index[1], batch + index[2])

    return torch.stack(index, dim=0)

class return_1:
    def __init__(self, *args):
        return         
    
    def __call__(self, data):
        return torch.tensor(1, dtype=data.dtype, device=data.device)
    
    #def forward(self, data):
    #    return torch.tensor(1, dtype=data.type, device=data.device)

# compute the rotation matrix given thetas in radians and axes
def compute_rotation_matrix(theta, axis):

    assert axis == "x" or axis == "y" or axis == "z"
    
    # form n 3x3 identity arrays
    #n = theta.shape[0] if type(theta) == type(np.array(1)) else 1
    #r = np.zeros((n,3,3),dtype=np.float32)
    #r[:,0,0] = 1
    #r[:,1,1] = 1
    #r[:,2,2] = 1
        
    r = torch.zeros(size=[theta.shape[0],3,3], dtype=theta.dtype, device=theta.device) # [len(theta), 3, 3]

    if axis == "x":
        #r = np.array([[1, 0,              0],
        #              [0, np.cos(theta), -np.sin(theta)],
        #              [0, np.sin(theta),  np.cos(theta)]])
        r[:,0,0] =  1
        r[:,1,1] =  torch.cos(theta)
        r[:,1,2] = -torch.sin(theta)
        r[:,2,1] =  torch.sin(theta)
        r[:,2,2] =  torch.cos(theta)
                     
    if axis == "y":
        #r = np.array([[ np.cos(theta), 0,  np.sin(theta)],
        #              [ 0,             1,  0],
        #              [-np.sin(theta), 0,  np.cos(theta)]])
        r[:,1,1] =  1
        r[:,0,0] =  torch.cos(theta)
        r[:,0,2] =  torch.sin(theta)
        r[:,2,0] = -torch.sin(theta)
        r[:,2,2] =  torch.cos(theta)

    if axis == "z":
        #r = np.array([[np.cos(theta), -np.sin(theta), 0],
        #              [np.sin(theta),  np.cos(theta), 0],
        #              [0,              0,             1]])
        r[:,2,2] =  1
        r[:,0,0] =  torch.cos(theta)
        r[:,0,1] = -torch.sin(theta)
        r[:,1,0] =  torch.sin(theta)
        r[:,1,1] =  torch.cos(theta)
    
    return r
    
"""
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

# Set data type
def get_dtypes():
    bool_dtype  = torch.BoolTensor
    long_dtype  = torch.LongTensor
    float_dtype = torch.FloatTensor
    device      = "cpu"
    if torch.cuda.device_count() > 0:
        bool_dtype  = torch.cuda.BoolTensor
        long_dtype  = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
        device      = torch.cuda.current_device()
    return bool_dtype, long_dtype, float_dtype, device