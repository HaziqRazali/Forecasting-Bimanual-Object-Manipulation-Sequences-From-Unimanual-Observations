import re
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_kit_rgbd_data(self, data, section, double_batch_size=False, center=False):
    
    """
    function used to prepare ground truth data into a dictionary for use in the ranked prediction module and ensemble selector module
    """
    
    # always remember to reshape from [batch,2] to [batch*2] before doing any processing
    # always remember to reshape from [batch,2] to [batch*2] before reshaping the other dimensions for safety ?
    
    assert section == "inp" or section == "out"
    length      = self.inp_length if section == "inp" else self.out_length
    batch_size  = self.batch_size if section == "inp" else self.batch_size * 2
    num_obj_wrist_classes   = self.num_obj_wrist_classes
    obj_padded_length       = self.obj_padded_length

    # masked object idxs     
    # - set timesteps where the person is not handling to 0 instead of obj_padded_length since obj_padded_length is out of bounds; does not matter because we will mask the loss
    # - note that we are not picking the object with id = 0, rather it is the 0th (1st) object in the list, whatever its id may be
    handled_obj_idxs = torch.clone(data[section+"_handled_obj_idxs"])       # [batch, length] or [batch, 2, length]
    handled_obj_idxs[handled_obj_idxs == -1] = 0                            # [batch, length] or [batch, 2, length]
    handled_obj_idxs = torch.reshape(handled_obj_idxs,[batch_size,length])  # [batch, length] or [batch* 2, length]

    # mask for loss computation
    # - 1 if person is handling something so we compute the loss
    # - 0 if person not handling anything so we do not compute the loss
    # - always set to 1 before 0
    mask = torch.clone(data[section+"_handled_obj_idxs"]) # [batch, length] or [batch, 2, length]
    mask[mask != -1] = 1
    mask[mask == -1] = 0
    mask = torch.reshape(mask,[batch_size,-1])

    """
    object data
    """
    
    # obj_pos
    # - set the z coordinate to 0 as we use it to center the object and human along the horizontal plane
    #obj_pos = torch.clone(data[section+"_obj_pos"])    # [batch, length, obj_padded_length, obj_dim]
    #obj_pos[:,:,:,-1] = 0                              # [batch, length, obj_padded_length, obj_dim]
    obj_pos = torch.clone(data[section+"_obj_xyz"])     # [batch, length, obj_padded_length, num_markers, obj_dim]
    obj_pos = torch.mean(obj_pos,axis=3)                # [batch, length, obj_padded_length, obj_dim]
    if obj_pos.shape[-1] == 3:
        obj_pos[:,:,:,-1] = 0                           # [batch, length, obj_padded_length, obj_dim] set z axis to 0 so we dont subtract by it later
    elif obj_pos.shape[-1] == 2:
        obj_pos[:,:,:,-1] = 0                           # [batch, length, obj_padded_length, obj_dim] set y axis to 0 so we dont subtract by it later
    else:
        print("Unknown object dimension in get_data.py")
        sys.exit()
        
    """
    obj_pos = obj_pos[:,None,:,:,:]                                                     # [batch, 1, length, obj_padded_length, 3]
    obj_pos = obj_pos.repeat(1,1,1,1,1) if dbs == False else obj_pos.repeat(1,2,1,1,1)  # [batch, 1, length, obj_padded_length, 3] or [batch, 2, length, obj_padded_length, 3]
    """
    
    # handled_obj_pos
    handled_obj_pos = torch.clone(obj_pos)                                                                          # [batch, length, obj_padded_length, obj_dim]
    handled_obj_pos_shape = handled_obj_pos.shape
    idx = handled_obj_idxs.unsqueeze(-1).unsqueeze(-1)                                                              # [batch, length, 1, 1]                        or [batch*2, length, 1, 1]
    idx = idx.repeat(1,1,1,handled_obj_pos_shape[3])                                                                # [batch, length, 1, obj_dim]                  or [batch*2, length, 1, 1]
    handled_obj_pos = handled_obj_pos[:,None,:,:,:]                                                                 # [batch, 1, length, obj_padded_length, obj_dim]
    handled_obj_pos = handled_obj_pos.repeat(1,1,1,1,1) if section == "inp" else handled_obj_pos.repeat(1,2,1,1,1)  # [batch, 1, length, obj_padded_length, obj_dim] or [batch, 2, length, obj_padded_length, obj_dim]
    handled_obj_pos = torch.reshape(handled_obj_pos,[batch_size,length,obj_padded_length,handled_obj_pos_shape[3]]) # [batch, length, obj_padded_length, obj_dim]    or [batch* 2, length, obj_padded_length, obj_dim]
    handled_obj_pos = torch.gather(handled_obj_pos,2,idx)                                                           # [batch, length,                 1, obj_dim]    or [batch* 2, length,                 1, obj_dim]
        
    # obj_xyz
    # - never center since it is only being used for the temporal decoder
    obj_xyz = torch.clone(data[section+"_obj_xyz"])                                                                  # [batch. length, obj_padded_length, 4, 3]
    obj_xyz = torch.reshape(obj_xyz,[self.batch_size,length,obj_padded_length,-1])                                   # [batch, length, obj_padded_length, 12]
    
    # masked_obj_xyz
    # - do not center for the temporal encoder
    # - skip if "out"
    if section == "inp":
        masked_obj_xyz = torch.clone(data[section+"_masked_obj_xyz"])                           # [batch, length, obj_padded_length, 4, 3] or
        masked_obj_xyz = torch.reshape(masked_obj_xyz,[batch_size,length,obj_padded_length,-1]) # [batch, length, obj_padded_length, 12]  or
    else:
        masked_obj_xyz = None
        
    """# verify
    idx = inp_handled_obj_idxs.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)                                        # [batch, inp_length, 1,                 1, 1]
    idx = idx.repeat(1,1,1,4,3)                                                                                 # [batch, inp_length, 1,                 4, 3]
    masked_obj_xyz = torch.reshape(masked_obj_xyz,[self.batch_size,self.inp_length,self.obj_padded_length,4,3]) # [batch, inp_length, obj_padded_length, 4, 3]
    masked_obj_xyz = torch.squeeze(torch.gather(masked_obj_xyz,2,idx))                                          # [batch, inp_length, 1,                 4, 3]
    for x,y in zip(inp_handled_obj_idxs,masked_obj_xyz):
        print(x)
        print(y)
        input()
    sys.exit()"""
    
    # handled_obj_xyz
    # - center wrt handled_obj_pos for grabnet
    # - Why do I do this ? Why dont I simply compute grabnet for every object then mask out the loss later ?
    #   - because I only need to compute the loss for the true object during training
    #   - and because during testing I will be using a separate code that processes obj_xyz and human
    handled_obj_xyz = torch.clone(data[section+"_obj_xyz"])                                                                                     # [batch, length, obj_padded_length, num_markers, obj_dim]
    handled_obj_xyz_shape = handled_obj_xyz.shape
    idx = handled_obj_idxs.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)                                                                            # [batch, length, 1, 1, 1]                  or [batch* 2, length, 1, 1, 1]
    idx = idx.repeat(1,1,1,handled_obj_xyz_shape[3],handled_obj_xyz_shape[4])                                                                   # [batch, length, 1, num_markers, obj_dim] or [batch* 2, length, 1, 1, 1]
    handled_obj_xyz = handled_obj_xyz[:,None,:,:,:,:]                                                                                           # [batch, 1, length, obj_padded_length, num_markers, obj_dim]
    handled_obj_xyz = handled_obj_xyz.repeat(1,1,1,1,1,1) if section == "inp" else handled_obj_xyz.repeat(1,2,1,1,1,1)                          # [batch, 1, length, obj_padded_length, num_markers, obj_dim] or [batch, 2, length, obj_padded_length, num_markers, obj_dim]
    handled_obj_xyz = torch.reshape(handled_obj_xyz,[batch_size,length,obj_padded_length,handled_obj_xyz_shape[3],handled_obj_xyz_shape[4]])    # [batch, length, obj_padded_length, num_markers, obj_dim]    or [batch* 2, length, obj_padded_length, num_markers, obj_dim]
    handled_obj_xyz = torch.gather(handled_obj_xyz,2,idx)[:,:,0]                                                                                # [batch, length,                    num_markers, obj_dim]    or [batch* 2, length,                    num_markers, obj_dim]
    handled_obj_xyz = handled_obj_xyz - handled_obj_pos if center == True else handled_obj_xyz                                                  # [batch, length,                    num_markers, obj_dim]    or [batch* 2, length,                    num_markers, obj_dim]
    handled_obj_xyz = torch.reshape(handled_obj_xyz,[batch_size,length,-1])                                                                     # [batch, length,                    num_markers* obj_dim]    or [batch* 2, length,                    num_markers* obj_dim]
        
    # obj_ohs
    obj_ohs = torch.clone(data["obj_ohs"])              # [batch,         obj_padded_length, num_obj_wrist_classes]
    obj_ohs = obj_ohs[:,None,:,:].repeat(1,length,1,1)  # [batch, length, obj_padded_length, num_obj_wrist_classes]
    
    # handled_obj_ohs
    idx = handled_obj_idxs.unsqueeze(-1).unsqueeze(-1)              # [batch, length, 1, 1]                     or [batch* 2, length, 1, 1]
    idx = idx.repeat(1,1,1,obj_ohs.shape[-1])                       # [batch, length, 1, num_obj_wrist_classes] or [batch* 2, length, 1, num_obj_wrist_classes]
    handled_obj_ohs = torch.clone(obj_ohs)                                                                          # [batch, length, obj_padded_length, num_obj_wrist_classes]
    handled_obj_ohs = handled_obj_ohs[:,None,:,:]                                                                   # [batch, 1, length, obj_padded_length, num_obj_wrist_classes]
    handled_obj_ohs = handled_obj_ohs.repeat(1,1,1,1,1) if section == "inp" else handled_obj_ohs.repeat(1,2,1,1,1)  # [batch, 1, length, obj_padded_length, num_obj_wrist_classes]      or [batch, 2, length, obj_padded_length, num_obj_wrist_classes]
    handled_obj_ohs = torch.reshape(handled_obj_ohs,[batch_size,length,obj_padded_length,num_obj_wrist_classes])          # [batch, length, obj_padded_length, num_obj_wrist_classes]   or [batch* 2, length, obj_padded_length, num_obj_wrist_classes]
    handled_obj_ohs = torch.squeeze(torch.gather(handled_obj_ohs,2,idx))                                            # [batch, length,                    num_obj_wrist_classes]         or [batch* 2, length,                    num_obj_wrist_classes]

    """
    human data
    """
        
    # masked_xyz
    # - do not center if sending to the temporal encoder
    # - center wrt the object being handled if sending to grabnet
    xyz_mask_idxs = torch.clone(data[section+"_xyz_mask_idxs"])                         # [batch, -1] or [batch, 2, -1]
    xyz_mask_idxs = torch.reshape(xyz_mask_idxs,[batch_size,xyz_mask_idxs.shape[-1]])   # [batch, -1] or [batch* 2, -1]
    masked_xyz = torch.clone(data[section+"_masked_xyz"])                                                   # [batch, length, num_joints, body_dim] or [batch, 2, length, num_joints, body_dim]
    masked_xyz_shape = masked_xyz.shape
    masked_xyz = torch.reshape(masked_xyz,[batch_size,length,masked_xyz_shape[-2],masked_xyz_shape[-1]])    # [batch, length, num_joints, body_dim] or [batch* 2, length, num_joints, body_dim]
    if center == True:
                
        # center the human mocap markers
        masked_xyz = masked_xyz - handled_obj_pos # [batch, length, 53, 3] or [batch* 2, length, 53, 3]
        
        # make sure the missing joints that are 0 remain 0
        for i,xyz_mask_idxs_i in enumerate(xyz_mask_idxs):
            masked_xyz[i,:,xyz_mask_idxs_i,:] = 0
    masked_xyz = torch.reshape(masked_xyz,[batch_size,length,-1]) # [batch, length, 159]
    
    # masked_finger
    masked_finger = torch.clone(data[section+"_masked_finger"])                                                         # [batch, length, 2, finger_dim] or [batch, 2, length, 2, finger_dim]
    masked_finger_shape = masked_finger.shape
    masked_finger = torch.reshape(masked_finger,[batch_size,length,masked_finger_shape[-2],masked_finger_shape[-1]])    # [batch, length, 2, finger_dim] or [batch* 2, length, 2, finger_dim]
    masked_finger = torch.reshape(masked_finger,[batch_size,length,-1])                                                 # [batch, length, 2* finger_dim]
    
    # xyz
    # - never center since it is only being used for the temporal decoder
    xyz     = torch.clone(data[section+"_xyz"])                 # [batch, inp_length, num_joints, body_dim]
    xyz     = torch.reshape(xyz,[self.batch_size,length,-1])    # [batch, inp_length, num_joints* body_dim]
    finger  = torch.clone(data[section+"_finger"])              # [batch, inp_length, 2, finger_dim]
    finger  = torch.reshape(finger,[self.batch_size,length,-1]) # [batch, inp_length, 2* finger_dim]
    
    # finger_mask_idxs
    finger_mask_idxs = data[section+"_finger_mask_idxs"] # [batch, -1]
    finger_mask_idxs = torch.reshape(finger_mask_idxs,[batch_size,-1])
    
    # wrist data
    masked_wrist_xyz    = data[section+"_masked_wrist_xyz"]
    wrist_xyz           = data[section+"_wrist_xyz"]
    wrist_ohs           = data["wrist_ohs"]
    wrist_ohs           = wrist_ohs[:,None,:,:].repeat(1,length,1,1)
    
    # return_data             
    return_data = {# object data
                    "obj_xyz":obj_xyz, "masked_obj_xyz":masked_obj_xyz, "obj_ohs":obj_ohs,
                    "handled_obj_xyz":handled_obj_xyz, "handled_obj_pos":handled_obj_pos, "handled_obj_ohs":handled_obj_ohs, "handled_obj_idxs":handled_obj_idxs, 
                   # human data
                    "xyz":xyz, "finger":finger,
                    "masked_xyz":masked_xyz, "masked_finger":masked_finger, "finger_mask_idxs":finger_mask_idxs,
                    "mask":mask,
                   # wrist data
                    "masked_wrist_xyz":masked_wrist_xyz, "wrist_xyz":wrist_xyz, "wrist_ohs":wrist_ohs}
    return return_data

def get_data(self, data, section, double_batch_size=False, center=False):
    
    """
    function used to prepare ground truth data into a dictionary for use in the ranked prediction module and ensemble selector module
    """
    
    # always remember to reshape from [batch,2] to [batch*2] before doing any processing
    # always remember to reshape from [batch,2] to [batch*2] before reshaping the other dimensions for safety ?
    
    assert section == "inp" or section == "out"
    length      = self.inp_length if section == "inp" else self.out_length
    batch_size  = self.batch_size if section == "inp" else self.batch_size * 2
    num_obj_classes   = self.num_obj_classes
    obj_padded_length = self.obj_padded_length

    # masked object idxs     
    # - set timesteps where the person is not handling to 0 instead of obj_padded_length since obj_padded_length is out of bounds; does not matter because we will mask the loss
    # - note that we are not picking the object with id = 0, rather it is the 0th (1st) object in the list, whatever its id may be
    handled_obj_idxs = torch.clone(data[section+"_handled_obj_idxs"])       # [batch, length] or [batch, 2, length]
    handled_obj_idxs[handled_obj_idxs == -1] = 0                            # [batch, length] or [batch, 2, length]
    handled_obj_idxs = torch.reshape(handled_obj_idxs,[batch_size,length])  # [batch, length] or [batch* 2, length]

    # mask for loss computation
    # - 1 if person is handling something so we compute the loss
    # - 0 if person not handling anything so we do not compute the loss
    # - always set to 1 before 0
    mask = torch.clone(data[section+"_handled_obj_idxs"]) # [batch, length] or [batch, 2, length]
    mask[mask != -1] = 1
    mask[mask == -1] = 0
    mask = torch.reshape(mask,[batch_size,-1])

    """
    object data
    """
    
    # obj_pos
    # - set the z coordinate to 0 as we use it to center the object and human along the horizontal plane
    #obj_pos = torch.clone(data[section+"_obj_pos"])    # [batch, length, obj_padded_length, obj_dim]
    #obj_pos[:,:,:,-1] = 0                              # [batch, length, obj_padded_length, obj_dim]
    obj_pos = torch.clone(data[section+"_obj_xyz"])     # [batch, length, obj_padded_length, num_markers, obj_dim]
    obj_pos = torch.mean(obj_pos,axis=3)                # [batch, length, obj_padded_length, obj_dim]
    obj_pos_zc = torch.clone(obj_pos)                   # [batch, length, obj_padded_length, obj_dim] for real zero centering
    if obj_pos.shape[-1] == 3:
        obj_pos[:,:,:,-1] = 0                           # [batch, length, obj_padded_length, obj_dim] set z axis to 0 so we dont subtract by it later
    elif obj_pos.shape[-1] == 2:
        obj_pos[:,:,:,-1] = 0                           # [batch, length, obj_padded_length, obj_dim] set y axis to 0 so we dont subtract by it later
    else:
        print("Unknown object dimension in get_data.py")
        sys.exit()
        
    """
    obj_pos = obj_pos[:,None,:,:,:]                                                     # [batch, 1, length, obj_padded_length, 3]
    obj_pos = obj_pos.repeat(1,1,1,1,1) if dbs == False else obj_pos.repeat(1,2,1,1,1)  # [batch, 1, length, obj_padded_length, 3] or [batch, 2, length, obj_padded_length, 3]
    """
    
    # handled_obj_pos
    handled_obj_pos = torch.clone(obj_pos)                                                                          # [batch, length, obj_padded_length, obj_dim]
    handled_obj_pos_shape = handled_obj_pos.shape
    idx = handled_obj_idxs.unsqueeze(-1).unsqueeze(-1)                                                              # [batch, length, 1, 1]                        or [batch*2, length, 1, 1]
    idx = idx.repeat(1,1,1,handled_obj_pos_shape[3])                                                                # [batch, length, 1, obj_dim]                  or [batch*2, length, 1, 1]
    handled_obj_pos = handled_obj_pos[:,None,:,:,:]                                                                 # [batch, 1, length, obj_padded_length, obj_dim]
    handled_obj_pos = handled_obj_pos.repeat(1,1,1,1,1) if section == "inp" else handled_obj_pos.repeat(1,2,1,1,1)  # [batch, 1, length, obj_padded_length, obj_dim] or [batch, 2, length, obj_padded_length, obj_dim]
    handled_obj_pos = torch.reshape(handled_obj_pos,[batch_size,length,obj_padded_length,handled_obj_pos_shape[3]]) # [batch, length, obj_padded_length, obj_dim]    or [batch* 2, length, obj_padded_length, obj_dim]
    handled_obj_pos = torch.gather(handled_obj_pos,2,idx)                                                           # [batch, length,                 1, obj_dim]    or [batch* 2, length,                 1, obj_dim]
        
    # handled_obj_pos_zc
    handled_obj_pos_zc = torch.clone(obj_pos_zc)                                                                                # [batch, length, obj_padded_length, obj_dim]
    handled_obj_pos_zc_shape = handled_obj_pos_zc.shape
    idx = handled_obj_idxs.unsqueeze(-1).unsqueeze(-1)                                                                          # [batch, length, 1, 1]                        or [batch*2, length, 1, 1]
    idx = idx.repeat(1,1,1,handled_obj_pos_zc_shape[3])                                                                         # [batch, length, 1, obj_dim]                  or [batch*2, length, 1, 1]
    handled_obj_pos_zc = handled_obj_pos_zc[:,None,:,:,:]                                                                       # [batch, 1, length, obj_padded_length, obj_dim]
    handled_obj_pos_zc = handled_obj_pos_zc.repeat(1,1,1,1,1) if section == "inp" else handled_obj_pos_zc.repeat(1,2,1,1,1)     # [batch, 1, length, obj_padded_length, obj_dim] or [batch, 2, length, obj_padded_length, obj_dim]
    handled_obj_pos_zc = torch.reshape(handled_obj_pos_zc,[batch_size,length,obj_padded_length,handled_obj_pos_zc_shape[3]])    # [batch, length, obj_padded_length, obj_dim]    or [batch* 2, length, obj_padded_length, obj_dim]
    handled_obj_pos_zc = torch.gather(handled_obj_pos_zc,2,idx)                                                                 # [batch, length,                 1, obj_dim]    or [batch* 2, length,                 1, obj_dim]
    
    # obj_xyz
    # - never center since it is only being used for the temporal decoder
    obj_xyz = torch.clone(data[section+"_obj_xyz"])                                                                  # [batch. length, obj_padded_length, 4, 3]
    obj_xyz = torch.reshape(obj_xyz,[self.batch_size,length,obj_padded_length,-1])                                   # [batch, length, obj_padded_length, 12]
    
    # masked_obj_xyz
    # - do not center for the temporal encoder
    # - skip if "out"
    if section == "inp":
        masked_obj_xyz = torch.clone(data[section+"_masked_obj_xyz"])                           # [batch, length, obj_padded_length, 4, 3] or
        masked_obj_xyz = torch.reshape(masked_obj_xyz,[batch_size,length,obj_padded_length,-1]) # [batch, length, obj_padded_length, 12]  or
    else:
        masked_obj_xyz = None
        
    """# verify
    idx = inp_handled_obj_idxs.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)                                        # [batch, inp_length, 1,                 1, 1]
    idx = idx.repeat(1,1,1,4,3)                                                                                 # [batch, inp_length, 1,                 4, 3]
    masked_obj_xyz = torch.reshape(masked_obj_xyz,[self.batch_size,self.inp_length,self.obj_padded_length,4,3]) # [batch, inp_length, obj_padded_length, 4, 3]
    masked_obj_xyz = torch.squeeze(torch.gather(masked_obj_xyz,2,idx))                                          # [batch, inp_length, 1,                 4, 3]
    for x,y in zip(inp_handled_obj_idxs,masked_obj_xyz):
        print(x)
        print(y)
        input()
    sys.exit()"""
    
    # handled_obj_xyz
    # - center wrt handled_obj_pos for grabnet
    # - Why do I do this ? Why dont I simply compute grabnet for every object then mask out the loss later ?
    #   - because I only need to compute the loss for the true object during training
    #   - and because during testing I will be using a separate code that processes obj_xyz and human
    handled_obj_xyz = torch.clone(data[section+"_obj_xyz"])                                                                                     # [batch, length, obj_padded_length, num_markers, obj_dim]
    handled_obj_xyz_shape = handled_obj_xyz.shape
    idx = handled_obj_idxs.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)                                                                            # [batch, length, 1, 1, 1]                  or [batch* 2, length, 1, 1, 1]
    idx = idx.repeat(1,1,1,handled_obj_xyz_shape[3],handled_obj_xyz_shape[4])                                                                   # [batch, length, 1, num_markers, obj_dim] or [batch* 2, length, 1, 1, 1]
    handled_obj_xyz = handled_obj_xyz[:,None,:,:,:,:]                                                                                           # [batch, 1, length, obj_padded_length, num_markers, obj_dim]
    handled_obj_xyz = handled_obj_xyz.repeat(1,1,1,1,1,1) if section == "inp" else handled_obj_xyz.repeat(1,2,1,1,1,1)                          # [batch, 1, length, obj_padded_length, num_markers, obj_dim] or [batch, 2, length, obj_padded_length, num_markers, obj_dim]
    handled_obj_xyz = torch.reshape(handled_obj_xyz,[batch_size,length,obj_padded_length,handled_obj_xyz_shape[3],handled_obj_xyz_shape[4]])    # [batch, length, obj_padded_length, num_markers, obj_dim]    or [batch* 2, length, obj_padded_length, num_markers, obj_dim]
    handled_obj_xyz = torch.gather(handled_obj_xyz,2,idx)[:,:,0]                                                                                # [batch, length,                    num_markers, obj_dim]    or [batch* 2, length,                    num_markers, obj_dim]
    handled_obj_xyz_zc = torch.clone(handled_obj_xyz)                                                                                           # [batch, length,                    num_markers, obj_dim]    or [batch* 2, length,                    num_markers, obj_dim] really zero centered
    handled_obj_xyz = handled_obj_xyz - handled_obj_pos if center == True else handled_obj_xyz                                                  # [batch, length,                    num_markers, obj_dim]    or [batch* 2, length,                    num_markers, obj_dim]
    handled_obj_xyz = torch.reshape(handled_obj_xyz,[batch_size,length,-1])                                                                     # [batch, length,                    num_markers* obj_dim]    or [batch* 2, length,                    num_markers* obj_dim]
    handled_obj_xyz_zc = handled_obj_xyz_zc - handled_obj_pos_zc
    handled_obj_xyz_zc = torch.reshape(handled_obj_xyz_zc,[batch_size,length,-1])                                                               # [batch, length,                    num_markers* obj_dim]    or [batch* 2, length,                    num_markers* obj_dim]
        
    # obj_ohs
    obj_ohs = torch.clone(data["obj_ohs"])              # [batch,         obj_padded_length, num_obj_classes]
    obj_ohs = obj_ohs[:,None,:,:].repeat(1,length,1,1)  # [batch, length, obj_padded_length, num_obj_classes]
    
    # handled_obj_ohs
    idx = handled_obj_idxs.unsqueeze(-1).unsqueeze(-1)              # [batch, length, 1, 1]               or [batch* 2, length, 1, 1]
    idx = idx.repeat(1,1,1,obj_ohs.shape[-1])                       # [batch, length, 1, num_obj_classes] or [batch* 2, length, 1, num_obj_classes]
    handled_obj_ohs = torch.clone(obj_ohs)                                                                          # [batch, length, obj_padded_length, num_obj_classes]
    handled_obj_ohs = handled_obj_ohs[:,None,:,:]                                                                   # [batch, 1, length, obj_padded_length, num_obj_classes]
    handled_obj_ohs = handled_obj_ohs.repeat(1,1,1,1,1) if section == "inp" else handled_obj_ohs.repeat(1,2,1,1,1)  # [batch, 1, length, obj_padded_length, num_obj_classes] or [batch, 2, length, obj_padded_length, num_obj_classes]
    handled_obj_ohs = torch.reshape(handled_obj_ohs,[batch_size,length,obj_padded_length,num_obj_classes])          # [batch, length, obj_padded_length, num_obj_classes]    or [batch* 2, length, obj_padded_length, num_obj_classes]
    handled_obj_ohs = torch.squeeze(torch.gather(handled_obj_ohs,2,idx))                                            # [batch, length,                    num_obj_classes]    or [batch* 2, length,                    num_obj_classes]

    """
    human data
    """
        
    # masked_xyz
    # - do not center if sending to the temporal encoder
    # - center wrt the object being handled if sending to grabnet
    xyz_mask_idxs = torch.clone(data[section+"_xyz_mask_idxs"])                         # [batch, -1] or [batch, 2, -1]
    xyz_mask_idxs = torch.reshape(xyz_mask_idxs,[batch_size,xyz_mask_idxs.shape[-1]])   # [batch, -1] or [batch* 2, -1]
    masked_xyz = torch.clone(data[section+"_masked_xyz"])                                                   # [batch, length, num_joints, body_dim] or [batch, 2, length, num_joints, body_dim]
    masked_xyz_shape = masked_xyz.shape
    masked_xyz = torch.reshape(masked_xyz,[batch_size,length,masked_xyz_shape[-2],masked_xyz_shape[-1]])    # [batch, length, num_joints, body_dim] or [batch* 2, length, num_joints, body_dim]
    if center == True:
                
        # center the human mocap markers
        masked_xyz = masked_xyz - handled_obj_pos # [batch, length, 53, 3] or [batch* 2, length, 53, 3]
        
        # make sure the missing joints that are 0 remain 0
        for i,xyz_mask_idxs_i in enumerate(xyz_mask_idxs):
            masked_xyz[i,:,xyz_mask_idxs_i,:] = 0
    masked_xyz = torch.reshape(masked_xyz,[batch_size,length,-1]) # [batch, length, 159]
    
    # masked_finger
    masked_finger = torch.clone(data[section+"_masked_finger"])                                                         # [batch, length, 2, finger_dim] or [batch, 2, length, 2, finger_dim]
    masked_finger_shape = masked_finger.shape
    masked_finger = torch.reshape(masked_finger,[batch_size,length,masked_finger_shape[-2],masked_finger_shape[-1]])    # [batch, length, 2, finger_dim] or [batch* 2, length, 2, finger_dim]
    masked_finger = torch.reshape(masked_finger,[batch_size,length,-1])                                                 # [batch, length, 2* finger_dim]
    
    # xyz
    # - never center since it is only being used for the temporal decoder
    xyz     = torch.clone(data[section+"_xyz"])                 # [batch, inp_length, num_joints, body_dim]
    xyz     = torch.reshape(xyz,[self.batch_size,length,-1])    # [batch, inp_length, num_joints* body_dim]
    finger  = torch.clone(data[section+"_finger"])              # [batch, inp_length, 2, finger_dim]
    finger  = torch.reshape(finger,[self.batch_size,length,-1]) # [batch, inp_length, 2* finger_dim]
    
    # finger_mask_idxs
    finger_mask_idxs = data[section+"_finger_mask_idxs"]                # [batch, -1]
    finger_mask_idxs = torch.reshape(finger_mask_idxs,[batch_size,-1])
    
    # wrist
    # - has not been centered
    # - this will simply be used to compute the loss
    wrist_xyz = torch.clone(data[section+"_xyz"])                           # [batch, inp_length, num_joints, body_dim]
    wrist_xyz = wrist_xyz[:,:,self.hand_xyz_dims]                           # [batch, inp_length, 10, body_dim]
    wrist_xyz = torch.reshape(wrist_xyz,[self.batch_size,length,2,5,-1])    # [batch, inp_length, 2, 5, body_dim]
    wrist_xyz = torch.permute(wrist_xyz,[0,2,1,3,4])                        # [batch, 2, inp_length, 5, body_dim]
    
    # return_data             
    return_data = {# object data
                    "obj_xyz":obj_xyz, "masked_obj_xyz":masked_obj_xyz, "obj_ohs":obj_ohs,
                    "handled_obj_xyz":handled_obj_xyz, "handled_obj_pos":handled_obj_pos, "handled_obj_ohs":handled_obj_ohs, "handled_obj_idxs":handled_obj_idxs, 
                    "handled_obj_xyz_zc":handled_obj_xyz_zc, "handled_obj_pos_zc":handled_obj_pos_zc,
                   # human data
                    "xyz":xyz, "finger":finger,
                    "masked_xyz":masked_xyz, "masked_finger":masked_finger, "finger_mask_idxs":finger_mask_idxs,
                    "wrist_xyz":wrist_xyz,
                    "mask":mask}
    return return_data
    
def prepare_pred_data_for_out_grab_net(self, pred_xyz, pred_handled_obj_xyz, handled_obj_ohs, length):
    
    num_body_joints = self.num_body_joints
    body_dim        = self.body_dim
    num_obj_markers = self.num_obj_markers
    obj_dim         = self.obj_dim
    num_hands       = self.num_hands
    finger_dim      = self.finger_dim
    
    """
    get masked_xyz
    """
    
    # create masked_xyz
    masked_xyz = torch.clone(pred_xyz)[:,None,:,:,:]                                    # [batch, 1, length, num_body_joints, body_dim]
    masked_xyz = masked_xyz.repeat(1,2,1,1,1)                                           # [batch, 2, length, num_body_joints, body_dim]
    for i,xyz_mask_idxs in enumerate([self.l_arm_mocap_idxs,self.r_arm_mocap_idxs]):
        masked_xyz[:,i,:,xyz_mask_idxs,:] = 0
    
    """
    get the left and right handled objects
    """

    #handled_obj_xyz = pred_handled_obj_xyz
    
    # mocap center
    handled_obj_pos = torch.mean(pred_handled_obj_xyz,axis=2,keepdim=True)                           # [batch* 2, length, 1, obj_dim]
    handled_obj_pos_zc = torch.clone(handled_obj_pos)                                           # [batch* 2, length, 1, obj_dim] # true centering
    # - set the z value to 0 for centering
    handled_obj_pos[:,:,:,-1] = 0
    handled_obj_pos    = torch.reshape(handled_obj_pos,[self.batch_size,2,length,1,obj_dim])       # [batch, 2, length, 1, obj_dim]
    handled_obj_pos_zc = torch.reshape(handled_obj_pos_zc,[self.batch_size,2,length,1,obj_dim])
      
    # handled obj ohs
    handled_obj_ohs = handled_obj_ohs
    
    """
    center masked_xyz and handled_obj_xyz wrt handled_obj_pos
    """
    
    # center masked_xyz wrt handled_obj_pos but keep masked mocap zero
    masked_xyz = masked_xyz - handled_obj_pos
    for i,xyz_mask_idxs in enumerate([self.l_arm_mocap_idxs,self.r_arm_mocap_idxs]):
        masked_xyz[:,i,:,xyz_mask_idxs,:] = 0
    masked_xyz = torch.reshape(masked_xyz,[self.batch_size*2,length,num_body_joints,body_dim])
    masked_xyz = torch.reshape(masked_xyz,[self.batch_size*2,length,-1])
        
    # center handled_obj_xyz wrt handled_obj_pos
    handled_obj_pos = torch.reshape(handled_obj_pos,[self.batch_size*2,length,1,obj_dim])
    handled_obj_xyz = pred_handled_obj_xyz - handled_obj_pos
    handled_obj_xyz = torch.reshape(handled_obj_xyz,[self.batch_size*2,length,-1])
    
    # center handled_obj_xyz_zc wrt handled_obj_pos_zc
    handled_obj_pos_zc = torch.reshape(handled_obj_pos_zc,[self.batch_size*2,length,1,obj_dim])
    handled_obj_xyz_zc = pred_handled_obj_xyz - handled_obj_pos_zc
    handled_obj_xyz_zc = torch.reshape(handled_obj_xyz_zc,[self.batch_size*2,length,-1])
    
    # finger mask idxs
    finger_mask_idxs = torch.Tensor([0,1]).type(torch.cuda.LongTensor)
    finger_mask_idxs = finger_mask_idxs[None,:].repeat(self.batch_size,1)
    finger_mask_idxs = torch.reshape(finger_mask_idxs,[self.batch_size*2,1])
    
    # return_data             
    return_data = {# object data
                    "handled_obj_xyz":handled_obj_xyz, "handled_obj_pos":handled_obj_pos, "handled_obj_ohs":handled_obj_ohs,
                    "handled_obj_xyz_zc":handled_obj_xyz_zc, "handled_obj_pos_zc":handled_obj_pos_zc,
                   # human data
                    "masked_xyz":masked_xyz, "finger_mask_idxs":finger_mask_idxs}
    return return_data

def agg_pred(self, pred_human, obj_ohs, lhand, rhand):

    length = self.inp_length
    
    # create masked_xyz
    masked_xyz = torch.clone(pred_human)[:,None,:,:,:]          # [batch, 1, length, 53, 3]
    masked_xyz = masked_xyz.repeat(1,2,1,1,1)                   # [batch, 2, length, 53, 3]
    for i,xyz_mask_idxs in enumerate([self.l_arm_mocap_idxs,self.r_arm_mocap_idxs]):
        masked_xyz[:,i,:,xyz_mask_idxs,:] = 0
    
    """
    get the left and right handled objects
    """

    # handled_obj
    handled_obj = torch.stack([lhand["pred_handled_obj"],rhand["pred_handled_obj"]],dim=1)  # [batch, 2, length, obj_body_padded_length]
    handled_obj = torch.reshape(handled_obj,[self.batch_size*2, length, -1])                # [batch* 2, length, obj_body_padded_length]
    handled_obj_idx = torch.argmax(handled_obj,dim=-1)                                      # [batch* 2, length]
        
    # mocap coordinates of the most probable object being handled
    # - set handled_obj_idx to 0 if no object being handled and we will mask it out later
    handled_obj_idx[handled_obj_idx == self.obj_padded_length] = 0                                    # [batch* 2, length]
    obj_xyz = torch.stack([lhand["pred_obj_xyz"],rhand["pred_obj_xyz"]],dim=1)                        # [batch, 2, length, obj_padded_length, 4, 3]
    obj_xyz = torch.reshape(obj_xyz,[self.batch_size*2, length, self.obj_padded_length, 4, 3])        # [batch* 2, length, obj_padded_length, 4, 3]
    handled_obj_xyz = torch.gather(obj_xyz,2,handled_obj_idx[:,:,None,None,None].repeat(1,1,1,4,3))   # [batch* 2, length, 1, 4, 3]
    handled_obj_xyz = torch.squeeze(handled_obj_xyz)                                                  # [batch* 2, length, 4, 3]
    
    # mocap center
    # - set the z value to 0 for centering
    handled_obj_pos = torch.mean(handled_obj_xyz,axis=2,keepdim=True)               # [batch* 2, length, 1, 3]
    handled_obj_pos[:,:,:,-1] = 0
    handled_obj_pos = torch.reshape(handled_obj_pos,[self.batch_size,2,length,1,3]) # [batch, 2, length, 1, 3]
      
    # handled obj ohs
    # obj_ohs                                                       [batch* 2,             obj_padded_length, num_obj_classes]
    # obj_ohs[:,None,:,:].repeat(1,length,1,1)                      [batch* 2, length, obj_padded_length, num_obj_classes] 
    # handled_obj                                                   [batch* 2, length]
    # handled_obj[:,:,None,None].repeat(1,1,1,self.num_obj_classes) [batch* 2, length, 1,               , num_obj_classes]   
    obj_ohs = obj_ohs[:,None,:,:].repeat(1,2,1,1)
    obj_ohs = torch.reshape(obj_ohs,[self.batch_size*2, self.obj_padded_length, -1])                                                                # [batch* 2,             obj_padded_length, num_obj_classes]
    handled_obj_ohs = torch.gather(obj_ohs[:,None,:,:].repeat(1,length,1,1),2,handled_obj_idx[:,:,None,None].repeat(1,1,1,self.num_obj_classes))    # [batch* 2, length, obj_padded_length, num_obj_classes]
    handled_obj_ohs = torch.squeeze(handled_obj_ohs)                                                                                                # [batch,    length,                    num_obj_classes]
    
    """
    center masked_xyz and handled_obj_xyz wrt handled_obj_pos
    """
    
    # center masked_xyz wrt handled_obj_pos but keep masked mocap zero
    masked_xyz = masked_xyz - handled_obj_pos
    for i,xyz_mask_idxs in enumerate([self.l_arm_mocap_idxs,self.r_arm_mocap_idxs]):
        masked_xyz[:,i,:,xyz_mask_idxs,:] = 0
    masked_xyz = torch.reshape(masked_xyz,[self.batch_size*2,length,53,3])
    masked_xyz = torch.reshape(masked_xyz,[self.batch_size*2,length,-1])
        
    # center handled_obj_xyz wrt handled_obj_pos
    handled_obj_pos = torch.reshape(handled_obj_pos,[self.batch_size*2,length,1,3])
    handled_obj_xyz = handled_obj_xyz - handled_obj_pos
    handled_obj_xyz = torch.reshape(handled_obj_xyz,[self.batch_size*2,length,-1])
    
    # finger mask idxs
    finger_mask_idxs = torch.Tensor([0,1]).type(torch.cuda.LongTensor)
    finger_mask_idxs = finger_mask_idxs[None,:].repeat(self.batch_size,1)
    finger_mask_idxs = torch.reshape(finger_mask_idxs,[self.batch_size*2,1])
    
    # return_data             
    return_data = {# object data
                    "handled_obj_xyz":handled_obj_xyz, "handled_obj_pos":handled_obj_pos, "handled_obj_ohs":handled_obj_ohs,
                   # human data
                    "masked_xyz":masked_xyz, "finger_mask_idxs":finger_mask_idxs}
    return return_data

def prepare_pred_data(self, xyz, xyz_mask_idxs, obj_xyz, obj_ohs, finger_mask_idxs):

    """
    function used to prepare predicted data into a dictionary for use in the ensemble selector module
    """
    
    # always remember to reshape from [batch,2] to [batch*2] before doing any processing
    # always remember to reshape from [batch,2] to [batch*2] before reshaping the other dimensions for safety ?
    
    length      = self.inp_length
    batch_size  = self.batch_size
    num_obj_classes   = self.num_obj_classes
    obj_padded_length = self.obj_padded_length
    
    # obj_xyz and obj_pos
    # - set obj_pos z coordinate to 0 as we use it to center the object and human along the horizontal plane
    obj_pos = torch.mean(obj_xyz,dim=-2,keepdim=True)                           # [batch, length, num_padded_objs, 1, 3]
    obj_pos[:,:,:,:,-1] = 0                                                     # [batch, length, num_padded_objs, 1, 3]
    obj_xyz = obj_xyz - obj_pos                                                 # [batch, length, num_padded_objs, 4, 3]
    obj_xyz = torch.reshape(obj_xyz,[batch_size,length,obj_padded_length,-1])   # [batch, length, num_padded_objs, 12]
    
    # obj_ohs
    obj_ohs = torch.unsqueeze(obj_ohs,dim=1).repeat(1,self.inp_length,1,1)  # [batch, length, obj_padded_length, num_obj_classes]
    
    # xyz
    xyz = xyz[:,:,None,:,:].repeat(1,1,obj_padded_length,1,1)   # [batch, inp_length, obj_padded_length, 53, 3]
    xyz = xyz - obj_pos         
    # make sure the missing joints that are 0 remain 0
    for i,xyz_mask_idxs_i in enumerate(xyz_mask_idxs):
        xyz[i,:,:,xyz_mask_idxs_i,:] = 0
    xyz = torch.reshape(xyz,[batch_size,length,obj_padded_length,159])
    
    # return_data             
    return_data = {# object data
                    "handled_obj_xyz":obj_xyz, "handled_obj_pos":obj_pos, "handled_obj_ohs":obj_ohs,
                   # human data
                    "masked_xyz":xyz, "finger_mask_idxs":finger_mask_idxs}
    return return_data