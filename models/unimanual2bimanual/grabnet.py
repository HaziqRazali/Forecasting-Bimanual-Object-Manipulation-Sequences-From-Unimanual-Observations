import torch.distributions as tdist
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch_geometric.nn as tgnn
import numpy as np
import time
import sys
sys.path.append("..")

from torch_geometric.nn import MessagePassing
from models.components import *
from models.utils import *
from models.unimanual2bimanual.utils import *
from models.unimanual2bimanual.components import *
        
class model(nn.Module):
    def __init__(self, args):
        super(model, self).__init__()
                 
        for key, value in args.__dict__.items():
            setattr(self, key, value)
                     
        # object encoder
        self.object_encoder = make_mlp(self.object_encoder_units, self.object_encoder_activations)
                     
        # human encoder decoder
        self.human_encoder = make_mlp(self.human_encoder_units, self.human_encoder_activations)
        self.human_decoder = HumanDecoder(self.human_decoder_units, self.human_decoder_activations, self.human_decoder_type)
    
        # finger decoder
        if self.num_finger_decoders != 0:
            assert self.num_finger_decoders <= 2
            self.finger_decoder = FingerDecoder(self.finger_decoder_units, self.finger_decoder_activations, self.finger_decoder_type, self.num_finger_decoders)
    
    def forward(self, data, mode):
            
        # get input data
        inp_data = get_data(self, data, center=True)
        
        """                                                     
        predict                                                   
        - current code always predicts the full 53 markers        
          - and then we either compute the loss over all 53 markers
          - or we select the markers corresponding to the xyz_mask_idxs
        """
        
        # predict body and finger together
        if len(self.finger_decoder_units) == 0:

            # form prior
            human_data  = self.human_encoder(torch.cat([inp_data[k] for k in self.human_decoder_prior["human"]],dim=-1))
            object_data = self.object_encoder(torch.cat([inp_data[k] for k in self.human_decoder_prior["object"]],dim=-1))
            prior       = torch.cat([human_data, object_data],dim=-1)   # [batch, inp_length, 135+obj_data or human_encoder+object_encoder] 
            posterior   = None 
            
            # decode
            human_decoder_out = self.human_decoder(prior=prior, posterior=posterior, mode=mode)
                    
            # collect predicted pose
            pred_xyz = human_decoder_out["out"][:,:,:self.human_dim[0]*self.human_dim[1]]   # [batch, inp_length, 159]
            pred_xyz = torch.reshape(pred_xyz,[self.batch_size,self.inp_length,-1,3])       # [batch, inp_length, 53, 3]
            pred_xyz = pred_xyz + obj_pos                                                   # [batch, inp_length, 53, 3] add the offset back
            
            # collect predicted finger joints
            pred_finger = human_decoder_out["out"][:,:,self.human_dim[0]*self.human_dim[1]:]                                 # [batch, inp_length, 38]
            pred_finger = torch.reshape(pred_finger,[self.batch_size,self.inp_length,self.finger_dim[0],self.finger_dim[1]]) # [batch, inp_length, 2, 19]
            if any([x == "inp_missing_finger" for x in self.loss_names]) and pred_finger.shape[2] == 2:
                pred_finger = torch.stack([pred_finger[i,:,finger_mask_idxs_i,:] for i,finger_mask_idxs_i in enumerate(data["finger_mask_idxs"])])
            else:
                pred_finger = torch.squeeze(pred_finger)
            
            # return data
            assert "xyz" in self.loss_names[0] and "finger" in self.loss_names[1]
            return_data = {self.loss_names[0]:pred_xyz, self.loss_names[1]:pred_finger}
            if "pose_posterior" in human_decoder_out:
                return_data = {**return_data, 
                               "pose_posterior":{"mu":human_decoder_out["pose_posterior"]["mu"],"log_var":human_decoder_out["pose_posterior"]["log_var"]}}
        
        # predict finger and body separately
        else:
            
            """
            decode body
            """
                        
            # form prior
            human_data  = self.human_encoder(torch.cat([inp_data[k] for k in self.human_decoder_prior["human"]],dim=-1))
            object_data = self.object_encoder(torch.cat([inp_data[k] for k in self.human_decoder_prior["object"]],dim=-1))
            prior       = torch.cat([human_data, object_data],dim=-1)   # [batch, inp_length, 135+obj_data or human_encoder+object_encoder] 
            posterior   = None 
            
            # decode
            human_decoder_out = self.human_decoder(prior=prior, posterior=posterior, mode=mode)
            
            # collect output
            pred_xyz = human_decoder_out["out"]                                                                         # [batch, inp_length, 159]
            pred_xyz = torch.reshape(pred_xyz,[self.batch_size,self.inp_length,self.human_dim[0],self.human_dim[1]])    # [batch, inp_length, 53, 3]
            pred_xyz = pred_xyz + inp_data["handled_obj_pos"][:,:,None,:]                                               # add x,y back

            """
            decode finger
            """
        
            # form prior
            object_data = self.object_encoder(torch.cat([inp_data[k] for k in self.finger_decoder_prior["object"]],dim=-1))            
            prior       = object_data
            posterior   = None
            
            # decode
            finger_decoder_out = self.finger_decoder(prior=prior, posterior=posterior, finger_mask_idxs=data["finger_mask_idxs"], mode=mode)
            
            # collect output
            pred_finger = finger_decoder_out["out"]
            pred_finger = torch.reshape(pred_finger,[self.batch_size,self.inp_length,self.finger_dim[0],self.finger_dim[1]])    # [batch, inp_length, 1 or 2, 19]
            
            # if we want to compute the loss for only the missing finger
            if self.loss_names[1] == "inp_missing_finger" and pred_finger.shape[2] == 2:
                pred_finger = torch.stack([pred_finger[i,:,finger_mask_idxs_i,:] for i,finger_mask_idxs_i in enumerate(data["finger_mask_idxs"])])  # [batch, inp_length, 19]
            else:
                pred_finger = torch.squeeze(pred_finger)                                                                                            # [batch, inp_length, 19] or [batch, inp_length, 2, 19]
        
            
            inp_xyz_mask = torch.clone(data["inp_handled_obj_idxs"])    # [batch, inp_length]
            inp_xyz_mask[inp_xyz_mask != -1] = 1                        # [batch, inp_length]
            inp_xyz_mask[inp_xyz_mask == -1] = 0                        # [batch, inp_length]
            inp_xyz_mask[inp_xyz_mask == self.obj_padded_length] = 0    # [batch, inp_length]
            inp_xyz_mask = inp_xyz_mask.unsqueeze(-1).unsqueeze(-1)     # [batch, inp_length, 1, 1]
                
            # return data
            return_data = {"pred_inp_xyz":pred_xyz,        "inp_missing_finger":pred_finger,
                           "true_inp_xyz":data["inp_xyz"],
                                "inp_xyz_mask":inp_xyz_mask}
        
        """
        collect outputs for visualization
        - only if the model predicts partially
        """
                    
        # load the ground truth human pose and replace the relevant elements with our predictions
        if self.loss_names[0] == "inp_missing_xyz":
            inp_xyz = torch.clone(data["inp_xyz"])                                      # [batch, inp_length, 53, 3]
            for i,xyz_mask_idxs_i in enumerate(data["xyz_mask_idxs"]):
                inp_xyz[i,:,xyz_mask_idxs_i,:] = pred_xyz[i,:,:,:]                      # [batch, inp_length, 53, 3]
            return_data = {**return_data, "inp_xyz":inp_xyz}
        
        # load the ground truth finger pose and replace the relevant elements with our predictions
        if self.loss_names[1] == "inp_missing_finger":
            inp_finger = torch.clone(data["inp_finger"])                                # [batch, inp_length, 2, 19]
            for i,finger_mask_idxs_i in enumerate(data["finger_mask_idxs"]):
                inp_finger[i,:,finger_mask_idxs_i,:] = pred_finger[i,:,:]               # [batch, inp_length, 2, 19]
            return_data = {**return_data, "inp_finger":inp_finger}
                
        return return_data
