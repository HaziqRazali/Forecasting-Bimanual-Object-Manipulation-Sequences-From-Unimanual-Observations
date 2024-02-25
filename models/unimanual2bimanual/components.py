import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
import torch_geometric.nn as tgnn

from models.components import *

class TransformerEncoder(nn.Module):
    def __init__(self, transformer_encoder_units):
        super(TransformerEncoder, self).__init__()
                
        # https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
        # https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
                
        # transformer encoder        
        transformer_encoder_layer = nn.TransformerEncoderLayer(transformer_encoder_units[0], transformer_encoder_units[1], transformer_encoder_units[2], batch_first=True)
        self.transformer_encoder  = nn.TransformerEncoder(transformer_encoder_layer, transformer_encoder_units[3])
        
    def forward(self, **kwargs):
                
        # pass kwargs to self
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        
            
        key  = self.key     # [batch, length, num_objs, -1]
        mask = self.mask    # [batch, length, num_objs]
        batch_size  = key.shape[0]
        length      = key.shape[1]
        num_objs    = key.shape[2]
        feature_dim = key.shape[3]
        
        # reshape data to the shape required by transformer
        key  = torch.reshape(key, [batch_size, length*num_objs, -1])
        mask = torch.reshape(mask,[batch_size, length*num_objs])
                
        # temporal encoder feedforward
        h = self.transformer_encoder(key, src_key_padding_mask=mask)  # [batch, length* num_objs, -1]
        h = torch.reshape(h,[batch_size, length, num_objs, -1])
        
        return h

class TransformerDecoder(nn.Module):
    def __init__(self, transformer_decoder_units):
        super(TransformerDecoder, self).__init__()
                
        # transformer decoder
        transformer_decoder_layer = nn.TransformerDecoderLayer(transformer_decoder_units[0], transformer_decoder_units[1], transformer_decoder_units[2], batch_first=True)
        self.transformer_decoder  = nn.TransformerDecoder(transformer_decoder_layer, transformer_decoder_units[3])
        
    def forward(self, **kwargs):
        
        # pass kwargs to self
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # temporal decoder feedforward
        h = self.transformer_decoder(self.key, memory=self.memory, tgt_key_padding_mask=self.mask)  # [batch, length* obj_body_padded_length, -1]
        
        return h

class KITRGBDObjectMotionModule(nn.Module):
    def __init__(self, args, **kwargs):
        super(KITRGBDObjectMotionModule, self).__init__()
    
        for key, value in args.__dict__.items():
            setattr(self, key, value)        
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        # coordinate decoders
        self.object_decoder = make_mlp(self.omm_object_decoder_units, self.omm_object_decoder_activations)
        
        # classifier
        self.obj_classifier = nn.ModuleList([make_mlp(self.omm_object_classifier_units, self.omm_object_classifier_activations) for _ in range(self.num_classifiers)])
        
    def forward(self, **kwargs):
        
        # pass kwargs to self
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        length = self.h.shape[1] # either 10 or 1
        
        # # # # # # # # # # 
        # object decoder  #
        # # # # # # # # # #
        
        obj_wrist_h = self.h                                                # [batch, length, obj_wrist_padded_length, -1]
        if self.omm_object_decoder_data_type == ["obj_ohs"]: 
            obj_ohs         = self.obj_ohs[:,None,:].repeat(1,length,1,1)   # [batch, length, obj_padded_length, num_obj_wrist_classes]
            wrist_ohs       = self.wrist_ohs[:,None,:].repeat(1,length,1,1) # [batch, length, 2,                 
            obj_wrist_ohs   = torch.cat([obj_ohs,wrist_ohs],axis=2)         # [batch, length, obj_wrist_padded_length, num_obj_wrist_classes]
            obj_wrist_h     = torch.cat([obj_wrist_h,obj_wrist_ohs],dim=-1) # [batch, length, obj_wrist_padded_length, -1 + num_obj_wrist_classes]
        else:
            pass
        pred_obj_wrist_xyz = self.object_decoder(obj_wrist_h)                                                                                               # [batch, length, obj_wrist_padded_length, num_obj_markers* obj_dim]
        pred_obj_wrist_xyz = torch.reshape(pred_obj_wrist_xyz,[self.batch_size, length, self.obj_wrist_padded_length, self.num_obj_markers, self.obj_dim])  # [batch, length, obj_wrist_padded_length, num_obj_markers, obj_dim]
        
        # # # # # # # # # # # 
        # object classifier #
        # # # # # # # # # # #

        all_h = torch.clone(self.h)                                                                                                                 # [batch, length,     obj_wrist_padded_length,             -1]
        if self.omm_object_classifier_data_type == ["obj_ohs"]:
            all_ohs = torch.clone(self.obj_ohs)                                                                                                     # [batch,             obj_padded_length,       num_obj_wrist_classes]
            all_ohs = torch.cat([all_ohs, torch.zeros(self.batch_size,2,self.num_obj_wrist_classes).to(device=torch.cuda.current_device())],dim=1)  # [batch,             obj_wrist_padded_length, num_obj_wrist_classes]
            all_ohs = torch.unsqueeze(all_ohs,1).repeat(1,length,1,1)                                                                               # [batch, length,     obj_wrist_padded_length, num_obj_wrist_classes]
            all_h = torch.cat([all_h,all_ohs],dim=-1)
        else:
            pass
        p_log = torch.stack([self.obj_classifier[i](all_h) for i in range(self.num_classifiers)])   # [2, batch, length, obj_wrist_padded_length, 1]
        p_log = p_log[:,:,:,:,0]                                                                    # [2, batch, length, obj_wrist_padded_length]
        p_log = p_log[0] if self.num_classifiers == 1 else p_log                                    #    [batch, length, obj_wrist_padded_length] or [2, batch, length, obj_wrist_padded_length]
        
        return_data = {"pred_obj_wrist_xyz":pred_obj_wrist_xyz,"p_log":p_log}
        return return_data

class ObjectMotionModule(nn.Module):
    def __init__(self, args, **kwargs):
        super(ObjectMotionModule, self).__init__()
    
        for key, value in args.__dict__.items():
            setattr(self, key, value)        
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        # coordinate decoders
        self.object_decoder = make_mlp(self.omm_object_decoder_units, self.omm_object_decoder_activations)
        
        # classifier
        self.obj_classifier = nn.ModuleList([make_mlp(self.omm_object_classifier_units, self.omm_object_classifier_activations) for _ in range(self.num_classifiers)])
        
    def forward(self, **kwargs):
        
        # pass kwargs to self
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        length = self.h.shape[1] # either 10 or 1
        
        # # # # # # # # # # 
        # object decoder  #
        # # # # # # # # # #
        
        obj_h   = torch.clone(self.h[:,:,:-1,:])                                                    # [batch, length, obj_padded_length, D* h_dim] all except the final human node
        if self.omm_object_decoder_data_type == ["obj_ohs"]: 
            obj_ohs = torch.unsqueeze(self.obj_ohs,dim=1).repeat(1,length,1,1)                      # [batch, length, obj_padded_length, num_obj_classes]
            obj_h   = torch.cat([obj_h,obj_ohs],dim=-1)                                             # [batch, length, obj_padded_length, D* h_dim + num_obj_classes]
        else:
            pass
        pred_obj_xyz = self.object_decoder(obj_h)                                                                                           # [batch, length, obj_padded_length, num_obj_markers* obj_dim]
        pred_obj_xyz = torch.reshape(pred_obj_xyz,[self.batch_size, length, self.obj_padded_length, self.num_obj_markers, self.obj_dim])    # [batch, length, obj_padded_length, num_obj_markers, obj_dim]
        
        # # # # # # # # # # # 
        # object classifier #
        # # # # # # # # # # #

        all_h = torch.clone(self.h)
        if self.omm_object_classifier_data_type == ["obj_ohs"]:
            all_ohs = torch.clone(self.obj_ohs)                                                                                                 # [batch,             obj_padded_length,      num_obj_classes]
            all_ohs = torch.cat([all_ohs, torch.zeros(self.batch_size,1,self.num_obj_classes).to(device=torch.cuda.current_device())],dim=1)    # [batch,             obj_body_padded_length, num_obj_classes]
            all_ohs = torch.unsqueeze(all_ohs,1).repeat(1,length,1,1)                                                                           # [batch, length,     obj_body_padded_length, num_obj_classes]
            all_h = torch.cat([all_h,all_ohs],dim=-1)
        else:
            pass
        p_log = torch.stack([self.obj_classifier[i](all_h) for i in range(self.num_classifiers)])   # [2, batch, length, obj_body_padded_length, 1]
        p_log = p_log[:,:,:,:,0]                                                                    # [2, batch, length, obj_body_padded_length]
        p_log = p_log[0] if self.num_classifiers == 1 else p_log                                    #    [batch, length, obj_body_padded_length] or [2, batch, length, obj_body_padded_length]
        
        #p_log   = self.obj_classifier(all_h)    # [batch, length, obj_body_padded_length, 1]
        #p_log   = torch.squeeze(p_log)          # [batch, length, obj_body_padded_length]

        return_data = {"pred_obj_xyz":pred_obj_xyz,"p_log":p_log}
        return return_data

class KITRGBDPoseEnsembleModule(nn.Module):
    def __init__(self, args):
        super(KITRGBDPoseEnsembleModule, self).__init__()
        
        for key, value in args.__dict__.items():
            setattr(self, key, value)
    
        # free_net
        self.free_net_human_decoder = make_mlp(self.free_net_human_decoder_units, self.free_net_human_decoder_activations)
        
        # grab_net
        self.grab_net_human_encoder  = make_mlp(self.grab_net_human_encoder_units, self.grab_net_human_encoder_activations)
        self.grab_net_object_encoder = make_mlp(self.grab_net_object_encoder_units, self.grab_net_object_encoder_activations)
        self.grab_net_human_decoder  = make_mlp(self.grab_net_human_decoder_units, self.grab_net_human_decoder_activations)
                
    def forward(self, human_h, data, mode, prefix=""):
            
        """
        free_net
        - we do the prediction outside during forecasting since we need to aggregate them
        """
        
        if human_h is not None:

            free_net_out    = self.free_net_human_decoder(human_h)                                                                          # [batch, inp_length, self.num_body_joints* self.body_dim + self.num_hands* self.finger_dim] only the final human node
            pred_free_net_human  = free_net_out[:,:,:self.num_body_joints*self.body_dim]                                                    # [batch, inp_length, self.num_body_joints* self.body_dim]
            pred_free_net_human  = torch.reshape(pred_free_net_human,[self.batch_size,self.inp_length,self.num_body_joints,self.body_dim])  # [batch, inp_length, self.num_body_joints, self.body_dim]

        else:
            
            pred_free_net_human  = None
            pred_free_net_finger = None

        """
        grab_net
        """
                
        # # # # # # # # #
        # prepare data  #
        # # # # # # # # #
        
        # get obj_data
        grab_net_obj_data = torch.cat([data[k] for k in self.grab_net_object_encoder_data_type],dim=-1)     # [batch, inp_length, -1]
        grab_net_obj_data = self.grab_net_object_encoder(grab_net_obj_data)                                 # [batch, inp_length, -1]
        
        # get human data
        grab_net_human_data = torch.cat([data[k] for k in self.grab_net_human_encoder_data_type],dim=-1)    # [batch, inp_length, -1] 
        grab_net_human_data = self.grab_net_human_encoder(grab_net_human_data)                              # [batch, inp_length, -1]
                
        # # # # # # # # # # # #
        # predict human pose  #
        # # # # # # # # # # # #
                    
        grab_net_out = self.grab_net_human_decoder(torch.cat([grab_net_obj_data,grab_net_human_data],dim=-1))           # [batch, inp_length, num_body_joints* body_dim]
        pred_grab_net_human = torch.reshape(grab_net_out,[*grab_net_out.shape[:-1],self.num_body_joints,self.body_dim]) # [batch, inp_length, num_body_joints, body_dim]
        
        # train - we predict the pose for only the masked_obj_xyz so we add the respective center
        pred_grab_net_human = pred_grab_net_human + data["handled_obj_pos"] # [batch, inp_length, 53, 3]
                    
        # # # # # # # # # # # #
        # predict finger pose #
        # # # # # # # # # # # #
        
        # decode finger
        if self.grab_net_num_finger_decoders != 0:
            grab_net_finger_decoder_out = self.grab_net_finger_decoder(prior=grab_net_obj_data, posterior=None, finger_mask_idxs=data["finger_mask_idxs"], mode=mode)   # [batch, inp_length, 19] or [batch, inp_length, obj_padded_length, 19]
            pred_grab_net_finger = grab_net_finger_decoder_out["out"]                                                                                                   # [batch, inp_length, 19] or [batch, inp_length, obj_padded_length, 19]
        else:
            pred_grab_net_finger = None
            
        """
        return data
        """
                       
        return_data = {# free_net
                       prefix+"pred_free_net_xyz":pred_free_net_human, prefix+"pred_free_net_finger":pred_free_net_finger,
                       
                       # grab_net
                       prefix+"pred_grab_net_xyz":pred_grab_net_human, prefix+"pred_grab_net_finger":pred_grab_net_finger}
        return return_data

class PoseEnsembleModule(nn.Module):
    def __init__(self, args):
        super(PoseEnsembleModule, self).__init__()
        
        # i may want to predict the hand mocap markers separately
        # or rather, i use the finger decoder to predict the hand mocap markers while the human decoder predicts the remaining joints as proposed in the paper
        # i can probably try to stabilize the output
        # hopefully the visual results are not so obvious
        
        for key, value in args.__dict__.items():
            setattr(self, key, value)
    
        # free_net
        self.free_net_human_decoder = make_mlp(self.free_net_human_decoder_units, self.free_net_human_decoder_activations)
        
        # grab_net
        self.grab_net_human_encoder  = make_mlp(self.grab_net_human_encoder_units, self.grab_net_human_encoder_activations)
        self.grab_net_object_encoder = make_mlp(self.grab_net_object_encoder_units, self.grab_net_object_encoder_activations)
        self.grab_net_human_decoder  = make_mlp(self.grab_net_human_decoder_units, self.grab_net_human_decoder_activations)
        
        # finger_decoder
        if self.grab_net_num_finger_decoders != 0:
            assert self.grab_net_num_finger_decoders <= 2
            self.grab_net_finger_decoder = FingerDecoder(self.grab_net_finger_decoder_units, self.grab_net_finger_decoder_activations, self.grab_net_finger_decoder_type, self.grab_net_num_finger_decoders)
        
    def forward(self, human_h, data, mode, prefix=""):
            
        """
        free_net
        - we do the prediction outside during forecasting since we need to aggregate them
        """
        
        if human_h is not None:

            free_net_out    = self.free_net_human_decoder(human_h)                                                                          # [batch, inp_length, self.num_body_joints* self.body_dim + self.num_hands* self.finger_dim] only the final human node
            pred_free_net_human  = free_net_out[:,:,:self.num_body_joints*self.body_dim]                                                    # [batch, inp_length, self.num_body_joints* self.body_dim]
            pred_free_net_human  = torch.reshape(pred_free_net_human,[self.batch_size,self.inp_length,self.num_body_joints,self.body_dim])  # [batch, inp_length, self.num_body_joints, self.body_dim]    
            if self.free_net_human_decoder_output_type == ["xyz","finger"]:
                pred_free_net_finger = free_net_out[:,:,self.num_body_joints*self.body_dim:]                                                    # [batch, inp_length, self.num_hands* self.finger_dim]
                pred_free_net_finger = torch.reshape(pred_free_net_finger,[self.batch_size,self.inp_length,self.num_hands,self.finger_dim])       # [batch, inp_length, self.num_hands, self.finger_dim]
            else:
                pred_free_net_finger = None

        else:
            
            pred_free_net_human  = None
            pred_free_net_finger = None

        """
        grab_net
        """
                
        # # # # # # # # #
        # prepare data  #
        # # # # # # # # #
        
        # get obj_data
        grab_net_obj_data = torch.cat([data[k] for k in self.grab_net_object_encoder_data_type],dim=-1)     # [batch, inp_length, -1]
        grab_net_obj_data = self.grab_net_object_encoder(grab_net_obj_data)                                 # [batch, inp_length, -1]
        
        # get human data
        grab_net_human_data = torch.cat([data[k] for k in self.grab_net_human_encoder_data_type],dim=-1)    # [batch, inp_length, -1] 
        grab_net_human_data = self.grab_net_human_encoder(grab_net_human_data)                              # [batch, inp_length, -1]
                
        # # # # # # # # # # # #
        # predict human pose  #
        # # # # # # # # # # # #
                    
        grab_net_out = self.grab_net_human_decoder(torch.cat([grab_net_obj_data,grab_net_human_data],dim=-1))           # [batch, inp_length, num_body_joints* body_dim]
        pred_grab_net_human = torch.reshape(grab_net_out,[*grab_net_out.shape[:-1],self.num_body_joints,self.body_dim]) # [batch, inp_length, num_body_joints, body_dim]
        
        # train - we predict the pose for only the masked_obj_xyz so we add the respective center
        pred_grab_net_human = pred_grab_net_human + data["handled_obj_pos"] # [batch, inp_length, 53, 3]
                    
        # # # # # # # # # # # #
        # predict finger pose #
        # # # # # # # # # # # #
        
        # decode finger
        if self.grab_net_num_finger_decoders != 0:
            grab_net_finger_decoder_out = self.grab_net_finger_decoder(prior=grab_net_obj_data, posterior=None, finger_mask_idxs=data["finger_mask_idxs"], mode=mode)   # [batch, inp_length, 19] or [batch, inp_length, obj_padded_length, 19]
            pred_grab_net_finger = grab_net_finger_decoder_out["out"]                                                                                                   # [batch, inp_length, 19] or [batch, inp_length, obj_padded_length, 19]
        else:
            pred_grab_net_finger = None
            
        """
        return data
        """
                       
        return_data = {# free_net
                       prefix+"pred_free_net_xyz":pred_free_net_human, prefix+"pred_free_net_finger":pred_free_net_finger,
                       
                       # grab_net
                       prefix+"pred_grab_net_xyz":pred_grab_net_human, prefix+"pred_grab_net_finger":pred_grab_net_finger}
        return return_data

class PoseEnsembleModule_v2(nn.Module):
    def __init__(self, args):
        super(PoseEnsembleModule_v2, self).__init__()
        
        # i may want to predict the hand mocap markers separately
        # or rather, i use the finger decoder to predict the hand mocap markers while the human decoder predicts the remaining joints as proposed in the paper
        # i can probably try to stabilize the output
        # hopefully the visual results are not so obvious
        
        for key, value in args.__dict__.items():
            setattr(self, key, value)
    
        # free_net
        self.free_net_human_decoder = make_mlp(self.free_net_human_decoder_units, self.free_net_human_decoder_activations)
        
        # grab_net
        self.grab_net_human_encoder  = make_mlp(self.grab_net_human_encoder_units, self.grab_net_human_encoder_activations)
        self.grab_net_object_encoder = make_mlp(self.grab_net_object_encoder_units, self.grab_net_object_encoder_activations)
        self.grab_net_human_decoder  = make_mlp(self.grab_net_human_decoder_units, self.grab_net_human_decoder_activations)
                
        # finger_decoder
        if self.grab_net_num_finger_decoders != 0:
            assert self.grab_net_num_finger_decoders <= 2
            self.grab_net_finger_encoder = make_mlp(self.grab_net_finger_encoder_units, self.grab_net_finger_encoder_activations)
            self.grab_net_finger_decoder = FingerDecoder(self.grab_net_finger_decoder_units, self.grab_net_finger_decoder_activations, self.grab_net_finger_decoder_type, self.grab_net_num_finger_decoders)
        
    def forward(self, human_h, data, mode, prefix=""):
            
        """
        free_net
        - we do the prediction outside during forecasting since we need to aggregate them
        """
        
        if human_h is not None:

            free_net_out    = self.free_net_human_decoder(human_h)                                                                          # [batch, inp_length, self.num_body_joints* self.body_dim + self.num_hands* self.finger_dim] only the final human node
            pred_free_net_human  = free_net_out[:,:,:self.num_body_joints*self.body_dim]                                                    # [batch, inp_length, self.num_body_joints* self.body_dim]
            pred_free_net_human  = torch.reshape(pred_free_net_human,[self.batch_size,self.inp_length,self.num_body_joints,self.body_dim])  # [batch, inp_length, self.num_body_joints, self.body_dim]    
            if self.free_net_human_decoder_output_type == ["xyz","finger"]:
                pred_free_net_finger = free_net_out[:,:,self.num_body_joints*self.body_dim:]                                                    # [batch, inp_length, self.num_hands* self.finger_dim]
                pred_free_net_finger = torch.reshape(pred_free_net_finger,[self.batch_size,self.inp_length,self.num_hands,self.finger_dim])       # [batch, inp_length, self.num_hands, self.finger_dim]
            else:
                pred_free_net_finger = None

        else:
            
            pred_free_net_human  = None
            pred_free_net_finger = None

        """
        grab_net
        """
                
        # # # # # # # # #
        # prepare data  #
        # # # # # # # # #
        
        # get obj_data
        grab_net_obj_data = torch.cat([data[k] for k in self.grab_net_object_encoder_data_type],dim=-1)     # [batch, inp_length, -1]
        grab_net_obj_data = self.grab_net_object_encoder(grab_net_obj_data)                                 # [batch, inp_length, -1]
        
        # get human data
        grab_net_human_data = torch.cat([data[k] for k in self.grab_net_human_encoder_data_type],dim=-1)    # [batch, inp_length, -1] 
        grab_net_human_data = self.grab_net_human_encoder(grab_net_human_data)                              # [batch, inp_length, -1]
                
        # # # # # # # # # # # #
        # predict human pose  #
        # # # # # # # # # # # #
                    
        grab_net_out = self.grab_net_human_decoder(torch.cat([grab_net_obj_data,grab_net_human_data],dim=-1))           # [batch, inp_length, num_body_joints* body_dim]
        pred_grab_net_human = torch.reshape(grab_net_out,[*grab_net_out.shape[:-1],self.num_body_joints,self.body_dim]) # [batch, inp_length, num_body_joints, body_dim]
        
        # train - we predict the pose for only the masked_obj_xyz so we add the respective center
        pred_grab_net_human = pred_grab_net_human + data["handled_obj_pos"] # [batch, inp_length, 53, 3]
                    
        # # # # # # # # # # # #
        # predict finger pose #
        # # # # # # # # # # # #
        # - input = obj_xyz, obj_ohs
        # - output = wrist, and finger joints (has to be really zero centered)
        
        # decode wrist and fingers
        if self.grab_net_num_finger_decoders != 0:
            grab_net_finger_data = torch.cat([data[k] for k in self.grab_net_finger_encoder_data_type],dim=-1)
            grab_net_finger_decoder_out = self.grab_net_finger_decoder(prior=grab_net_finger_data, posterior=None, finger_mask_idxs=data["finger_mask_idxs"], mode=mode)    # [batch, inp_length, 19] or [batch, inp_length, obj_padded_length, 19]
            pred_grab_net_finger = grab_net_finger_decoder_out["out"][:,:,:19]                                                                                              # [batch, inp_length, 19] or [batch, inp_length, obj_padded_length, 19]
            pred_grab_net_wrist  = grab_net_finger_decoder_out["out"][:,:,19:]   
            pred_grab_net_wrist  = torch.reshape(pred_grab_net_wrist,[*pred_grab_net_wrist.shape[:-1],5,3]) # [batch, inp_length, 5, 3]
            pred_grab_net_wrist  = pred_grab_net_wrist + data["handled_obj_pos_zc"]
            
        else:
            pred_grab_net_finger = None
            
        """
        return data
        """
                       
                       
        return_data = {# free_net
                       prefix+"pred_free_net_xyz":pred_free_net_human, prefix+"pred_free_net_finger":pred_free_net_finger,
                       
                       # grab_net
                       prefix+"pred_grab_net_xyz":pred_grab_net_human, prefix+"pred_grab_net_finger":pred_grab_net_finger, prefix+"pred_grab_net_wrist_xyz":pred_grab_net_wrist}
        return return_data

class PoseEnsembleModule_v3(nn.Module):
    def __init__(self, args):
        super(PoseEnsembleModule_v3, self).__init__()
        
        # i may want to predict the hand mocap markers separately
        # or rather, i use the finger decoder to predict the hand mocap markers while the human decoder predicts the remaining joints as proposed in the paper
        # i can probably try to stabilize the output
        # hopefully the visual results are not so obvious
        
        for key, value in args.__dict__.items():
            setattr(self, key, value)
    
        # free_net
        self.free_net_human_decoder = make_mlp(self.free_net_human_decoder_units, self.free_net_human_decoder_activations)
        
        # grab_net
        self.grab_net_human_encoder  = make_mlp(self.grab_net_human_encoder_units, self.grab_net_human_encoder_activations)
        self.grab_net_object_encoder = make_mlp(self.grab_net_object_encoder_units, self.grab_net_object_encoder_activations)
        self.grab_net_human_decoder  = make_mlp(self.grab_net_human_decoder_units, self.grab_net_human_decoder_activations)
                
        # finger_decoder
        if self.grab_net_num_finger_decoders != 0:
            assert self.grab_net_num_finger_decoders <= 2
            self.grab_net_finger_encoder = make_mlp(self.grab_net_finger_encoder_units, self.grab_net_finger_encoder_activations)
            self.grab_net_finger_decoder = FingerDecoder(self.grab_net_finger_decoder_units, self.grab_net_finger_decoder_activations, self.grab_net_finger_decoder_type, self.grab_net_num_finger_decoders)
        
    def forward(self, human_h, data, mode, prefix=""):
            
        """
        free_net
        - we do the prediction outside during forecasting since we need to aggregate them
        """
        
        if human_h is not None:

            free_net_out    = self.free_net_human_decoder(human_h)                                                                          # [batch, inp_length, self.num_body_joints* self.body_dim + self.num_hands* self.finger_dim] only the final human node
            pred_free_net_human  = free_net_out[:,:,:self.num_body_joints*self.body_dim]                                                    # [batch, inp_length, self.num_body_joints* self.body_dim]
            pred_free_net_human  = torch.reshape(pred_free_net_human,[self.batch_size,self.inp_length,self.num_body_joints,self.body_dim])  # [batch, inp_length, self.num_body_joints, self.body_dim]    
            if self.free_net_human_decoder_output_type == ["xyz","finger"]:
                pred_free_net_finger = free_net_out[:,:,self.num_body_joints*self.body_dim:]                                                    # [batch, inp_length, self.num_hands* self.finger_dim]
                pred_free_net_finger = torch.reshape(pred_free_net_finger,[self.batch_size,self.inp_length,self.num_hands,self.finger_dim])       # [batch, inp_length, self.num_hands, self.finger_dim]
            else:
                pred_free_net_finger = None

        else:
            
            pred_free_net_human  = None
            pred_free_net_finger = None

        """
        grab_net
        """
                
        # # # # # # # # #
        # prepare data  #
        # # # # # # # # #
        
        # get obj_data
        grab_net_obj_data = torch.cat([data[k] for k in self.grab_net_object_encoder_data_type],dim=-1)     # [batch, inp_length, -1]
        grab_net_obj_data = self.grab_net_object_encoder(grab_net_obj_data)                                 # [batch, inp_length, -1]
        
        # get human data
        grab_net_human_data = torch.cat([data[k] for k in self.grab_net_human_encoder_data_type],dim=-1)    # [batch, inp_length, -1] 
        grab_net_human_data = self.grab_net_human_encoder(grab_net_human_data)                              # [batch, inp_length, -1]
                
        # # # # # # # # # # # #
        # predict finger pose #
        # # # # # # # # # # # #
        # - input = obj_xyz, obj_ohs
        # - output = wrist, and finger joints (has to be really zero centered)
        
        # decode wrist and fingers
        if self.grab_net_num_finger_decoders != 0:
            grab_net_finger_data = torch.cat([data[k] for k in self.grab_net_finger_encoder_data_type],dim=-1)
            grab_net_finger_decoder_out = self.grab_net_finger_decoder(prior=grab_net_finger_data, posterior=None, finger_mask_idxs=data["finger_mask_idxs"], mode=mode)    # [batch, inp_length, 19] or [batch, inp_length, obj_padded_length, 19]
            pred_grab_net_finger = grab_net_finger_decoder_out["out"][:,:,:19]                                                                                              # [batch, inp_length, 19] or [batch, inp_length, obj_padded_length, 19]
            pred_grab_net_wrist  = grab_net_finger_decoder_out["out"][:,:,19:]   
            pred_grab_net_wrist  = torch.reshape(pred_grab_net_wrist,[*pred_grab_net_wrist.shape[:-1],5,3]) # [batch, inp_length, 5, 3]
            pred_grab_net_wrist  = pred_grab_net_wrist + data["handled_obj_pos_zc"]
            
        else:
            pred_grab_net_finger = None
            
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # predict human pose                                                    #
        # - use masked human but replace wrist with wrist predicted by grab_net #
        # - do i need to zero center it back ?
        # - do i need object data ? probably not anymore
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                    
        grab_net_out = self.grab_net_human_decoder(torch.cat([grab_net_obj_data,grab_net_human_data],dim=-1))           # [batch, inp_length, num_body_joints* body_dim]
        pred_grab_net_human = torch.reshape(grab_net_out,[*grab_net_out.shape[:-1],self.num_body_joints,self.body_dim]) # [batch, inp_length, num_body_joints, body_dim]
        
        # train - we predict the pose for only the masked_obj_xyz so we add the respective center
        pred_grab_net_human = pred_grab_net_human + data["handled_obj_pos"] # [batch, inp_length, 53, 3]
                                
        """
        return data
        """
                       
                       
        return_data = {# free_net
                       prefix+"pred_free_net_xyz":pred_free_net_human, prefix+"pred_free_net_finger":pred_free_net_finger,
                       
                       # grab_net
                       prefix+"pred_grab_net_xyz":pred_grab_net_human, prefix+"pred_grab_net_finger":pred_grab_net_finger, prefix+"pred_grab_net_wrist_xyz":pred_grab_net_wrist}
        return return_data

class GraphGRU(nn.Module):
    def __init__(self, graph_units, graph_activations, graph_type):
        super(GraphGRU, self).__init__()
        
        # we only want to MLP each unit at the input, not after the concatenation
        # the initial input is assumed to have already been MLP-ed
        units = [[]] + [graph_units[0]] if len(graph_units[0]) != 0 else [[]]
        activations = [[]] + [graph_activations[0]] if len(graph_activations[0]) != 0 else [[]]
        self.num_edge_convs = len(units)
                                    
        # always nn.Identity because PyTorch Geometric
        self.encoder    = nn.ModuleList([make_mlp(units_i, activations_i) for units_i,activations_i in zip(units,activations)])
        self.edge_conv  = nn.ModuleList([tgnn.EdgeConv(nn.Identity(), aggr="max") for _ in range(self.num_edge_convs)])
                    
        # rnn
        self.num_layers  = graph_units[1][2]
        self.num_directions = 2 if graph_units[1][3] == 1 else 1
        self.num_directions_and_layers = self.num_directions * self.num_layers
        self.rnn         = nn.GRU(graph_units[1][0], graph_units[1][1], graph_units[1][2],bidirectional=graph_units[1][3])
        self.rnn_hi      = nn.ModuleList([make_mlp(graph_units[1][4:], graph_activations[1]) for _ in range(self.num_directions_and_layers)])
    
    def forward(self, timesteps = None, **kwargs):
        
        # pass kwargs to self
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        # we initialize the hidden states here
        if timesteps is not None:
                
            # dimensions
            data = self.data            # [batch, length, obj_body_padded_length, -1]
            batch_size  = data.shape[0] # batch_size
            length      = data.shape[1] # length
            num_objs    = data.shape[2] # obj_body_padded_length
            feature_dim = data.shape[3] # -1
                
            # reshape data to the shape required by torch_geometric
            data = torch.permute(data, [1,0,2,3])                           # [length, batch, num_objs, -1]
            data = torch.reshape(data, [length, batch_size*num_objs, -1])   # [length, batch* num_objs, -1]
            
            h = None
            h_list = []
            for t in range(timesteps):
                
                # data at timestep t
                data_t = data[t] # [batch* num_objs, -1]
                
                # edge conv
                for i in range(self.num_edge_convs):
                    data_t = self.encoder[i](data_t)
                    data_t = self.edge_conv[i](data_t, self.edge_index)
            
                # maybe initialize hidden states
                h = torch.stack([self.rnn_hi[i](data_t) for i in range(self.num_directions_and_layers)]) if h is None else h
            
                # reshape input
                data_t = torch.unsqueeze(data_t,0)                                                      # [1, batch* num_objs, -1]
                            
                # compute GRU
                _, h = self.rnn(data_t, h)                                                              # [D* num_layers, batch* num_objs, -1]
                
                # reshape hidden state and collect
                out = torch.reshape(h,[self.num_directions, self.num_layers, h.shape[1], h.shape[2]])   # [D, num_layers, batch* num_objs, -1]
                out = out[:,-1]                                                                         # [D,             batch* num_objs, -1]
                h_list.append(out)
            
            h_list = torch.stack(h_list)                                                        # [timesteps, D, batch* num_objs,   h_dim]
            h_list = torch.reshape(h_list,[timesteps,-1,batch_size,num_objs,h_list.shape[-1]])  # [timesteps, D, batch, num_objs,   h_dim]
            h_list = torch.permute(h_list,[0,2,3,1,4])                                          # [timesteps, batch, num_objs, D,   h_dim]
            h_list = torch.reshape(h_list,[h_list.shape[0],h_list.shape[1],h_list.shape[2],-1]) # [timesteps, batch, num_objs, D*   h_dim]
            h_list = torch.permute(h_list,[1,0,2,3])                                            # [batch, timesteps, num_objs, D*   h_dim]
            return h_list
        
        else:
        
            # dimensions
            data = self.data            # [batch, length, obj_body_padded_length, -1]
            batch_size  = data.shape[0] # batch_size
            length      = data.shape[1] # length
            num_objs    = data.shape[2] # obj_body_padded_length
            feature_dim = data.shape[3] # -1
            
            # reshape data to the shape required by torch_geometric
            data = torch.permute(data, [1,0,2,3])                           # [length, batch, num_objs, -1]
            data = torch.reshape(data, [length, batch_size*num_objs, -1])   # [length, batch* num_objs, -1]
            
            # conv
            for i in range(self.num_edge_convs):
                data = self.encoder[i](data)
                data = self.edge_conv[i](data, self.edge_index)
                            
            # compute GRU
            h    = self.h                               # [batch, num_objs, -1]
            h    = torch.reshape(h,[-1,h.shape[-1]])    # [batch* num_objs, -1]
            h    = h[None,:,:]                          # [1, batch* num_objs, -1]
            _, h = self.rnn(data, h)                    # [1, batch* num_objs, -1]
            
            # reshape hidden state and collect
            h = torch.reshape(h,[self.num_directions, self.num_layers, h.shape[1], h.shape[2]]) # [1, 1, batch* num_objs, -1]
            h = h[:,-1]                                                                         # [1,    batch* num_objs, -1]
            h = torch.reshape(h,[batch_size,num_objs,h.shape[-1]])                              #       [batch, num_objs, -1]
            return h
            
class EncoderGraphGRU(nn.Module):
    def __init__(self, graph_encoder_units, graph_encoder_activations, encoder_type="graph"):
        super(EncoderGraphGRU, self).__init__()
    
        """
        https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
        https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/message_passing.html
        """
                
        """          
        1) MLP individual output (first will always be nn.Identity())
        2) EdgeConv (nn.Identity(), Concatenate and Max pool)     
        """
        
        self.encoder_type = encoder_type
        if self.encoder_type == "graph":
        
            # we only want to MLP each unit at the input, not after the concatenation
            # the initial input is assumed to have already been MLP-ed
            encoder_units = [] + [graph_encoder_units] if len(graph_encoder_units) != 0 else [[]]
            encoder_activations = [""] + [graph_encoder_activations] if len(graph_encoder_activations) != 0 else [[]]
            self.num_edge_convs = len(encoder_units)
        
            # always nn.Identity!
            self.encoder    = nn.ModuleList([make_mlp(encoder_units_i, encoder_activations_i) for encoder_units_i,encoder_activations_i in zip(encoder_units,encoder_activations)])
            self.edge_conv  = nn.ModuleList([tgnn.EdgeConv(nn.Identity(), aggr="max") for _ in range(self.num_edge_convs)])
        
        elif self.encoder_type == "graph_recurrent":
            
            # we only want to MLP each unit at the input, not after the concatenation
            # the initial input is assumed to have already been MLP-ed
            encoder_units = [[]] + [graph_encoder_units[0]] if len(graph_encoder_units[0]) != 0 else [[]]
            encoder_activations = [[]] + [graph_encoder_activations[0]] if len(graph_encoder_activations[0]) != 0 else [[]]
            self.num_edge_convs = len(encoder_units)
                                        
            # always nn.Identity because PyTorch Geometric
            self.encoder    = nn.ModuleList([make_mlp(encoder_units_i, encoder_activations_i) for encoder_units_i,encoder_activations_i in zip(encoder_units,encoder_activations)])
            self.edge_conv  = nn.ModuleList([tgnn.EdgeConv(nn.Identity(), aggr="max") for _ in range(self.num_edge_convs)])
                        
            # rnn
            self.num_layers  = graph_encoder_units[1][2]
            self.num_directions = 2 if graph_encoder_units[1][3] == 1 else 1
            self.num_directions_and_layers = self.num_directions * self.num_layers
            self.rnn         = nn.GRU(graph_encoder_units[1][0], graph_encoder_units[1][1], graph_encoder_units[1][2],bidirectional=graph_encoder_units[1][3])
            self.rnn_hi      = nn.ModuleList([make_mlp(graph_encoder_units[1][4:], graph_encoder_activations[1]) for _ in range(self.num_directions_and_layers)])
                      
        else:
            print("Unknown graph_encoder_type: {}".format(self.encoder_type))
            sys.exit()
            
    def forward(self, **kwargs):
    
        # pass kwargs to self
        for key, value in kwargs.items():
            setattr(self, key, value)
                    
        # # # # # # # # # #
        # message passing #
        # # # # # # # # # #
        
        if self.encoder_type == "graph":        
            data = self.data
            for i in range(self.num_edge_convs):
                
                #print(self.data.shape) # [batch* num_obj, feature_dim]
                data = self.encoder[i](data)
                data = self.edge_conv[i](data, self.edge_index)           
        
            return {"out":data}                             
        
        elif self.encoder_type == "graph_recurrent":
            data = self.data
            for i in range(self.num_edge_convs):
                
                #print(self.data.shape) # [batch* num_obj, feature_dim]
                data = self.encoder[i](data)
                data = self.edge_conv[i](data, self.edge_index)
            
            # maybe initialize hidden states
            h = torch.stack([self.rnn_hi[i](data) for i in range(self.num_directions_and_layers)]) if self.h is None else self.h
        
            # reshape input
            data = torch.unsqueeze(data,0)                                                          # [1, batch* num_obj, feature_dim]
                        
            # compute GRU
            _, h = self.rnn(data, h)                                                                # [D* num_layers, batch * num_obj, rnn_encoder_hidden]
            
            # get output
            data = torch.reshape(h,[self.num_directions, self.num_layers, h.shape[1], h.shape[2]])  # [D, num_layers, batch * num_obj, rnn_encoder_hidden]
            data = data[:,-1]                                                                       # [D, batch* num_obj, rnn_encoder_hidden]
            
            return {"out":data, "h":h}
        
        else:
            print("Unknown graph_encoder_type: {}".format(self.encoder_type))
            sys.exit()
            
class HumanDecoder(nn.Module):
    def __init__(self, decoder_units, decoder_activations, decoder_type):
        super(HumanDecoder, self).__init__()

        self.decoder_type = decoder_type

        if decoder_type == "mlp":
            self.decoder = make_mlp(decoder_units, decoder_activations)
        
        if decoder_type == "vae":
            self.prior_encoder      = make_mlp(decoder_units[0], decoder_activations[0])
            self.posterior_encoder  = make_mlp(decoder_units[1], decoder_activations[1])
            self.mu_log_var_encoder = make_mlp(decoder_units[2], decoder_activations[2])
            self.decoder            = make_mlp(decoder_units[3], decoder_activations[3])
            self.norm               = tdist.Normal(torch.tensor(0.0), torch.tensor(1.0))
            self.z_dim              = decoder_units[2][-1]//2
        
    def forward(self, **kwargs):

        # pass kwargs to self
        for key, value in kwargs.items():
            setattr(self, key, value)
        assert self.mode == "train" or self.mode == "val"
        
        # # # # # # # # # #
        # if using an mlp #
        # # # # # # # # # #
        
        if self.decoder_type == "mlp":
            prior = self.prior              # [batch, t, data dim]
            out = self.decoder(self.prior)  # [batch, t, decoder_dim]    
            return_data = {"out":out}
        
        # # # # # # # # # #
        # if using a vae  #
        # # # # # # # # # #
        
        if self.decoder_type == "vae":
            
            prior = self.prior                              # [batch, t, dim1]
            posterior = self.posterior                      # [batch, t, dim2]
            
            # compute prior and posterior
            prior       = self.prior_encoder(prior)         # [batch, t, prior/posterior]
            posterior   = self.posterior_encoder(posterior) # [batch, t, prior/posterior]
                        
            # VAE
            # make sure to sample it per-batch and not per-batch and per-time
            if self.mode == "train":

                # compute mu and var
                mu_log_var  = self.mu_log_var_encoder(torch.cat([prior,posterior],dim=-1)) # [batch, t, mu_log_var]
                mu          = mu_log_var[:,:,:mu_log_var.shape[-1]//2]                     # [batch, t, mu_log_var//2]
                log_var     = mu_log_var[:,:,mu_log_var.shape[-1]//2:]                     # [batch, t, mu_log_var//2]
                
                # sample from mu and log_var
                std = torch.exp(0.5*log_var)                                    # [batch, t, mu_log_var//2]
                eps = self.norm.sample([prior.shape[0], 1, self.z_dim]).cuda()  # [batch, 1, mu_log_var//2]
                eps = eps.repeat(1,prior.shape[1],1)                            # [batch, t, mu_log_var//2]
                z   = mu + eps*std                                              # [batch, t, mu_log_var//2]
                
            if self.mode == "val":
                            
                # compute mu and var
                mu_log_var  = self.mu_log_var_encoder(torch.cat([prior,posterior],dim=-1)) # [batch, t, mu_log_var]
                mu          = mu_log_var[:,:,:mu_log_var.shape[-1]//2]                     # [batch, t, mu_log_var//2]
                log_var     = mu_log_var[:,:,mu_log_var.shape[-1]//2:]                     # [batch, t, mu_log_var//2]
                
                # sample from unit gaussian
                z   = self.norm.sample([prior.shape[0], 1, self.z_dim]).cuda()  # [batch, t, mu_log_var//2]
                z   = z.repeat(1,prior.shape[1],1)
            
            # forecast
            out = self.decoder(torch.cat([prior,z],dim=-1))
            return_data = {"out":out, "pose_posterior":{"mu":mu, "log_var":log_var}}
            
        return return_data
        
class FingerDecoder(nn.Module):
    def __init__(self, decoder_units, decoder_activations, decoder_type, num_decoders):
        super(FingerDecoder, self).__init__()

        self.decoder_type = decoder_type

        if decoder_type == "mlp":
            if num_decoders > 1:
                self.decoder = nn.ModuleList([make_mlp(decoder_units, decoder_activations) for _ in range(num_decoders)])
            else:
                self.decoder = make_mlp(decoder_units, decoder_activations)
        
        if decoder_type == "vae":
            self.prior_encoder      = make_mlp(decoder_units[0], decoder_activations[0])
            self.posterior_encoder  = make_mlp(decoder_units[1], decoder_activations[1])
            self.mu_log_var_encoder = make_mlp(decoder_units[2], decoder_activations[2])
            self.decoder            = make_mlp(decoder_units[3], decoder_activations[3])
            self.norm               = tdist.Normal(torch.tensor(0.0), torch.tensor(1.0))
            self.z_dim              = decoder_units[2][-1]//2
        
    def forward(self, **kwargs):

        # pass kwargs to self
        for key, value in kwargs.items():
            setattr(self, key, value)
        assert self.mode == "train" or self.mode == "val"
        
        # # # # # # # # # #
        # if using an mlp #
        # # # # # # # # # #
        
        if self.decoder_type == "mlp":
            prior = self.prior                  # [batch, t, data dim]
            if self.finger_mask_idxs is not None:
                out = torch.stack([self.decoder[finger_mask_idx](prior[i]) for i,finger_mask_idx in enumerate(self.finger_mask_idxs)])   # [batch, t, decoder_dim]                   
            else:
                out = self.decoder(self.prior)  # [batch, t, decoder_dim]                   
            return_data = {"out":out}
        
        # # # # # # # # # #
        # if using a vae  #
        # # # # # # # # # #
        
        if self.decoder_type == "vae":
            
            prior = self.prior                              # [batch, t, dim1]
            posterior = self.posterior                      # [batch, t, dim2]
            
            # compute prior and posterior
            prior       = self.prior_encoder(prior)         # [batch, t, prior/posterior]
            posterior   = self.posterior_encoder(posterior) # [batch, t, prior/posterior]
                        
            # VAE
            # make sure to sample it per-batch and not per-batch and per-time
            if self.mode == "train":

                # compute mu and var
                mu_log_var  = self.mu_log_var_encoder(torch.cat([prior,posterior],dim=-1)) # [batch, t, mu_log_var]
                mu          = mu_log_var[:,:,:mu_log_var.shape[-1]//2]                     # [batch, t, mu_log_var//2]
                log_var     = mu_log_var[:,:,mu_log_var.shape[-1]//2:]                     # [batch, t, mu_log_var//2]
                
                # sample from mu and log_var
                std = torch.exp(0.5*log_var)                                    # [batch, t, mu_log_var//2]
                eps = self.norm.sample([prior.shape[0], 1, self.z_dim]).cuda()  # [batch, 1, mu_log_var//2]
                eps = eps.repeat(1,prior.shape[1],1)                            # [batch, t, mu_log_var//2]
                z   = mu + eps*std                                              # [batch, t, mu_log_var//2]
                
            if self.mode == "val":
                            
                # compute mu and var
                mu_log_var  = self.mu_log_var_encoder(torch.cat([prior,posterior],dim=-1)) # [batch, t, mu_log_var]
                mu          = mu_log_var[:,:,:mu_log_var.shape[-1]//2]                     # [batch, t, mu_log_var//2]
                log_var     = mu_log_var[:,:,mu_log_var.shape[-1]//2:]                     # [batch, t, mu_log_var//2]
                
                # sample from unit gaussian
                z   = self.norm.sample([prior.shape[0], 1, self.z_dim]).cuda()  # [batch, t, mu_log_var//2]
                z   = z.repeat(1,prior.shape[1],1)
            
            # forecast
            out = self.decoder(torch.cat([prior,z],dim=-1))
            return_data = {"out":out, "pose_posterior":{"mu":mu, "log_var":log_var}}
            
        return return_data
        
"""
outdated version
- this version formatted the data this way [batch, obj_padded_length]
- the proper version formatted this way [batch* obj_padded_length]
class PoseEnsembleModule(nn.Module):
    def __init__(self, args):
        super(PoseEnsembleModule, self).__init__()
        
        for key, value in args.__dict__.items():
            setattr(self, key, value)
    
        # free_net
        self.free_net_human_decoder = make_mlp(self.free_net_human_decoder_units, self.free_net_human_decoder_activations)
        
        # grab_net
        self.grab_net_human_encoder  = make_mlp(self.grab_net_human_encoder_units, self.grab_net_human_encoder_activations)
        self.grab_net_object_encoder = make_mlp(self.grab_net_object_encoder_units, self.grab_net_object_encoder_activations)
        self.grab_net_human_decoder  = make_mlp(self.grab_net_human_decoder_units, self.grab_net_human_decoder_activations)
        
        # finger_decoder
        if self.grab_net_num_finger_decoders != 0:
            assert self.grab_net_num_finger_decoders <= 2
            self.grab_net_finger_decoder = FingerDecoder(self.grab_net_finger_decoder_units, self.grab_net_finger_decoder_activations, self.grab_net_finger_decoder_type, self.grab_net_num_finger_decoders)
        
    def forward(self, **kwargs):
    
        # pass kwargs to self
        for key, value in kwargs.items():
            setattr(self, key, value)
               
        if self.human_h is not None:

            free_net_out    = self.free_net_human_decoder(self.human_h)                                         # [batch, inp_length, 197] only the final human node
            print(free_net_out.shape)
            sys.exit()
            pred_free_net_human  = free_net_out[:,:,:159]                                                       # [batch, inp_length, 159]
            pred_free_net_human  = torch.reshape(pred_free_net_human,[self.batch_size,self.inp_length,53,3])    # [batch, inp_length, 53, 3]        
            pred_free_net_finger = free_net_out[:,:,159:]                                                       # [batch, inp_length, 38]
            pred_free_net_finger = torch.reshape(pred_free_net_finger,[self.batch_size,self.inp_length,2,19])   # [batch, inp_length, 2, 19]

        else:
            
            pred_free_net_human  = None
            pred_free_net_finger = None
        
        # we use the handled object when training
        if self.grab_net_mode == "train":
        
            # # # # # # # # # # # # # 
            # prepare grab_net data #
            # # # # # # # # # # # # #
            
            # get obj_data
            grab_net_obj_data = torch.cat([self.grab_net_data[k] for k in self.grab_net_object_encoder_data_type],dim=-1)   # [batch, inp_length, -1]
            grab_net_obj_data = self.grab_net_object_encoder(grab_net_obj_data)                                             # [batch, inp_length, -1]
            
            # get human data
            grab_net_human_data = torch.cat([self.grab_net_data[k] for k in self.grab_net_human_encoder_data_type],dim=-1)  # [batch, inp_length, -1] 
            grab_net_human_data = self.grab_net_human_encoder(grab_net_human_data)                                          # [batch, inp_length, -1]
            
            # # # # # # # # # # # # # # # #
            # predict grab_net human pose #
            # # # # # # # # # # # # # # # #
                        
            grab_net_out = self.grab_net_human_decoder(torch.cat([grab_net_obj_data,grab_net_human_data],dim=-1))   # [batch, inp_length, 159]
            pred_grab_net_human = torch.reshape(grab_net_out,[grab_net_out.shape[0],grab_net_out.shape[1],53,3])    # [batch, inp_length, 53, 3]
            
            # train - we predict the pose for only the masked_obj_xyz so we add the respective center
            pred_grab_net_human = pred_grab_net_human + self.grab_net_data["handled_obj_pos"] # [batch, inp_length, 53, 3]
                        
            # # # # # # # # # # # # # # # # #
            # predict grab_net finger pose  #
            # # # # # # # # # # # # # # # # #
            
            # decode finger
            grab_net_finger_decoder_out = self.grab_net_finger_decoder(prior=grab_net_obj_data, posterior=None, finger_mask_idxs=self.grab_net_data["finger_mask_idxs"], mode=self.mode)    # [batch, inp_length, 19] or [batch, inp_length, obj_padded_length, 19]
            pred_grab_net_finger = grab_net_finger_decoder_out["out"]                                                                                                                       # [batch, inp_length, 19] or [batch, inp_length, obj_padded_length, 19]
        
        # we use the predicted objects when evaluating
        elif self.grab_net_mode == "test":
                        
            # # # # # # # # # # # # # 
            # prepare grab_net data #
            # # # # # # # # # # # # #
            
            # obj_xyz and obj_pos
            # - set obj_pos z coordinate to 0 as we use it to center the object and human along the horizontal plane
            obj_xyz = self.pred_obj_xyz                                                                     # [batch, length, num_padded_objs, 4, 3] suppress ?
            obj_pos = torch.mean(obj_xyz,dim=-2,keepdim=True)                                               # [batch, length, num_padded_objs, 1, 3]
            obj_pos[:,:,:,:,-1] = 0                                                                         # [batch, length, num_padded_objs, 1, 3] 
            obj_xyz = obj_xyz - obj_pos                                                                     # [batch, length, num_padded_objs, 4, 3]
            obj_xyz = torch.reshape(obj_xyz,[self.batch_size,self.inp_length,self.object_padded_length,-1]) # [batch, length, num_padded_objs, 12]
            
            # obj_ohs
            obj_ohs = self.obj_ohs                                                  # [batch,         obj_padded_length, num_obj_classes]
            obj_ohs = torch.unsqueeze(obj_ohs,dim=1).repeat(1,self.inp_length,1,1)  # [batch, length, obj_padded_length, num_obj_classes]
                                                
            # grab_net obj_data
            grab_net_obj_data = torch.cat([obj_xyz,obj_ohs],dim=-1)             # [batch, inp_length, obj_padded_length, 12 + num_obj_classes]
            grab_net_obj_data = self.grab_net_object_encoder(grab_net_obj_data) # [batch, inp_length, obj_padded_length, -1]
            
            # grab_net human_data
            grab_net_human_data = self.inp_masked_xyz                                                           # [batch, inp_length,                    53, 3]
            grab_net_human_data = grab_net_human_data[:,:,None,:,:].repeat(1,1,self.object_padded_length,1,1)   # [batch, inp_length, obj_padded_length, 53, 3]
            grab_net_human_data = grab_net_human_data - obj_pos                                                 # [batch, inp_length, obj_padded_length, 53, 3]            
            # make sure the missing joints that are 0 remain 0
            for i,xyz_mask_idxs_i in enumerate(self.xyz_mask_idxs):
                grab_net_human_data[i,:,:,xyz_mask_idxs_i,:] = 0
            # reshape
            grab_net_human_data = torch.reshape(grab_net_human_data,[self.batch_size,self.inp_length,self.object_padded_length,-1]) # [batch, inp_length, obj_padded_length, 159]
            # encode
            grab_net_human_data = self.grab_net_human_encoder(grab_net_human_data)                                                  # [batch, inp_length, obj_padded_length, 159]
            
            # # # # # # # # # # # # # # # #
            # predict grab_net human pose #
            # # # # # # # # # # # # # # # #
            
            # decode
            grab_net_out = self.grab_net_human_decoder(torch.cat([grab_net_obj_data,grab_net_human_data],dim=-1))               # [batch, inp_length, obj_padded_length, 159]
            pred_grab_net_human = torch.reshape(grab_net_out,[self.batch_size,self.inp_length,self.object_padded_length,53,3])  # [batch, inp_length, obj_padded_length, 53, 3]
            
            # re-add centers
            pred_grab_net_human = pred_grab_net_human + obj_pos # [batch, inp_length, obj_padded_length, 53, 3]
            
            # # # # # # # # # # # # # # # # #
            # predict grab_net finger pose  #
            # # # # # # # # # # # # # # # # #
            
            # decode finger
            grab_net_finger_decoder_out = self.grab_net_finger_decoder(prior=grab_net_obj_data, posterior=None, finger_mask_idxs=self.finger_mask_idxs, mode=self.mode) # [batch, inp_length, obj_padded_length, 19]
            pred_grab_net_finger = grab_net_finger_decoder_out["out"]  
        
        else:
            print("Unknown grab_net_mode:",self.grab_net_mode)
            sys.exit()
                                
        return_data = {# free_net
                       "pred_free_net_xyz":pred_free_net_human, "pred_free_net_finger":pred_free_net_finger,
                       
                       # grab_net
                       "pred_grab_net_xyz":pred_grab_net_human, "pred_grab_net_finger":pred_grab_net_finger}
        return return_data
"""