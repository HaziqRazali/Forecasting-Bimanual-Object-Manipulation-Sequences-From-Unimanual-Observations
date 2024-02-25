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
      
class ensemble_reconstruction_module(nn.Module):
    def __init__(self, args):
        super(ensemble_reconstruction_module, self).__init__()
                 
        for key, value in args.__dict__.items():
            setattr(self, key, value)
        
        # for transformer
        if self.temporal_encoder_type == "TransformerEncoder" or self.temporal_decoder_type == "TransformerDecoder":
            self.inp_object_label_embedder = nn.Embedding(self.object_label_embedder_units[0], self.object_label_embedder_units[1])
            
        """
        reconstruction modules
        """
                
        # individual encoders
        self.inp_human_encoder  = make_mlp(self.inp_human_encoder_units, self.inp_human_encoder_activations)
        self.inp_object_encoder = make_mlp(self.inp_object_encoder_units, self.inp_object_encoder_activations)
        
        # temporal encoder
        if self.temporal_encoder_type == "GraphGRU":
            self.temporal_encoder = GraphGRU(self.graph_encoder_units, self.graph_encoder_activations, self.graph_encoder_type)
        elif self.temporal_encoder_type == "TransformerEncoder":
            self.temporal_encoder = TransformerEncoder(self.transformer_encoder_units)
        else:
            print("Unknown self.temporal_encoder_type: {}".format(self.temporal_encoder_type))
            sys.exit()
        
        # ranked prediction module
        self.encoder_omm = ObjectMotionModule(args,num_classifiers=1)
        
        # ensemble selector module
        self.encoder_pem = PoseEnsembleModule(args)
                        
    def forward(self, data, mode):
        assert mode == "train" or mode == "val"
        return_data = {}
                              
        # variables
        num_obj_classes         = self.num_obj_classes
        obj_padded_length       = self.obj_padded_length
        obj_body_padded_length  = self.obj_body_padded_length
        obj_ids_unpadded_length = data["obj_ids_unpadded_length"]
                      
        # shapes
        #_,_,num_body_joints,body_dim = data["inp_xyz"].shape
        #_,_,num_hands,hand_dim       = data["inp_finger"].shape
        #_,_,_,num_markers,obj_dim    = data["inp_obj_xyz"].shape
                              
        # # # # # #
        # Bi-GRNN #
        # # # # # #
            
        if self.temporal_encoder_type == "GraphGRU":
        
            """
            create mega adjacency graph then convert it to the edge index list
            - table not in
            - 0 are padded objects not connected to anything
            - everything connected to each other
            - body is the last index
            """
            
            batch_adj = []
            for obj_ids in data["obj_ids"]:
                adj = 1 - torch.eye(obj_body_padded_length)
                for i,obj_id in enumerate(obj_ids):
                    # if obj_id is zero, detach it from all
                    if obj_id == 0:
                        adj = detach(adj, i)
                batch_adj.append(adj)
            edge_index = dense_to_sparse(torch.stack(batch_adj)).to(device=torch.cuda.current_device())
            
            """
            form inputs
            """
            
            # fetch data
            inp_data = get_data(self, data, section="inp", center=False)
            
            # obj_data
            obj_data = torch.cat([inp_data[k] for k in self.inp_object_encoder_data_type],dim=-1)   # [batch, inp_length, obj_padded_length, -1]
            obj_data = self.inp_object_encoder(obj_data)                                            # [batch, inp_length, obj_padded_length, -1]
            
            # human_data
            human_data = torch.cat([inp_data[k] for k in self.inp_human_encoder_data_type],dim=-1)  # [batch, inp_length,    -1]
            human_data = self.inp_human_encoder(human_data)                                         # [batch, inp_length,    -1]
            human_data = torch.unsqueeze(human_data,dim=2)                                          # [batch, inp_length, 1, -1]
            
            # concatenate obj_data and human_data
            obj_human_data = torch.cat([obj_data,human_data],dim=2)                                 # [batch, inp_length, obj_body_padded_length, -1]
                                    
            """
            bi-grnn feedforward
            """
                  
            h_out = self.temporal_encoder(timesteps=self.inp_length, data=obj_human_data, edge_index=edge_index)  # [batch, inp_length, obj_body_padded_length, -1]
            
        # # # # # # # # # # # #
        # TransformerEncoder  #
        # # # # # # # # # # # #
        
        if self.temporal_encoder_type == "TransformerEncoder":
            
            """
            form key
            """
            
            # fetch data
            inp_data = get_data(self, data, section="inp", center=False)
            
            # obj_data
            # ["masked_obj_xyz","obj_ohs"]
            obj_data = torch.cat([inp_data[k] for k in self.inp_object_encoder_data_type],dim=-1)   # [batch, inp_length, obj_padded_length, -1]
            obj_data = self.inp_object_encoder(obj_data)                                            # [batch, inp_length, obj_padded_length, -1]
            
            # human_data
            # ["masked_xyz","masked_finger"]
            human_data = torch.cat([inp_data[k] for k in self.inp_human_encoder_data_type],dim=-1)  # [batch, inp_length,    -1]
            human_data = self.inp_human_encoder(human_data)                                         # [batch, inp_length,    -1]
            human_data = torch.unsqueeze(human_data,dim=2)                                          # [batch, inp_length, 1, -1]
            
            # concatenate obj_data and human_data
            obj_human_data = torch.cat([obj_data,human_data],dim=2)                                 # [batch, inp_length, obj_body_padded_length, -1]
                        
            # object and human ids
            all_ids = data["obj_human_ids"]                                             # [batch,             obj_body_padded_length]
            all_ids = all_ids[:,None,:].repeat(1,self.inp_length,1)                     # [batch, inp_length, obj_body_padded_length]
            all_ids = self.object_label_embedder(all_ids.type(torch.cuda.LongTensor))   # [batch, inp_length, obj_body_padded_length, -1]
            
            # positional data
            inp_length = [self.inp_length for _ in range(self.batch_size)]
            pos_emb = cosine_positional_embedder(inp_length, self.inp_length, self.position_embedder_units)    # [batch, inp_length, -1]
            pos_emb = torch.unsqueeze(pos_emb,2).repeat(1,1,self.obj_body_padded_length,1)                     # [batch, inp_length, obj_body_padded_length, -1]
            
            # form key
            key = obj_human_data + all_ids + pos_emb    # [batch, inp_length, obj_body_padded_length, -1]
        
            """
            form mask
            """
            
            # attend 0 ignore 1
            mask = torch.zeros([self.batch_size,self.inp_length,self.obj_body_padded_length]) # [batch, inp_length, obj_body_padded_length]
            for i,obj_ids_unpadded_length_i in enumerate(obj_ids_unpadded_length):
                mask[i,:,obj_ids_unpadded_length_i:] = 1
            # attend to body
            mask[:,:,-1] = 0
            mask = mask.type(torch.cuda.BoolTensor)
        
            """
            transformer feedforward
            """
            
            h_out = self.temporal_encoder(key=key, mask=mask)    # [batch, pose_padded_length, transformer_encoder_units[0]] # [32, 50, 128]
        
        """
        encoder ranked prediction module
        """
        
        # omm
        encoder_omm_out = self.encoder_omm(h=h_out, obj_ohs=data["obj_ohs"], mode=mode)
        inp_obj_xyz = encoder_omm_out["pred_obj_xyz"]
        
        # mask for loss computation
        inp_obj_xyz_mask = torch.ones(data["inp_obj_xyz"].shape).to(device=torch.cuda.current_device())
        for i,obj_ids_unpadded_length_i in enumerate(data["obj_ids_unpadded_length"]):
            inp_obj_xyz_mask[i,:,obj_ids_unpadded_length_i:] = 0
        
        # ground truth
        inp_handled_obj_idxs = torch.clone(data["inp_handled_obj_idxs"])            # [batch, inp_length]
        inp_handled_obj_idxs[inp_handled_obj_idxs == -1] = self.obj_padded_length   # [batch, inp_length]
        
        #print(inp_handled_obj_idxs.shape)
        #print(encoder_omm_out["p_log"].shape)
        encoder_omm_out["p_log"] = torch.permute(encoder_omm_out["p_log"],[0,2,1])
        
        # return_data
        return_data = {**return_data,
        
                        # object coordinates
                        "pred_inp_obj_xyz":encoder_omm_out["pred_obj_xyz"], 
                        "true_inp_obj_xyz":data["inp_obj_xyz"],
                             "inp_obj_xyz_mask":inp_obj_xyz_mask,
                            
                        # object classification
                        "pred_inp_handled_obj":encoder_omm_out["p_log"],
                        "true_inp_handled_obj":inp_handled_obj_idxs}
        
        """
        encoder ensemble and selector module
        - predict given the ground truth object coordinates
        """
                
        # - free_net uses the hidden states
        # - grab_net uses the one ground truth object coordinates
        inp_data = get_data(self, data, section="inp", center=True)
        encoder_pem_out = self.encoder_pem(human_h=h_out[:,:,-1,:], data=inp_data, mode=mode)
                        
        # mask for loss computation
        # - 1 if person is handling something so we compute the loss
        # - 0 if person not handling anything so we do not compute the loss
        # - always set to 1 before 0
        mask = torch.clone(data["inp_handled_obj_idxs"])    # [batch, inp_length]
        mask[mask != -1] = 1                                # [batch, inp_length]
        mask[mask == -1] = 0                                # [batch, inp_length]
        
        # confidence mask
        free_net_inp_finger_confidence = 1
        grab_net_inp_finger_confidence = 1
        if "inp_finger_confidence" in data:
        
            # free_net finger confidence
            free_net_inp_finger_confidence = data["inp_finger_confidence"]                                  # [batch, inp_length, num_hands, finger_dim]
            
            # grab_net finger confidence
            grab_net_inp_finger_confidence = data["inp_finger_confidence"]                                  # [batch, inp_length, num_hands, finger_dim]
            masked_hand_idx = data["masked_hand_idx"]                                                       # [batch]
            masked_hand_idx = masked_hand_idx[:,None,None,None].repeat(1,self.inp_length,1,self.finger_dim) # [batch, inp_length, 1,         finger_dim]
            grab_net_inp_finger_confidence = torch.gather(grab_net_inp_finger_confidence,2,masked_hand_idx) # [batch, inp_length, 1,         finger_dim]
            grab_net_inp_finger_confidence = torch.squeeze(grab_net_inp_finger_confidence)                  # [batch, inp_length,            finger_dim]
                
        # return_data
        return_data = {**return_data,
        
                        # free_net xyz                                                  # free_net finger
                        # - no mask required                                            # - no mask required
                                                                                        # - confidence for kit rgbd
                        "pred_free_net_inp_xyz":encoder_pem_out["pred_free_net_xyz"],   "pred_free_net_inp_finger":encoder_pem_out["pred_free_net_finger"],
                        "true_free_net_inp_xyz":data["inp_xyz"],                        "true_free_net_inp_finger":data["inp_finger"],
                                                                                             "free_net_inp_finger_confidence":free_net_inp_finger_confidence,
                                        
                        # grab_net xyz                                                  # grab_net finger
                        # - mask for when the person is not handling any object         # - mask for when the person is not handling any object
                                                                                        # - confidence for kit rgbd
                        "pred_grab_net_inp_xyz":encoder_pem_out["pred_grab_net_xyz"],   "pred_grab_net_inp_finger":encoder_pem_out["pred_grab_net_finger"], # [batch, length, 19]
                        "true_grab_net_inp_xyz":data["inp_xyz"],                        "true_grab_net_inp_finger":data["inp_missing_finger"],
                             "grab_net_inp_xyz_mask":mask[:,:,None,None],                    "grab_net_inp_finger_mask":mask[:,:,None],
                                                                                             "grab_net_inp_finger_confidence":grab_net_inp_finger_confidence
                        }
         
        """
        additional operations for visualization and evaluation
        """
        
        if self.additional_operations == 1:
            
            # - free_net uses the hidden states
            # - grab_net uses every predicted object coordinates
            pred_data = prepare_pred_data(self, xyz=data["inp_masked_xyz"], xyz_mask_idxs=data["inp_xyz_mask_idxs"], obj_xyz=encoder_omm_out["pred_obj_xyz"], obj_ohs=data["obj_ohs"], finger_mask_idxs=data["inp_finger_mask_idxs"])
            encoder_pem_out_all_objects = self.encoder_pem(human_h=h_out[:,:,-1,:], data=pred_data, mode=mode)
        
            # get top 1 classifications
            # - to be used for selecting between grab_net and free_net
            pred_inp_handled_obj = return_data["pred_inp_handled_obj"]                          # [batch, inp_length, obj_body_padded_length]
            pred_inp_handled_obj = torch.argmax(pred_inp_handled_obj,dim=-1)                    # [batch, inp_length]
            pred_inp_handled_obj = pred_inp_handled_obj[:,:,None,None,None].repeat(1,1,1,53,3)  # [batch, inp_length, 1, 53, 3]
        
            # get xyz
            pred_grab_net_inp_xyz = encoder_pem_out_all_objects["pred_grab_net_xyz"]    # [batch, inp_length, obj_padded_length, 53, 3]
            pred_free_net_inp_xyz = encoder_pem_out_all_objects["pred_free_net_xyz"]    # [batch, inp_length,                    53, 3]
            pred_free_net_inp_xyz = pred_free_net_inp_xyz[:,:,None,:,:]                 # [batch, inp_length,                 1, 53, 3]
                                              
            # merge
            pred_inp_xyz = torch.cat([pred_grab_net_inp_xyz,pred_free_net_inp_xyz],axis=2)  # [batch, inp_length, obj_body_padded_length, 53, 3]
            pred_inp_xyz = torch.squeeze(torch.gather(pred_inp_xyz,2,pred_inp_handled_obj)) # [batch, inp_length, 53, 3]  
            
            # return_data
            return_data = {**return_data, "true_inp_xyz":data["inp_xyz"], "pred_inp_xyz":pred_inp_xyz, "pred_grab_net_inp_xyz_all_objects":pred_grab_net_inp_xyz}
                
        return return_data