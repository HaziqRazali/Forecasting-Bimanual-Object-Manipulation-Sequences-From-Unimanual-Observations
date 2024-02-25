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
      
def project_to_image(x, y, z, cx, cy, fx, fy):
        
    z = z * -1
    x =  (x * fx / z) + cx
    y = -(y * fy / z) + cy
    return x,y    
    
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
        self.encoder_omm = KITRGBDObjectMotionModule(args,num_classifiers=1)
        
        # ensemble selector module
        self.encoder_pem = KITRGBDPoseEnsembleModule(args)
                        
    def forward(self, data, mode):
        assert mode == "train" or mode == "val"
        return_data = {}
                              
        # variables
        num_obj_wrist_classes = self.num_obj_wrist_classes
        obj_wrist_padded_length = self.obj_wrist_padded_length
        #obj_padded_length       = self.obj_padded_length
        #obj_body_padded_length  = self.obj_body_padded_length
        #obj_ids_unpadded_length = data["obj_ids_unpadded_length"]
                      
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
                adj = 1 - torch.eye(obj_wrist_padded_length)
                for i,obj_id in enumerate(obj_ids):
                    # if obj_id is zero, detach it from all
                    if obj_id == 0:
                        adj = detach(adj, i)
                batch_adj.append(adj)
            edge_index = dense_to_sparse(torch.stack(batch_adj)).to(device=torch.cuda.current_device())
            
            """# create mega adjacency graph then convert it to the edge index list
            # - table not in
            # - 0 are padded objects not connected to anything
            # - hands and objects connected to each other except the zeros
            # - hands are the last 2 indices
            batch_adj = []
            for rhand_obj_ids in data["obj_ids"]:
                adj = 1 - torch.eye(obj_wrist_padded_length) # [obj_wrist_padded_length, obj_wrist_padded_length] should be adj = 1 - torch.eye(obj_wrist_padded_length)
                for i,rhand_obj_id in enumerate(rhand_obj_ids):
                    # if rhand_obj_id is zero, detach it from all
                    if rhand_obj_id == 0:
                        adj = detach(adj, i)
                batch_adj.append(adj)
                #print(adj)
                #sys.exit()
            edge_index = dense_to_sparse(torch.stack(batch_adj)).to(device=torch.cuda.current_device())"""
            
            """
            form inputs
            """
            
            # fetch data
            inp_data = get_kit_rgbd_data(self, data, section="inp", center=False)
            
            # obj_data
            obj_data   = torch.cat([inp_data[k] for k in self.inp_object_encoder_data_type],dim=-1) # [batch, inp_length, obj_padded_length, -1]
            wrist_data = torch.cat([inp_data[k] for k in self.inp_human_encoder_data_type],dim=-1)  # [batch, inp_length, 2,                 -1]
            obj_wrist_data = torch.cat([obj_data,wrist_data],axis=2)                                # [batch, inp_length, obj_wrist_padded_length, -1]
            obj_wrist_data = self.inp_object_encoder(obj_wrist_data)                                # [batch, inp_length, obj_wrist_padded_length, -1]
                                                            
            """
            bi-grnn feedforward
            """
                  
            h_out = self.temporal_encoder(timesteps=self.inp_length, data=obj_wrist_data, edge_index=edge_index)  # [batch, inp_length, obj_wrist_padded_length, -1]
            
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
        encoder object motion module
        - we treat the wrist as an object so we continue to predict it
        TODO
        - mask for obj and wrist
        """
                
        # omm
        encoder_omm_out = self.encoder_omm(h=h_out, obj_ohs=data["obj_ohs"], wrist_ohs=data["wrist_ohs"], mode=mode)
        pred_inp_obj_xyz    = encoder_omm_out["pred_obj_wrist_xyz"][:,:,:-2] # [batch, inp_length, obj_padded_length, 1, 3]
        pred_inp_wrist_xyz  = encoder_omm_out["pred_obj_wrist_xyz"][:,:,-2:] # [batch, inp_length, 2,                 1, 3]
        pred_inp_wrist_xyz  = torch.squeeze(pred_inp_wrist_xyz)              # [batch,             2,                    3]
        
        # mask for loss computation
        inp_obj_xyz_mask = torch.ones(data["inp_obj_xyz"].shape).to(device=torch.cuda.current_device())
        for i,obj_ids_unpadded_length_i in enumerate(data["obj_ids_unpadded_length"]):
            inp_obj_xyz_mask[i,:,obj_ids_unpadded_length_i:] = 0
        
        # ground truth
        inp_handled_obj_idxs = torch.clone(data["inp_handled_obj_idxs"])            # [batch, inp_length]
        inp_handled_obj_idxs[inp_handled_obj_idxs == -1] = self.obj_padded_length   # [batch, inp_length]
                
        # return_data
        # - dont need to reconstruct the wrist
        return_data = {**return_data,
        
                        # object coordinates
                        "pred_inp_obj_xyz":pred_inp_obj_xyz, 
                        "true_inp_obj_xyz":data["inp_obj_xyz"],
                             "inp_obj_xyz_mask":inp_obj_xyz_mask,
                        
                        # wrist coordinates
                        "pred_inp_wrist_xyz":pred_inp_wrist_xyz,
                        "true_inp_wrist_xyz":data["inp_wrist_xyz"],
                        
                        # object classification
                        "pred_inp_handled_obj":encoder_omm_out["p_log"],
                        "true_inp_handled_obj":inp_handled_obj_idxs}
        
        """
        predict body
        """
        
        # undo scaling
        wrist_xy = inp_data["wrist_xyz"].detach().clone() / self.xyz_scale         # [batch, inp_length, 2, 3]
        
        # denormalize
        wrist_xy = wrist_xy + data["table_center"][:,None,None,:]                 # [batch, inp_length, 2, 3]
        wrist_xy = torch.permute(wrist_xy,[1,0,2,3])                              # [inp_length, batch, 2, 3]
        wrist_xy = torch.permute(wrist_xy,[0,1,3,2])                              # [inp_length, batch, 3, 2]
        wrist_xy = torch.stack([torch.matmul(data["rx"], x) for x in wrist_xy])   # [inp_length, batch, 3, 2]
        wrist_xy = torch.permute(wrist_xy,[0,1,3,2])                              # [inp_length, batch, 2, 3]
        wrist_xy = torch.permute(wrist_xy,[1,0,2,3])                              # [batch, inp_length, 2, 3]]
        
        # project to image
        x = wrist_xy[:,:,:,0] # [batch, inp_length, hands=2]
        y = wrist_xy[:,:,:,1] # [batch, inp_length, hands=2]
        z = wrist_xy[:,:,:,2] # [batch, inp_length, hands=2]
        x = torch.reshape(x,[-1,2])
        y = torch.reshape(y,[-1,2])
        z = torch.reshape(z,[-1,2])
        cx = data["cx"][:,None,:].repeat(1,self.inp_length,1) # [batch, inp_length, 1]
        cy = data["cy"][:,None,:].repeat(1,self.inp_length,1) # [batch, inp_length, 1]
        fx = data["fx"][:,None,:].repeat(1,self.inp_length,1) # [batch, inp_length, 1]
        fy = data["fy"][:,None,:].repeat(1,self.inp_length,1) # [batch, inp_length, 1]
        cx = torch.reshape(cx,[-1,1])
        cy = torch.reshape(cy,[-1,1])
        fx = torch.reshape(fx,[-1,1])
        fy = torch.reshape(fy,[-1,1])
        x, y = project_to_image(x, y, z, cx, cy, fx, fy) # [batch* inp_length], [batch* inp_length]
        x /= torch.tensor([640,480]).to(device=torch.cuda.current_device())
        y /= torch.tensor([640,480]).to(device=torch.cuda.current_device()) 
        wrist_xy = torch.stack([x,y],dim=1)                                           # [batch* inp_length, 2, 2]
        wrist_xy = torch.reshape(wrist_xy,[self.batch_size, self.inp_length, 2, 2])   # [batch, inp_length, 2, 2]
        
        """
        # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # sanity check to make sure the values are correct  #
        # # # # # # # # # # # # # # # # # # # # # # # # # # #
        
        print(data["wrist_xyz"].shape)      # [2, 150, 2, 3]     
        print(data["table_center"].shape)   # [2, 3]
        wrist = data["wrist_xyz"][:,0,:,:]  # [2, 2, 3]
        wrist = wrist / self.xyz_scale
        
        # denormalize
        wrist = wrist + torch.unsqueeze(data["table_center"],dim=1)
        wrist = torch.matmul(data["rx"], torch.permute(wrist,[0,2,1]))
        wrist = torch.permute(wrist,[0,2,1])                              # [batch, hands=2, dim=3]
        
        x = wrist[:,:,0]  # [batch=2, hands=2]
        y = wrist[:,:,1]  # [batch=2, hands=2]
        z = wrist[:,:,2]  # [batch=2, hands=2]
        x, y = project_to_image(x, y, z, data["cx"], data["cy"], data["fx"], data["fy"]) # [2,2], [2,2], [2,2]
        pred_wrist_xyz_t1_detached = torch.stack([x,y],dim=2) # [batch=2, hands=2, dim=2]
        print(pred_wrist_xyz_t1_detached.shape)
        for i in range(2):                        
            print(data["sequence"][i])
            print(pred_wrist_xyz_t1_detached[i,0])
            print(pred_wrist_xyz_t1_detached[i,1])
        sys.exit()
        """
        
        """
        reconstruct body given projected 2D wrist and masked inp_xyz
        - maybe i should use rnn here for the human decoder ... ?
          - no need because its already conditioned on the masked body
        """
        
        # inputs
        wrist_xy                = torch.reshape(wrist_xy,[self.batch_size,self.inp_length,-1])                             # [batch, inp_length, 4]
        inp_masked_xyz          = inp_data["masked_xyz"]                                                                    # [batch, inp_length, 30]
        zeros                   = torch.zeros([self.batch_size,self.inp_length,4]).to(device=torch.cuda.current_device())   # [batch, inp_length, 4]
                
        # free_net
        free_net_inp            = torch.cat([inp_masked_xyz,zeros],axis=-1)
        pred_free_net_inp_xyz   = self.encoder_pem.free_net_human_decoder(free_net_inp)                         # [batch, inp_length, 34]
        pred_free_net_inp_xyz   = torch.reshape(pred_free_net_inp_xyz,[self.batch_size,self.inp_length, 15, 2]) # [batch, inp_length, 15 ,2]
                
        # grab_net
        grab_net_inp            = torch.cat([inp_masked_xyz,wrist_xy],axis=-1)                                 # [batch, inp_length, 34]
        pred_grab_net_inp_xyz   = self.encoder_pem.grab_net_human_decoder(grab_net_inp)                         # [batch, inp_length, 30]
        pred_grab_net_inp_xyz   = torch.reshape(pred_grab_net_inp_xyz,[self.batch_size,self.inp_length, 15, 2]) # [batch, inp_length, 15 ,2]
                
        # return_data
        return_data = {**return_data,
        
                        # free_net xyz                                                  # free_net finger
                        # - no mask required                                            # - no mask required
                                                                                        # - confidence for kit rgbd
                        "pred_free_net_inp_xyz":pred_free_net_inp_xyz,                  #"pred_free_net_inp_finger":encoder_pem_out["pred_free_net_finger"],
                        "true_free_net_inp_xyz":data["inp_xyz"],                        #"true_free_net_inp_finger":data["inp_finger"],
                                                                                        #     "free_net_inp_finger_confidence":free_net_inp_finger_confidence,
                                        
                        # grab_net xyz                                                  # grab_net finger
                        # - mask for when the person is not handling any object         # - mask for when the person is not handling any object
                                                                                        # - confidence for kit rgbd
                        "pred_grab_net_inp_xyz":pred_grab_net_inp_xyz,                  #"pred_grab_net_inp_finger":encoder_pem_out["pred_grab_net_finger"], # [batch, length, 19]
                        "true_grab_net_inp_xyz":data["inp_xyz"]                         #"true_grab_net_inp_finger":data["inp_missing_finger"],
                             #"grab_net_inp_xyz_mask":mask[:,:,None,None]               #     "grab_net_inp_finger_mask":mask[:,:,None],
                                                                                        #     "grab_net_inp_finger_confidence":grab_net_inp_finger_confidence
                        }
        
        #for k,v in return_data.items():
        #    print(k, v.shape) 
        return return_data