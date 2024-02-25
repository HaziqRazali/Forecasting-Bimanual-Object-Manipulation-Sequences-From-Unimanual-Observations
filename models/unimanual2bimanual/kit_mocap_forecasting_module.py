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

class ensemble_forecasting_module(nn.Module):
    def __init__(self, args):
        super(ensemble_forecasting_module, self).__init__()
                 
        for key, value in args.__dict__.items():
            setattr(self, key, value)
                     
        # for transformer
        if self.temporal_encoder_type == "TransformerEncoder" or self.temporal_decoder_type == "TransformerDecoder":
            self.out_object_label_embedder = nn.Embedding(self.object_label_embedder_units[0], self.object_label_embedder_units[1])
                          
        # individual encoders
        self.out_human_encoder  = make_mlp(self.out_human_encoder_units, self.out_human_encoder_activations)
        self.out_object_encoder = make_mlp(self.out_object_encoder_units, self.out_object_encoder_activations)
        
        # temporal decoder
        # - we are still using the TransformerEncoder
        if self.temporal_decoder_type == "GraphGRU":
            self.temporal_encoder = GraphGRU(self.graph_encoder_units, self.graph_encoder_activations, self.graph_encoder_type)
            self.temporal_decoder = GraphGRU(self.graph_decoder_units, self.graph_decoder_activations, self.graph_decoder_type)
        elif self.temporal_decoder_type == "TransformerDecoder":
            self.temporal_encoder = TransformerEncoder(self.transformer_encoder_units)
        else:
            print("Unknown self.temporal_decoder_type: {}".format(self.temporal_decoder_type))
            sys.exit()
                                                                              
        # omm
        self.decoder_omm = ObjectMotionModule(args,num_classifiers=2)

        # pem
        self.decoder_pem = PoseEnsembleModule(args)
    
    def forward(self, data, mode):
        assert mode == "train" or mode == "val"
        return_data = {}
        t1 = time.time()
                
        # variables
        num_obj_classes         = self.num_obj_classes
        obj_padded_length       = self.obj_padded_length
        obj_body_padded_length  = self.obj_body_padded_length
        obj_ids_unpadded_length = data["obj_ids_unpadded_length"]
        
        num_body_joints = self.num_body_joints
        body_dim        = self.body_dim
        num_obj_markers = self.num_obj_markers
        obj_dim         = self.obj_dim
        num_hands       = self.num_hands
        finger_dim      = self.finger_dim
        
        # teacher forcing, 0 uses ground truth, 1 uses predictions
        rand = torch.rand(self.batch_size)
        teacher_force = rand > self.teacher_force_ratio
        teacher_force = teacher_force.long()
                                
        # # # # # # # # #
        # Graph Decoder #
        # # # # # # # # #
        
        if self.temporal_decoder_type == "GraphGRU":
            
            # # # # # # # # # #
            # Encoding Phase  #
            # # # # # # # # # #
            
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
            obj_data = torch.cat([inp_data[k] for k in self.out_object_encoder_data_type],dim=-1)   # [batch, inp_length, obj_padded_length, -1]
            obj_data = self.out_object_encoder(obj_data)                                            # [batch, inp_length, obj_padded_length, -1]
            
            # human_data
            human_data = torch.cat([inp_data[k] for k in self.out_human_encoder_data_type],dim=-1)  # [batch, inp_length,    -1]
            human_data = self.out_human_encoder(human_data)                                         # [batch, inp_length,    -1]
            human_data = torch.unsqueeze(human_data,dim=2)                                          # [batch, inp_length, 1, -1]
            
            # concatenate obj_data and human_data
            obj_human_data = torch.cat([obj_data,human_data],dim=2)                                 # [batch, inp_length, obj_body_padded_length, -1]
                                    
            """
            bi-grnn feedforward
            """
                  
            h_out = self.temporal_encoder(timesteps=self.inp_length, data=obj_human_data, edge_index=edge_index)  # [batch, inp_length, obj_body_padded_length, -1]
            h_out = h_out[:,-1]                                                                                   # [batch,             obj_body_padded_length, -1]
            
            # # # # # # # # # #
            # Decoding Phase  #
            # # # # # # # # # #
            
            prev_xyz     = data["inp_xyz"][:,-1]        # [batch, 53, 3]
            prev_obj_xyz = data["inp_obj_xyz"][:,-1]    # [batch, obj_padded_length, 4, 3]
            out_xyz = []
            out_finger = []
            out_obj_xyz = []
            out_p_log = []
            for i in range(self.out_length):
            
                # ranked prediction module
                decoder_omm_out = self.decoder_omm(h=h_out[:,None,:,:], obj_ohs=data["obj_ohs"], mode=mode)
                
                """
                object classification scores
                """
                
                # collect object classification scores
                p_log = decoder_omm_out["p_log"] # [2, batch, length=1, obj_body_padded_length]
                p_log = torch.squeeze(p_log)     # [2, batch, obj_body_padded_length]
                out_p_log.append(p_log)
                
                """
                object coordinates
                """
                
                # collect object coordinates
                obj_xyz = decoder_omm_out["pred_obj_xyz"]   # [batch, length=1, obj_padded_length, 4, 3]
                obj_xyz = obj_xyz[:,0]                      # [batch, obj_padded_length, 4, 3]
                if self.predict_object_velocity == 1:
                    obj_xyz      = obj_xyz + prev_obj_xyz
                    prev_obj_xyz = obj_xyz
                out_obj_xyz.append(obj_xyz)
                
                # teacher force for next iteration
                pred_inp_obj_xyz = obj_xyz
                true_inp_obj_xyz = data["out_obj_xyz"][:,i]
                inp_obj_xyz      = torch.stack([true_inp_obj_xyz,pred_inp_obj_xyz],dim=1)
                inp_obj_xyz      = torch.stack([p[tf] for p,tf in zip(inp_obj_xyz,teacher_force)])  # [batch, obj_wrist_padded_length, num_obj_markers, obj_dim]
                
                """
                human coordinates
                """
                
                # predict human coordinates
                xyz_finger = self.decoder_pem.free_net_human_decoder(h_out[:,-1])               # [batch, num_body_joints* body_dim + num_hands* finger_dim]
                xyz = xyz_finger[:,:self.num_body_joints*self.body_dim]                         # [batch, num_body_joints* body_dim]
                xyz = torch.reshape(xyz,[self.batch_size,self.num_body_joints,self.body_dim])   # [batch, num_body_joints, body_dim]
                if self.predict_body_velocity == 1:
                    xyz      = xyz + prev_xyz
                    prev_xyz = xyz
                out_xyz.append(xyz)
                
                # predict finger
                if self.free_net_human_decoder_output_type == ["xyz","finger"]:
                    finger = xyz_finger[:,self.num_body_joints*self.body_dim:]                      # [batch, num_hands* finger_dim]
                    finger = torch.reshape(finger,[self.batch_size,self.num_hands,self.finger_dim]) # [batch, num_hands, finger_dim]
                    out_finger.append(finger)
                
                # teacher force for next iteration
                pred_inp_xyz = xyz
                true_inp_xyz = data["out_xyz"][:,i]
                inp_xyz      = torch.stack([true_inp_xyz,pred_inp_xyz],dim=1)
                inp_xyz      = torch.stack([p[tf] for p,tf in zip(inp_xyz,teacher_force)])
                
                """
                form input for grnn
                """
                
                human_data      = torch.reshape(inp_xyz,[self.batch_size,-1])                           # [batch, num_body_joints* body_dim]
                if self.out_human_encoder == ["xyz","finger"]:
                    finger_data = torch.reshape(finger,[self.batch_size,-1])                            # [batch, num_hands* finger_dim]
                    human_data  = torch.cat([human_data,finger_data])                                   # [batch, num_body_joints* body_dim + num_hands* finger_dim]
                human_data      = self.out_human_encoder(human_data)                                    # [batch, -1]
                human_data      = human_data[:,None,:]                                                  # [batch, 1, -1]
                obj_data        = torch.reshape(inp_obj_xyz,[self.batch_size,obj_padded_length,-1])     # [batch, obj_padded_length, num_obj_markers* obj_dim]
                obj_data        = torch.cat([obj_data,data["obj_ohs"]],dim=-1)                          # [batch, obj_padded_length, num_obj_markers* obj_dim + num_obj_classes]
                obj_data        = self.out_object_encoder(obj_data)                                     # [batch, obj_padded_length, -1]
                obj_human_data  = torch.cat([obj_data,human_data],dim=1)                                # [batch, obj_body_padded_length, -1]
                obj_human_data  = obj_human_data[:,None,:,:]                                            # [batch, length=1, obj_body_padded_length, -1]
            
                """
                grnn feedforward
                """
            
                h_out = self.temporal_encoder(data=obj_human_data, edge_index=edge_index, h=h_out)  # [batch, obj_body_padded_length, -1]
                        
            # stack
            out_xyz     = torch.stack(out_xyz)                                      # [out_length, batch, 53, 3]
            out_finger  = torch.stack(out_finger) if len(out_finger) != 0 else None # [out_length, batch, 2, 19]
            out_obj_xyz = torch.stack(out_obj_xyz)                                  # [out_length, batch, obj_padded_length, 4, 3]
            out_p_log   = torch.stack(out_p_log,dim=2)                              # [2, batch, out_length, 9]
                   
            # permute
            out_xyz     = torch.permute(out_xyz,[1,0,2,3])                                          # [batch, out_length, 53, 3]
            out_finger  = torch.permute(out_finger,[1,0,2,3]) if out_finger is not None else None   # [batch, out_length, 2, 19]
            out_obj_xyz = torch.permute(out_obj_xyz,[1,0,2,3,4])                                    # [batch, out_length, obj_padded_length, 4, 3]    
                                     
            # handled_obj_idxs ground truth
            out_handled_obj_idxs = torch.clone(data["out_handled_obj_idxs"])            # [batch, 2, out_length]
            out_handled_obj_idxs[out_handled_obj_idxs == -1] = self.obj_padded_length   # [batch, 2, out_length]
            
            # obj_xyz_mask for loss computation
            obj_xyz_mask = torch.ones(out_obj_xyz.shape).to(device=torch.cuda.current_device())
            for i,obj_ids_unpadded_length_i in enumerate(data["obj_ids_unpadded_length"]):
                obj_xyz_mask[i,:,obj_ids_unpadded_length_i:] = 0
            
            # # # # # # #
            # grabnets  #
            # # # # # # #
            
            """
            refine pose outputs using ground truth data
            """
            
            # fetch data
            out_data        = get_data(self, data, section="out", center=True)
            
            # feed forward
            decoder_pem_out = self.decoder_pem(human_h=None, data=out_data, mode=mode)
            #print(decoder_pem_out["pred_grab_net_xyz"].shape)       # [batch*2, out_length, num_body_joints, body_dim]
            #print(decoder_pem_out["pred_grab_net_finger"].shape)    # [batch*2, out_length, finger_dim]
            decoder_pem_out["pred_grab_net_xyz"]    = torch.reshape(decoder_pem_out["pred_grab_net_xyz"],[self.batch_size, 2, self.out_length, self.num_body_joints, self.body_dim])
            if decoder_pem_out["pred_grab_net_finger"] is not None:
                decoder_pem_out["pred_grab_net_finger"] = torch.reshape(decoder_pem_out["pred_grab_net_finger"],[self.batch_size, 2, self.out_length, self.finger_dim])
            
            # prepare xyz ground truth
            xyz = out_data["xyz"]                                                                           # [batch,    out_length, num_body_joints* body_dim]
            xyz = torch.reshape(xyz,[self.batch_size,self.out_length,self.num_body_joints,self.body_dim])   # [batch,    out_length, num_body_joints, body_dim]
            xyz = xyz[:,None,:,:].repeat(1,2,1,1,1)                                                         # [batch, 2, out_length, num_body_joints, body_dim]
            #xyz = torch.reshape(xyz,[-1,self.out_length,xyz.shape[-2],xyz.shape[-1]])                      # [batch* 2, out_length, num_body_joints, body_dim]

            # prepare xyz mask
            # - mask joints
            xyz_mask = torch.zeros(xyz.shape).to(device=torch.cuda.current_device())                                    # [batch* 2, out_length, num_body_joints, body_dim]
            xyz_mask = torch.reshape(xyz_mask,[self.batch_size,2,self.out_length,self.num_body_joints,self.body_dim])   # [batch, 2, out_length, num_body_joints, body_dim]
            if self.grab_net_output_type == "full_body":
                xyz_mask = xyz_mask + 1
            elif self.grab_net_output_type == "arm":
                xyz_mask[:,0,:,self.l_arm_mocap_idxs,:] = 1
                xyz_mask[:,1,:,self.r_arm_mocap_idxs,:] = 1
            else:
                print("Unknown self.grab_net_output_type:", self.grab_net_output_type)
                sys.exit()
            xyz_mask = torch.reshape(xyz_mask,[self.batch_size*2, self.out_length, self.num_body_joints, self.body_dim]) # [batch* 2, out_length, num_body_joints, body_dim]
            # - mask batch and timesteps for when the person is not holding onto anything
            xyz_mask = xyz_mask * out_data["mask"][:,:,None,None]
            # reshape
            xyz_mask = torch.reshape(xyz_mask,[self.batch_size,2, self.out_length, self.num_body_joints, self.body_dim]) # [batch, 2, out_length, num_body_joints, body_dim]
                        
            # prepare finger ground truth
            finger = out_data["finger"]                                                 # [batch, out_length, num_hands* finger_dim]
            finger = torch.reshape(finger,[self.batch_size, self.out_length, 2, -1])    # [batch, out_length, num_hands, finger_dim]
            finger = torch.permute(finger,[0,2,1,3])                                    # [batch, num_hands, out_length, finger_dim]  
            #finger = torch.reshape(finger,[self.batch_size*2, self.out_length, -1])    # [batch* num_hands, out_length, finger_dim]
            
            # prepare finger mask
            # - mask batch and timesteps for when the person is not holding onto anything
            finger_mask = out_data["mask"]                                                  # [batch* 2, out_length]
            finger_mask = torch.reshape(finger_mask,[self.batch_size, 2, self.out_length])  # [batch, 2, out_length]
            finger_mask = finger_mask[:,:,:,None]                                           # [batch, 2, out_length, 1]
            
            # confidence mask
            free_net_out_finger_confidence = 1
            grab_net_out_finger_confidence = 1
            if "out_finger_confidence" in data:
            
                # free_net finger confidence
                free_net_out_finger_confidence = data["out_finger_confidence"]                                  # [batch, out_length, num_hands, finger_dim]
                
                # grab net finger confidence
                grab_net_out_finger_confidence = data["out_finger_confidence"]                                  # [batch, out_length, num_hands, finger_dim]
                grab_net_out_finger_confidence = torch.permute(grab_net_out_finger_confidence,[0,2,1,3])        # [batch, num_hands, out_length, finger_dim]
            
            """
            refine pose outputs using predicted data
            """
            
            # grab_net using 
            # - predicted human data during training and testing
            # - ground truth object data during training
            # - predicted object data during testing
            if self.grab_net_agg_mode == "train":
                handled_obj_xyz = out_data["handled_obj_xyz"]                                                                       # [batch* 2, out_length, num_obj_markers* obj_dim] centered
                handled_obj_xyz = torch.reshape(handled_obj_xyz,[self.batch_size*2, self.out_length, num_obj_markers, obj_dim])     # [batch* 2, out_length, num_obj_markers, obj_dim]
                handled_obj_pos = out_data["handled_obj_pos"]                                                                       # [batch* 2, out_length, 1, obj_dim]
                handled_obj_xyz = handled_obj_xyz + handled_obj_pos                                                                 # un-center the data because it will be re-centered. note that obj_pos = mean(obj_xyz)
                
            elif self.grab_net_agg_mode == "test":
                
                handled_obj_idxs = out_data["handled_obj_idxs"]                                                 # [batch* 2, out_length]
                handled_obj_idxs = handled_obj_idxs[:,:,None,None,None].repeat(1,1,1,num_obj_markers,obj_dim)   # [batch* 2, out_length, 1, num_obj_markers, obj_dim]
                
                """
                # # # # # #
                # for rebuttal
                # - show the hand grasping a different object
                handled_obj_idxs = out_data["handled_obj_idxs"]                                                 # [batch* 2, out_length]
                handled_obj_idxs = torch.reshape(handled_obj_idxs,[self.batch_size,2,self.out_length])          # [batch, 2, out_length]
                handled_obj_idxs[:,1] = 4                                                                       # [batch, 2, out_length]
                handled_obj_idxs = torch.reshape(handled_obj_idxs,[self.batch_size*2,self.out_length])          # [batch* 2, out_length]     
                handled_obj_idxs = handled_obj_idxs[:,:,None,None,None].repeat(1,1,1,num_obj_markers,obj_dim)   # [batch* 2, out_length, 1, num_obj_markers, obj_dim]   
                # # # # # #
                """
                
                handled_obj_xyz = torch.clone(out_obj_xyz)                                                                                              # [batch,    out_length, obj_padded_length, num_obj_markers, obj_dim] uncentered
                handled_obj_xyz = handled_obj_xyz[:,None,:,:,:].repeat(1,2,1,1,1,1)                                                                     # [batch, 2, out_length, obj_padded_length, num_obj_markers, obj_dim] 
                handled_obj_xyz = torch.reshape(handled_obj_xyz,[self.batch_size*2, self.out_length, self.obj_padded_length, num_obj_markers, obj_dim]) # [batch* 2, out_length, obj_padded_length, num_obj_markers, obj_dim]
                handled_obj_xyz = torch.gather(handled_obj_xyz,2,handled_obj_idxs)                                                                      # [batch* 2, out_length,                 1, num_obj_markers, obj_dim]
                handled_obj_xyz = handled_obj_xyz[:,:,0,:,:]
                
                """
                handled_obj_xyz = torch.clone(out_obj_xyz)                                                                                              # [batch,    out_length, obj_padded_length, num_obj_markers, obj_dim] uncentered
                handled_obj_xyz = handled_obj_xyz[:,None,:,:,:].repeat(1,2,1,1,1,1)                                                                     # [batch, 2, out_length, obj_padded_length, num_obj_markers, obj_dim] 
                handled_obj_xyz = torch.reshape(handled_obj_xyz,[self.batch_size*2, self.out_length, self.obj_padded_length, num_obj_markers, obj_dim]) # [batch* 2, out_length, obj_padded_length, num_obj_markers, obj_dim]
                handled_obj_xyz = torch.gather(handled_obj_xyz,2,handled_obj_idxs)                                                                      # [batch* 2, out_length,                 1, num_obj_markers, obj_dim]
                handled_obj_xyz = handled_obj_xyz[:,:,0,:,:]
                #handled_obj_xyz = torch.squeeze(handled_obj_xyz)                                                                                        # [batch* 2, out_length,                    num_obj_markers, obj_dim] dont squeeze due to KIT RGBD
                """
            else:
                print("Unknown self.grab_net_agg_mode:".format(self.grab_net_agg_mode))
                sys.exit()
                
            # masks out_xyz
            # centers out_xyz and handled_obj_xyz wrt the handled_obj_pos
            agg_data = prepare_pred_data_for_out_grab_net(self, out_xyz, handled_obj_xyz, out_data["handled_obj_ohs"], self.out_length)
            decoder_pem_agg = self.decoder_pem(human_h=None, data=agg_data, mode=mode)
            #print(decoder_pem_agg["pred_grab_net_xyz"].shape)       # [batch*2, out_length, 53, 3]
            #print(decoder_pem_agg["pred_grab_net_finger"].shape)    # [batch*2, out_length, 19]
            decoder_pem_agg["pred_grab_net_xyz"]    = torch.reshape(decoder_pem_agg["pred_grab_net_xyz"],[self.batch_size, 2, self.out_length, num_body_joints, body_dim])
            if decoder_pem_agg["pred_grab_net_finger"] is not None:
                decoder_pem_agg["pred_grab_net_finger"] = torch.reshape(decoder_pem_agg["pred_grab_net_finger"],[self.batch_size, 2, self.out_length, finger_dim])
            
            """
            SANITY CHECKS
            - compare finger mask to object handled idx
            """
            
            """
            x = data["out_handled_obj_idxs"]
            y = out_data["mask"]
            y = torch.reshape(y,[self.batch_size,2,self.out_length])
            print(x.shape)
            print(y.shape)
            for b in range(self.batch_size):
                for h in range(2):
                    print(x[b,h])
                    print(y[b,h])
                    print()
            sys.exit()
            """
            
            return_data = {**return_data,

                            # inp human data
                            "PRED_inp_xyz":data["inp_xyz"], "PRED_inp_finger":data["inp_finger"], # for visualization, so it does not overwrite the reconstruction module's output
                            "TRUE_inp_xyz":data["inp_xyz"], "TRUE_inp_finger":data["inp_finger"], # for visualization, so it does not overwrite the reconstruction module's output
                            
                            # inp object data
                            "PRED_inp_obj_xyz":data["inp_obj_xyz"], # for visualization, so it does not overwrite the reconstruction module's output
                            "TRUE_inp_obj_xyz":data["inp_obj_xyz"], # for visualization, so it does not overwrite the reconstruction module's output
                            
                            # free_net xyz                           # free_net finger
                            # - no mask needed                       # - confidence
                            "pred_free_net_out_xyz":out_xyz,         "pred_free_net_out_finger":out_finger,
                            "true_free_net_out_xyz":data["out_xyz"], "true_free_net_out_finger":data["out_finger"],
                                                        
                            # out object data
                            # - no mask needed
                            "pred_out_obj_xyz":out_obj_xyz,
                            "true_out_obj_xyz":data["out_obj_xyz"],
                                 "out_obj_xyz_mask":obj_xyz_mask,  
                         
                            # left and right omm object classification
                            # - no mask needed since we assign the last element for when no object is being handled
                            "pred_lhand_out_handled_obj":torch.permute(out_p_log[0],[0,2,1]),   "pred_rhand_out_handled_obj":torch.permute(out_p_log[1],[0,2,1]),
                            "true_lhand_out_handled_obj":out_handled_obj_idxs[:,0],             "true_rhand_out_handled_obj":out_handled_obj_idxs[:,1], # [batch, inp_length]
                            
                            # grab_net xyz                                                                  # grab_net finger
                            # - mask needed if we are computing the loss over the left / right arm          # - mask for when the person is not handling any object
                            # - mask for when the person is not handling any object                         # - confidence for kit rgbd
                            #
                            "pred_grab_net_out_xyz":decoder_pem_out["pred_grab_net_xyz"],                   "pred_grab_net_out_finger":decoder_pem_out["pred_grab_net_finger"],
                            "true_grab_net_out_xyz":xyz,                                                    "true_grab_net_out_finger":finger,
                                 "grab_net_out_xyz_mask":xyz_mask,                                               "grab_net_out_finger_mask":finger_mask,
                                                                                                                 "grab_net_out_finger_confidence":grab_net_out_finger_confidence,
                            
                            # out refined human data                                                        # grab_net finger
                            # - mask needed if we are computing the loss over the left / right arm          # - mask for when the person is not handling any object
                            # - mask needed for when the person is not holding onto an object               # - confidence for kit rgbd
                            "pred_grab_net_agg_xyz":decoder_pem_agg["pred_grab_net_xyz"],                   "pred_grab_net_agg_finger":decoder_pem_agg["pred_grab_net_finger"],
                            "true_grab_net_agg_xyz":xyz,                                                    "true_grab_net_agg_finger":finger,
                                 "grab_net_agg_xyz_mask":xyz_mask,                                               "grab_net_agg_finger_mask":finger_mask,
                                                                                                                 "grab_net_agg_finger_confidence":grab_net_out_finger_confidence
                            }
                            
                                
                                
            t2 = time.time()
            #print("Feedforward:", t2-t1)
            return return_data
                
        # # # # # # # # # # # #
        # TransformerDecoder  #
        # # # # # # # # # # # #
        
        if self.temporal_decoder_type == "TransformerDecoder":
        
            """
            form human and object data
            """
        
            inp_data = get_data(self, data, section="inp", center=False)
        
            # obj_data
            obj_data = torch.cat([inp_data[k] for k in self.out_object_encoder_data_type],dim=-1)   # [batch, inp_length, obj_padded_length, -1]
            obj_data = self.out_object_encoder(obj_data)                                            # [batch, inp_length, obj_padded_length, -1]
            
            # human_data
            human_data = torch.cat([inp_data[k] for k in self.out_human_encoder_data_type],dim=-1)  # [batch, inp_length,    -1]
            human_data = self.out_human_encoder(human_data)                                         # [batch, inp_length,    -1]
            human_data = torch.unsqueeze(human_data,dim=2)                                          # [batch, inp_length, 1, -1]
            
            # concatenate obj_data and human_data
            obj_human_data = torch.cat([obj_data,human_data],dim=2)                                 # [batch, inp_length, obj_body_padded_length, -1]
            
            # concatenate inp and out data (zero padped data)
            zeros = torch.zeros([self.batch_size,self.out_length,obj_body_padded_length,obj_human_data.shape[-1]]).to(device=torch.cuda.current_device()) # [batch, out_length, obj_body_padded_length, -1]
            obj_human_data = torch.cat([obj_human_data,zeros],dim=1)    # [batch, inp+out_length, obj_body_padded_length, -1]
            
            # object and human ids
            all_ids = data["obj_human_ids"]                                                 # [batch,                 obj_body_padded_length]
            all_ids = all_ids[:,None,:].repeat(1,self.inp_length+self.out_length,1)         # [batch, inp+out_length, obj_body_padded_length]
            all_ids = self.out_object_label_embedder(all_ids.type(torch.cuda.LongTensor))   # [batch, inp+out_length, obj_body_padded_length, -1]
            
            # positional data
            inp_out_length = [self.inp_length+self.out_length for _ in range(self.batch_size)]
            pos_emb = cosine_positional_embedder(inp_out_length, self.inp_length+self.out_length, self.position_embedder_units) # [batch, inp+out_length, -1]
            pos_emb = torch.unsqueeze(pos_emb,2).repeat(1,1,self.obj_body_padded_length,1)                                      # [batch, inp+out_length, obj_body_padded_length, -1]
            
            # form key
            key = obj_human_data + all_ids + pos_emb                    # [batch, inp+out_length, obj_body_padded_length, -1]
            #key = torch.reshape(key,[self.batch_size,-1,key.shape[-1]]) # [batch, inp+out_length* obj_body_padded_length, -1]
                
            """
            form mask
            """
            
            # attend 0 ignore 1
            mask = torch.zeros([self.batch_size,self.inp_length+self.out_length,self.obj_body_padded_length]) # [batch, inp+out_length, obj_body_padded_length]
            for i,obj_ids_unpadded_length_i in enumerate(obj_ids_unpadded_length):
                mask[i,:,obj_ids_unpadded_length_i:] = 1
            # attend to body
            mask[:,:,-1] = 0
            #mask = torch.reshape(mask,[self.batch_size,-1,self.obj_body_padded_length]) # [batch, inp+out_length* obj_body_padded_length]
            mask = mask.type(torch.cuda.BoolTensor)
            
            # why no reshape?
            
            """
            transformer feedforward
            """
            
            h_out = self.temporal_encoder(key=key, mask=mask)    # [batch, inp+out_length, obj_body_padded_length, -1]
            
            if self.residual_features == 1:
                h_out = h_out + key
                        
            """
            human decoder
            """
            
            xyz    = self.out_human_decoder(h_out[:,:,-1,:])                                        # [batch, inp+out_length, 197] only the final human node
            human  = xyz[:,:,:159]                                                                  # [batch, inp+out_length, 159]
            human  = torch.reshape(human,[self.batch_size,self.inp_length+self.out_length,53,3])    # [batch, inp+out_length, 53, 3]        
            #finger = xyz[:,:,159:]                                                                  # [batch, inp+out_length, 38]
            #finger = torch.reshape(finger,[self.batch_size,self.inp_length+self.out_length,2,19])   # [batch, inp+out_length, 2, 19]
                                                
            # predict velocity          
            if self.predict_body_velocity == 1:
                human[:,self.inp_length:] = human[:,self.inp_length:] + data["inp_xyz"][:,-1][:,None,:,:]
            
            """
            object decoder
            """
            
            obj_h   = h_out[:,:,:-1,:]                                                                                          # [batch, inp+out_length, obj_padded_length, -1] all except the final human node
            if self.omm_object_decoder_data_type == ["obj_xyz","obj_ohs"]:
                obj_ohs = torch.unsqueeze(data["obj_ohs"],dim=1).repeat(1,self.inp_length+self.out_length,1,1)                  # [batch, inp+out_length, obj_padded_length, num_obj_classes]
                obj_h   = torch.cat([obj_h,obj_ohs],dim=-1)                                                                     # [batch, inp+out_length, obj_padded_length, -1]
            obj_xyz = self.out_object_decoder(obj_h)                                                                            # [batch, inp+out_length, obj_padded_length, 12]
            obj_xyz = torch.reshape(obj_xyz,[self.batch_size, self.inp_length+self.out_length, self.obj_padded_length, 4, 3])   # [batch, inp+out_length, obj_padded_length, 4, 3]
            
            # mask for loss computation
            obj_xyz_mask = torch.ones(obj_xyz.shape).to(device=torch.cuda.current_device())
            for i,obj_ids_unpadded_length_i in enumerate(data["obj_ids_unpadded_length"]):
                obj_xyz_mask[i,:,obj_ids_unpadded_length_i:] = 0
        
            # predict velocity
            if self.predict_object_velocity == 1:
                obj_xyz[:,self.inp_length:] = obj_xyz[:,self.inp_length:] + data["inp_obj_xyz"][:,-1][:,None,:,:,:]
            
            return_data = {**return_data,
            
                            # inp human data
                            "pred_inp_xyz":data["inp_xyz"], #"pred_inp_finger":finger[:,:self.inp_length],
                            "true_inp_xyz":data["inp_xyz"], #"true_inp_finger":data["inp_finger"],
                            
                            # out human data
                            "pred_out_xyz":human[:,self.inp_length:], #"pred_out_finger":finger[:,self.inp_length:],
                            "true_out_xyz":data["out_xyz"],           #"true_out_finger":data["out_finger"],
                            
                            # inp object data
                            "pred_inp_obj_xyz":data["inp_obj_xyz"],
                            "true_inp_obj_xyz":data["inp_obj_xyz"],
                                 "inp_obj_xyz_mask":obj_xyz_mask[:,:self.inp_length],
                            
                            # out object data
                            "pred_out_obj_xyz":obj_xyz[:,self.inp_length:],
                            "true_out_obj_xyz":data["out_obj_xyz"],
                                 "out_obj_xyz_mask":obj_xyz_mask[:,self.inp_length:]}                            
                            
            return return_data