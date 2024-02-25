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

def project_wrist_to_2d(wrist_xyz, xyz_scale, table_center, rx, cx, cy, fx, fy):
    
    batch_size = wrist_xyz.shape[0]
    
    # undo scaling
    wrist_xy = wrist_xyz / xyz_scale               # [batch, hands=2, 3]
    
    # denormalize
    wrist_xy = wrist_xy + table_center[:,None,:]    # [batch, hands=2, 3]
    wrist_xy = torch.permute(wrist_xy,[0,2,1])      # [batch, 3, hands=2]
    wrist_xy = torch.matmul(rx, wrist_xy)           # [batch, 3, hands=2]
    wrist_xy = torch.permute(wrist_xy,[0,2,1])      # [batch, hands=2, 3]
    
    # project to image
    x = wrist_xy[:,:,0] # [batch, hands=2]
    y = wrist_xy[:,:,1] # [batch, hands=2]
    z = wrist_xy[:,:,2] # [batch, hands=2]
    x, y = project_to_image(x, y, z, cx, cy, fx, fy)                    # [batch], [batch]
    x /= torch.tensor([640,480]).to(device=torch.cuda.current_device())
    y /= torch.tensor([640,480]).to(device=torch.cuda.current_device()) 
    wrist_xy = torch.stack([x,y],dim=1)                                 # [batch, 2, 2]
    wrist_xy = torch.reshape(wrist_xy,[batch_size, 2, 2])          # [batch, 2, 2]
    
    return wrist_xy

def project_to_image(x, y, z, cx, cy, fx, fy):
        
    z = z * -1
    x =  (x * fx / z) + cx
    y = -(y * fy / z) + cy
    return x,y    
    
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
        self.decoder_omm = KITRGBDObjectMotionModule(args,num_classifiers=2)

        # pem
        self.decoder_pem = KITRGBDPoseEnsembleModule(args)
    
    def forward(self, data, mode):
        assert mode == "train" or mode == "val"
        return_data = {}
        t1 = time.time()
                
        # variables
        num_obj_wrist_classes = self.num_obj_wrist_classes
        obj_wrist_padded_length = self.obj_wrist_padded_length
        
        #num_body_joints = self.num_body_joints
        #body_dim        = self.body_dim
        #num_obj_markers = self.num_obj_markers
        #obj_dim         = self.obj_dim
        #num_hands       = self.num_hands
        #finger_dim      = self.finger_dim
        
        # teacher forcing, 0 uses ground truth, 1 uses predictions
        rand = torch.rand(self.batch_size)
        teacher_force = rand > self.teacher_force_ratio
        teacher_force = teacher_force.long()
        #print(teacher_force)
        #sys.exit()
                                
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
                adj = 1 - torch.eye(obj_wrist_padded_length)
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
            inp_data = get_kit_rgbd_data(self, data, section="inp", center=False)
            
            # obj_data
            obj_data   = torch.cat([inp_data[k] for k in self.inp_object_encoder_data_type],dim=-1) # [batch, inp_length, obj_padded_length, -1]
            wrist_data = torch.cat([inp_data[k] for k in self.inp_human_encoder_data_type],dim=-1)  # [batch, inp_length, 2,                 -1]
            obj_wrist_data = torch.cat([obj_data,wrist_data],axis=2)                                # [batch, inp_length, obj_wrist_padded_length, -1]
            obj_wrist_data = self.out_object_encoder(obj_wrist_data)                                # [batch, inp_length, obj_wrist_padded_length, -1]
                                                
            """
            bi-grnn feedforward
            """
                  
            h_out = self.temporal_decoder(timesteps=self.inp_length, data=obj_wrist_data, edge_index=edge_index)  # [batch, inp_length, obj_wrist_padded_length, -1]
            h_out = h_out[:,-1]                                                                                   # [batch,             obj_wrist_padded_length, -1]
                        
            # # # # # # # # # #
            # Decoding Phase  #
            # # # # # # # # # #
            
            prev_free_net_xyz   = data["inp_xyz"][:,-1]         # [batch, 15, 2]
            prev_grab_net_xyz   = data["inp_xyz"][:,-1]         # [batch, 15, 2]
            prev_wrist_xyz      = data["inp_wrist_xyz"][:,-1]   # [batch, hands=2, 3]
            prev_obj_xyz        = data["inp_obj_xyz"][:,-1]     # [batch, obj_padded_length, 1, 3]
                                                
            pred_free_net_out_xyz   = []
            pred_grab_net_out_xyz   = []
            out_wrist_xyz           = []
            out_obj_xyz             = []
            out_p_log               = []
            for i in range(self.out_length):
            
                # ranked prediction module
                decoder_omm_out = self.decoder_omm(h=h_out[:,None,:,:], obj_ohs=data["obj_ohs"], wrist_ohs=data["wrist_ohs"], mode=mode)
                                
                """
                object classification scores
                """
                
                # collect object classification scores
                p_log = decoder_omm_out["p_log"] # [2, batch, length=1, obj_wrist_padded_length]
                p_log = torch.squeeze(p_log)     # [2, batch, obj_wrist_padded_length]
                out_p_log.append(p_log)
                
                """
                object coordinates
                """
                                
                # collect object coordinates
                obj_xyz = decoder_omm_out["pred_obj_wrist_xyz"][:,:,:-2]    # [batch, length=1, obj_padded_length, 1, 3]
                obj_xyz = obj_xyz[:,0]                                      # [batch,           obj_padded_length, 1, 3]
                if self.predict_object_velocity == 1:
                    obj_xyz      = obj_xyz + prev_obj_xyz
                    prev_obj_xyz = obj_xyz
                out_obj_xyz.append(obj_xyz)
                                
                # teacher force for next iteration
                pred_inp_obj_xyz = obj_xyz
                true_inp_obj_xyz = data["out_obj_xyz"][:,i]
                inp_obj_xyz      = torch.stack([true_inp_obj_xyz,pred_inp_obj_xyz],dim=1)
                inp_obj_xyz      = torch.stack([p[tf] for p,tf in zip(inp_obj_xyz,teacher_force)])  # [batch, obj_padded_length, 1, 3]
                
                """
                wrist coordinates
                """
                
                # collect wrist coordinates
                wrist_xyz = decoder_omm_out["pred_obj_wrist_xyz"][:,:,-2:]  # [batch, length=1, hands=2, 1, 3]
                wrist_xyz = wrist_xyz[:,0]                                  # [batch,           hands=2, 1, 3]
                wrist_xyz = torch.squeeze(wrist_xyz)                        # [batch,           hands=2,    3]
                if self.predict_object_velocity == 1:
                    wrist_xyz      = wrist_xyz + prev_wrist_xyz
                    prev_wrist_xyz = wrist_xyz
                out_wrist_xyz.append(wrist_xyz)
                
                # teacher force for next iteration
                pred_inp_wrist_xyz = wrist_xyz
                true_inp_wrist_xyz = data["out_wrist_xyz"][:,i]
                inp_wrist_xyz = torch.stack([true_inp_wrist_xyz,pred_inp_wrist_xyz],dim=1)
                inp_wrist_xyz = torch.stack([p[tf] for p,tf in zip(inp_wrist_xyz,teacher_force)])  # [batch, hands=2, 3]
                                                                
                """
                human coordinates
                """
                                    
                # # # # # # #
                # free_net  #
                # # # # # # #
                                   
                # project pred_wrist_xyz to 2D
                pred_wrist_xy = project_wrist_to_2d(inp_wrist_xyz, self.xyz_scale, data["table_center"], data["rx"], data["cx"], data["cy"], data["fx"], data["fy"])
                
                # predict human coordinates conditioned on the predicted wrist
                free_net_inp = torch.cat([torch.reshape(prev_free_net_xyz,[self.batch_size,-1]),torch.reshape(pred_wrist_xy,[self.batch_size,-1])],axis=-1)
                free_net_xyz = self.decoder_pem.free_net_human_decoder(free_net_inp)    # [batch, 30]
                free_net_xyz = torch.reshape(free_net_xyz,[self.batch_size,15,2])       # [batch, 15, 2]
                if self.predict_body_velocity == 1:
                    free_net_xyz      = free_net_xyz + prev_free_net_xyz
                    prev_free_net_xyz = free_net_xyz
                pred_free_net_out_xyz.append(free_net_xyz)
                                
                # teacher force for next iteration
                pred_free_net_inp_xyz = free_net_xyz
                true_free_net_inp_xyz = data["out_xyz"][:,i]
                free_net_inp_xyz    = torch.stack([true_free_net_inp_xyz,pred_free_net_inp_xyz],dim=1)
                free_net_inp_xyz    = torch.stack([p[tf] for p,tf in zip(free_net_inp_xyz,teacher_force)])
                  
                # # # # # # #
                # grab_net  #
                # # # # # # #
                
                # project true_wrist_xyz to 2D
                true_wrist_xy = project_wrist_to_2d(data["out_wrist_xyz"][:,i], self.xyz_scale, data["table_center"], data["rx"], data["cx"], data["cy"], data["fx"], data["fy"])
                
                # predict human coordinates conditioned on the ground truth wrist positions
                grab_net_inp = torch.cat([torch.reshape(prev_grab_net_xyz,[self.batch_size,-1]),torch.reshape(true_wrist_xy,[self.batch_size,-1])],axis=-1)
                grab_net_xyz = self.decoder_pem.grab_net_human_decoder(grab_net_inp)    # [batch, 30]
                grab_net_xyz = torch.reshape(grab_net_xyz,[self.batch_size,15,2])       # [batch, 15, 2]
                if self.predict_body_velocity == 1:
                    grab_net_xyz      = grab_net_xyz + prev_grab_net_xyz
                    prev_grab_net_xyz = grab_net_xyz
                pred_grab_net_out_xyz.append(grab_net_xyz)
                
                # teacher force for next iteration
                pred_grab_net_inp_xyz = grab_net_xyz
                true_grab_net_inp_xyz = data["out_xyz"][:,i]
                grab_net_inp_xyz    = torch.stack([true_grab_net_inp_xyz,pred_grab_net_inp_xyz],dim=1)
                grab_net_inp_xyz    = torch.stack([p[tf] for p,tf in zip(grab_net_inp_xyz,teacher_force)])
                
                """
                form input for grnn
                """
                
                obj_data        = torch.reshape(inp_obj_xyz,[self.batch_size,self.obj_padded_length,-1])    # [batch, obj_padded_length,       3]
                wrist_data      = torch.reshape(inp_wrist_xyz,[self.batch_size,2,-1])                       # [batch, 2,                       3]
                obj_wrist_data  = torch.cat([obj_data,wrist_data],axis=1)                                   # [batch, obj_wrist_padded_length, 3]
                obj_wrist_ohs   = torch.cat([data["obj_ohs"],data["wrist_ohs"]],axis=1)                     # [batch, obj_wrist_padded_length, num_obj_wrist_classes]
                obj_wrist_data  = torch.cat([obj_wrist_data,obj_wrist_ohs],axis=-1)                         # [batch, obj_wrist_padded_length, 3 + num_obj_classes]
                obj_wrist_data  = self.out_object_encoder(obj_wrist_data)                                   # [batch, obj_wrist_padded_length, -1]
                obj_wrist_data  = obj_wrist_data[:,None,:,:]                                                # [batch, length=1, obj_wrist_padded_length, -1]
                            
                """
                grnn feedforward
                """
            
                h_out = self.temporal_encoder(data=obj_wrist_data, edge_index=edge_index, h=h_out)  # [batch, obj_wrist_padded_length, -1]
            
            # stack
            pred_free_net_out_xyz   = torch.stack(pred_free_net_out_xyz)    # [out_length, batch, 15 ,2]
            pred_grab_net_out_xyz   = torch.stack(pred_grab_net_out_xyz)    # [out_length, batch, 15 ,2]
            out_wrist_xyz           = torch.stack(out_wrist_xyz)            # [out_length, batch, hands=2, 3]
            out_obj_xyz             = torch.stack(out_obj_xyz)              # [out_length, batch, obj_padded_length, 1, 3]
            out_p_log               = torch.stack(out_p_log,dim=2)          # [2, batch, out_length, obj_wrist_padded_length]
                     
            # permute
            pred_free_net_out_xyz   = torch.permute(pred_free_net_out_xyz,[1,0,2,3])    # [batch, out_length, 15 ,2]
            pred_grab_net_out_xyz   = torch.permute(pred_grab_net_out_xyz,[1,0,2,3])    # [batch, out_length, 15 ,2]
            out_wrist_xyz           = torch.permute(out_wrist_xyz,[1,0,2,3])            # [batch, out_length, 2, 3]
            out_obj_xyz             = torch.permute(out_obj_xyz,[1,0,2,3,4])            # [batch, out_length, obj_padded_length, 1, 3]
                                                
            # handled_obj_idxs ground truth
            out_handled_obj_idxs = torch.clone(data["out_handled_obj_idxs"])            # [batch, 2, out_length]
            out_handled_obj_idxs[out_handled_obj_idxs == -1] = self.obj_padded_length   # [batch, 2, out_length]
            
            # obj_xyz_mask for loss computation
            obj_xyz_mask = torch.ones(out_obj_xyz.shape).to(device=torch.cuda.current_device())
            for i,obj_ids_unpadded_length_i in enumerate(data["obj_ids_unpadded_length"]):
                obj_xyz_mask[i,:,obj_ids_unpadded_length_i:] = 0
                        
            return_data = {**return_data,
                            
                            # free_net xyz
                            "pred_free_net_out_xyz":pred_free_net_out_xyz,
                            "true_free_net_out_xyz":data["out_xyz"],
                                                        
                            # out object data
                            "pred_out_obj_xyz":out_obj_xyz,
                            "true_out_obj_xyz":data["out_obj_xyz"],
                                 "out_obj_xyz_mask":obj_xyz_mask,  
                         
                            # out wrist data
                            "pred_out_wrist_xyz":out_wrist_xyz,
                            "true_out_wrist_xyz":data["out_wrist_xyz"],
                         
                            # left and right omm object classification
                            # - no mask needed since we assign the last element for when no object is being handled
                            "pred_lhand_out_handled_obj":out_p_log[0],              "pred_rhand_out_handled_obj":out_p_log[1],
                            "true_lhand_out_handled_obj":out_handled_obj_idxs[:,0], "true_rhand_out_handled_obj":out_handled_obj_idxs[:,1], # [batch, inp_length]
                            
                            # grab_net xyz
                            "pred_grab_net_out_xyz":pred_grab_net_out_xyz,
                            "true_grab_net_out_xyz":data["out_xyz"]
                            }
            
            #for k,v in return_data.items():
            #    print(k, v.shape)
            #sys.exit()
            
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