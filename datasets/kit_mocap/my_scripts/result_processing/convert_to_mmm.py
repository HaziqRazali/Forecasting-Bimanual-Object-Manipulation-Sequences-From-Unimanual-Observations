import os
import ast
import sys
import json
import argparse
import numpy as np
import xml.dom.minidom
import xml.etree.ElementTree as ET

from pathlib import Path
from glob import glob
sys.path.append(os.path.join(os.path.expanduser("~"),"Forecasting-Bimanual-Object-Manipulation-Sequences-From-Unimanual-Observations","datasets","kit_mocap","my_scripts"))
from utils_data import *
from utils_processing import *

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_root', default=os.path.join(os.path.expanduser("~"),"Keystate-Forecasting","results"), type=str) # "../../../../Keystate-Forecasting/results/"
    parser.add_argument('--result_name', required=True, type=str)
    args = parser.parse_args()
           
    print()
    print("============================================================")
    print(os.path.join(args.result_root,args.result_name,"*"))
    print()
    sequences = sorted(glob(os.path.join(args.result_root,args.result_name,"*")))
    for sequence in sequences:
        print(sequence)
    print("============================================================")
    print()
    
    kn_idx = 0
    for sequence in sequences:   
        # print(sequence) # ../../../../Keystate-Forecasting/results/action2pose/agraph/oh\Cut_files_motions_3021_Cut1_c_0_05cm_01    
        
        files = sorted(glob(os.path.join(sequence,"*")))
        for file in files:
            # print(file) # ../../../../Keystate-Forecasting/results/action2pose/agraph/oh\Cut_files_motions_3021_Cut1_c_0_05cm_01\0000000000.json
                        
            # # # # # # #
            #           #
            # load data #
            #           #
            # # # # # # #
            
            data = json.load(open(file,"r"))
            #for k,v in data.items():
            #    print(k)
            
            for k,v in data.items():
                if "obj_names" not in k and "xyz_names" not in k and "task_components" not in k and \
                   "graph_encoder_units" not in k and "graph_encoder_activations" not in k and \
                   "graph_decoder_units" not in k and "graph_decoder_activations" not in k and \
                   "checkpoint_layer_names" not in k and type(v) == type(list([])):
                    data[k] = np.array(v)
                        
            # sequence name
            sequence = data["sequence"]
            main_action = data["main_action"]
                          
            # time
            inp_length = len(data["inp_frames"])
            out_length = len(data["out_frames"])
            print(inp_length, out_length)
            frames = np.concatenate([data["inp_frames"],data["out_frames"]])
            
            # center
            table_pos = data["obj_table_pos"][0:1] if "obj_table_pos" in data.keys() else None # [1, 3]
            table_rot = data["obj_table_rot"][0:1] if "obj_table_rot" in data.keys() else None # [1, 3]
            
            p = data["p"]
                                                
            # scale
            xyz_scale, kin_scale = data["xyz_scale"], data["kin_scale"]
            
            """
            process object data
            """
             
            # # # # # # #
            # metadata  #
            # # # # # # #
            
            # object mocap names
            obj_mocap_names = ast.literal_eval(data["obj_mocap_names"])     
                        
            # object path names
            obj_paths = ast.literal_eval(data["obj_paths"])
            true_obj_paths = obj_paths + ["/home/haziq/MMMTools/data/Model/Objects/kitchen_sideboard/kitchen_sideboard.xml"]
            pred_obj_paths   = obj_paths
            masked_obj_paths = obj_paths
                        
            # object names
            obj_names = ast.literal_eval(data["obj_names"]) if "obj_names" in data.keys() else None   
            true_obj_names = ["true_"+obj_name for obj_name in obj_names]+["true_kitchen_sideboard"]
            pred_obj_names   = ["pred_"+obj_name for obj_name in obj_names]
            masked_obj_names = ["masked_"+obj_name for obj_name in obj_names]
            print(obj_names)
                                                   
            # number of objects and padding
            num_objects = data["obj_xyz_unpadded_objects"]
            obj_padded_length      = data["obj_padded_length"]
            obj_body_padded_length = data["obj_body_padded_length"]        
            
            # # # # # # # # # #
            # object position #                   
            # # # # # # # # # #
            
            true_obj_pos = np.transpose(data["obj_pos"], (1,0,2)) if "obj_pos" in data.keys() else None # [n-1, len, 3]             
            pred_obj_pos = None
                                           
            # # # # # # # # # #
            # object xyz      #                   
            # # # # # # # # # # 
            
            # true and pred obj xyz 
            true_obj_xyz = np.concatenate([data["true_inp_obj_xyz"],data["true_out_obj_xyz"]],axis=0)
            true_obj_xyz = np.transpose(true_obj_xyz, (1,0,2,3))
            pred_obj_xyz = np.concatenate([data["pred_inp_obj_xyz"],data["pred_out_obj_xyz"]],axis=0)               # [inp+out length, n-1, num_markers, 3]
            pred_obj_xyz = np.transpose(pred_obj_xyz, (1,0,2,3))                                                    # [n-1, inp+out length, num_markers, 3]
            
            # masked obj xyz
            inp_masked_obj_xyz = data["inp_masked_obj_xyz"]                 # [inp length, obj_padded_length, num_markers, 3]
            inp_masked_obj_xyz = np.transpose(inp_masked_obj_xyz,(1,0,2,3)) # [obj_padded_length, inp length, num_markers, 3]
            inp_masked_obj_xyz = inp_masked_obj_xyz[:num_objects]           # [num_objects, inp length, num_markers, 3]
            #inp_masked_obj_xyz[2,:] = inp_masked_obj_xyz[2,0]
            
            # - no longer required
            #if main_action == "Cut":
            #    print("updating")
            #    pred_obj_xyz[2,:10] = (pred_obj_xyz[2,:10] + true_obj_xyz[2,:10]) / 2
            
            # - this was for corona mix
            #pred_obj_xyz = 0.7 * pred_obj_xyz + 0.3 * true_obj_xyz
            
            """
            aggregate obj xyz
            """
                
            # get handled obj
            pred_inp_handled_obj = data["pred_inp_handled_obj"]                                                       # [inp_length,    obj_body_padded_length]
            true_inp_handled_obj = data["true_inp_handled_obj"]                                                       # [inp_length]
            pred_out_handled_obj = np.stack([data["pred_lhand_out_handled_obj"],data["pred_rhand_out_handled_obj"]])  # [2, out_length, obj_body_padded_length]
            true_out_handled_obj = np.stack([data["true_lhand_out_handled_obj"],data["true_rhand_out_handled_obj"]])  # [2, out_length]
            #print(pred_inp_handled_obj.shape, true_inp_handled_obj.shape, pred_out_handled_obj.shape, true_out_handled_obj.shape)
                                            
            # convert pred idx to one hot
            pred_inp_handled_obj = np.argmax(pred_inp_handled_obj,axis=-1)                                              # [inp_length]
            pred_inp_handled_obj = one_hot(pred_inp_handled_obj, obj_body_padded_length)                                # [inp_length, obj_body_padded_length]
            pred_out_handled_obj = np.argmax(pred_out_handled_obj,axis=-1)                                              # [2, out_length]
            pred_out_handled_obj = np.stack([one_hot(x, obj_body_padded_length) for x in pred_out_handled_obj])         # [2, out_length, obj_body_padded_length]
            
            # convert true idx to one hot
            true_inp_handled_obj = one_hot(true_inp_handled_obj, obj_body_padded_length)                                # [inp_length, obj_body_padded_length]
            true_out_handled_obj = np.stack([one_hot(x, obj_body_padded_length) for x in true_out_handled_obj])         # [2, out_length, obj_body_padded_length]
            
            # remove padded objects
            pred_inp_handled_obj = pred_inp_handled_obj[:,:num_objects]                                                 # [inp_length, num_objects]
            true_inp_handled_obj = true_inp_handled_obj[:,:num_objects]                                                 # [inp_length, num_objects]
            pred_out_handled_obj = pred_out_handled_obj[:,:,:num_objects]                                               # [2, out_length, num_objects]
            true_out_handled_obj = true_out_handled_obj[:,:,:num_objects]                                               # [2, out_length, num_objects]
            
            # transpose so data is in same format as obj_xyz
            pred_inp_handled_obj = np.transpose(pred_inp_handled_obj,[1,0])                                             # [num_objects, inp_length]
            true_inp_handled_obj = np.transpose(true_inp_handled_obj,[1,0])                                             # [num_objects, inp_length]
            pred_out_handled_obj = np.transpose(pred_out_handled_obj,[2,1,0])                                           # [num_objects, out_length, 2]
            true_out_handled_obj = np.transpose(true_out_handled_obj,[2,1,0])                                           # [num_objects, out_length, 2]
            
            # sum out idx
            pred_out_handled_obj = np.sum(pred_out_handled_obj,axis=-1)                                                 # [num_objects, out_length]
            pred_out_handled_obj[pred_out_handled_obj > 1] = 1
            true_out_handled_obj = np.sum(true_out_handled_obj,axis=-1)                                                 # [num_objects, out_length]
            true_out_handled_obj[true_out_handled_obj > 1] = 1
                        
            # concatenate
            true_handled_obj = np.concatenate([true_inp_handled_obj,true_out_handled_obj],axis=1)                       # [num_objects, inp+out length]
            pred_handled_obj = np.concatenate([pred_inp_handled_obj,pred_out_handled_obj],axis=1)                       # [num_objects, inp+out length]
            true_handled_obj = true_handled_obj[:,:,None,None]
            pred_handled_obj = pred_handled_obj[:,:,None,None]
                                        
            #x = object_aggregator(true_obj_xyz, true_obj_xyz, true_handled_obj, inp_length, out_length)
            pred_obj_xyz = object_aggregator_vis(true_obj_xyz, pred_obj_xyz, pred_handled_obj, inp_length, out_length)
                                               
            # # # # # # # # # #
            # object rotation #                   
            # # # # # # # # # #
            
            true_obj_rot = np.transpose(data["obj_rot"],(1,0,2))
            true_obj_rot = np.insert(true_obj_rot, len(true_obj_rot), np.expand_dims(table_rot,0), axis=0)
            pred_obj_rot = None
                                       
            # # # # # # # # # # #
            # unprocess object  #      
            # # # # # # # # # # #
            
            # unscale mocap before adding the table center
            true_obj_xyz = unprocess_obj(true_obj_xyz, table_pos, xyz_scale)
            pred_obj_xyz = unprocess_obj(pred_obj_xyz, table_pos, xyz_scale)
            inp_masked_obj_xyz = unprocess_obj(inp_masked_obj_xyz, table_pos, xyz_scale)
            
            # unscale centroid before adding the table center
            true_obj_pos = unprocess_obj(true_obj_pos, table_pos, xyz_scale) # [n-1, len, 3]
            pred_obj_pos = unprocess_obj(pred_obj_pos, table_pos, xyz_scale) # [n-1, len, 3]
            
            # insert table position
            true_obj_pos = np.insert(true_obj_pos, len(true_obj_pos), np.expand_dims(table_pos,0), axis=0)
            pred_obj_pos = None #np.insert(pred_obj_pos, -1, np.expand_dims(table_pos,0), axis=0)  if pred_obj_pos is not None else pred_obj_pos # [n,   len, 3] or None
            
            """
            subject metadata
            """
            
            subject_id          = data["subject_id"]
            subject_height      = data["subject_height"]
            subject_mass        = data["subject_mass"]
            subject_hand_length = data["subject_hand_length"]
            xyz_names       = ast.literal_eval(data["xyz_names"])
            kin_names       = ast.literal_eval(data["kin_names"])
            finger_names    = ast.literal_eval(data["finger_names"])
            lfinger_names   = finger_names[:19]
            rfinger_names   = finger_names[19:]
            
            """
            
            process human pose
            
            """
            
            # position and rotation
            pos = data["pos"] # [len, 3]
            pos = unprocess_pose(pos, table_pos, 1)
            rot = data["rot"] # [len, 3]
            
            # keys
            keys = {}
            # inp keys
            keys["true_inp_xyz"] = "true_free_net_inp_xyz" #"TRUE_inp_xyz"
            keys["pred_inp_xyz"] = "pred_free_net_inp_xyz" #"PRED_inp_xyz"
            # out keys
            keys["true_out_xyz"] = "true_free_net_out_xyz"
            keys["pred_out_xyz"] = "pred_free_net_out_xyz"
                                                           
            # get human pose
            true_xyz = np.concatenate([data[keys["true_inp_xyz"]],data[keys["true_out_xyz"]]],axis=0)   # [inp+out length, 53, 3]
            pred_xyz = np.concatenate([data[keys["pred_inp_xyz"]],data[keys["pred_out_xyz"]]],axis=0)   # [inp+out length, 53, 3]            
            # need to replace non-masked markers from the input sequence with the ground truth !
            # - replace all except right arm
            if p == 1:
                non_masked_mocap_idxs = [i for i in range(53) if i not in data["r_arm_mocap_idxs"]]
                pred_xyz[:inp_length,non_masked_mocap_idxs] = np.copy(true_xyz[:inp_length,non_masked_mocap_idxs])
            if p == 0:
                non_masked_mocap_idxs = [i for i in range(53) if i not in data["l_arm_mocap_idxs"]]
                pred_xyz[:inp_length,non_masked_mocap_idxs] = np.copy(true_xyz[:inp_length,non_masked_mocap_idxs])
            pred_refined_xyz = np.copy(pred_xyz)                                                        # [inp+out length, 53, 3]
                        
            # # # # # # # # # # # # # # # #
            # update with refined inputs  #
            # # # # # # # # # # # # # # # # 
            
            has_pred_grab_net_inp_xyz = any(["pred_grab_net_inp_xyz" in k for k in data.keys()])
            if has_pred_grab_net_inp_xyz == 1:
            
                inp_length                  = data["inp_length"]
                inp_xyz_mask_idxs           = data["inp_xyz_mask_idxs"]                         # [3  5 10 13 14 15 25 26] or [27 29 34 37 38 39 49 50]
                true_inp_handled_obj_idxs   = data["inp_handled_obj_idxs"]                      # [inp_length] ground truth
                pred_inp_handled_obj_idxs   = data["pred_inp_handled_obj"]                      # [inp_length, obj_body_padded_length]
                pred_inp_handled_obj_idxs   = np.argmax(pred_inp_handled_obj_idxs,axis=-1)      # [inp_length]
                pred_inp_handled_obj_idxs[pred_inp_handled_obj_idxs == obj_padded_length] = -1  # [inp_length]
                pred_grab_net_inp_xyz       = data["pred_grab_net_inp_xyz"]                     # [inp_length, 53, 3]
                
                #print(true_inp_handled_obj_idxs)
                #print(pred_inp_handled_obj_idxs)
                #input()
                                
                if "pred_grab_net_inp_wrist_xyz" in data.keys():    
                    inp_finger_mask_idx         = data["inp_finger_mask_idxs"]                  # 0 or 1
                    pred_grab_net_inp_wrist_xyz = data["pred_grab_net_inp_wrist_xyz"]
                    hand_xyz_dims               = np.array(data["hand_xyz_dims"])               # [13 14 15 25 26 37 38 39 49 50]
                    hand_xyz_dims               = np.reshape(hand_xyz_dims,[2,-1])          
                    hand_xyz_dims               = hand_xyz_dims[inp_finger_mask_idx]
                    pred_grab_net_inp_xyz[:,hand_xyz_dims] = pred_grab_net_inp_wrist_xyz        # replace
                
                # refine predictions
                for i in range(inp_length):
                    if true_inp_handled_obj_idxs[i] != -1:
                        pred_refined_xyz[i,inp_xyz_mask_idxs] = pred_grab_net_inp_xyz[i,inp_xyz_mask_idxs]
                        
            # # # # # # # # # # # # # # # #
            # update with refined outputs #
            # # # # # # # # # # # # # # # #
            has_pred_grab_net_agg_xyz = any(["pred_grab_net_agg_xyz" in k for k in data.keys()])
            if has_pred_grab_net_agg_xyz == 1:
                        
                inp_length = data["inp_length"]
                out_length = data["out_length"]
                l_arm_mocap_idxs = data["l_arm_mocap_idxs"]             # [ 3  5 10 13 14 15 25 26]
                r_arm_mocap_idxs = data["r_arm_mocap_idxs"]             # [27 29 34 37 38 39 49 50]
                out_handled_obj_idxs = data["out_handled_obj_idxs"]     # [2, out_length] ground truth
                pred_grab_net_agg_xyz = data["pred_grab_net_agg_xyz"]   # [2, out_length, 53, 3]
                                                     
                """
                if "pred_grab_net_agg_wrist_xyz" in data.keys():
                    pred_grab_net_agg_wrist_xyz = data["pred_grab_net_agg_wrist_xyz"] # [2, out_length, 5, 3]
                    hand_xyz_dims = np.array(data["hand_xyz_dims"])
                    hand_xyz_dims = np.reshape(hand_xyz_dims,[2,-1])
                    
                    # replace left wrist
                    pred_grab_net_agg_xyz[0][:,hand_xyz_dims[0]] = pred_grab_net_agg_wrist_xyz[0]
                    # replace right wrist
                    pred_grab_net_agg_xyz[1][:,hand_xyz_dims[1]] = pred_grab_net_agg_wrist_xyz[1]
                """
                               
                # refine predictions
                for i in range(out_length):
                    
                    # left hand
                    if out_handled_obj_idxs[0,i] != -1:
                        pred_refined_xyz[inp_length+i,l_arm_mocap_idxs] = pred_grab_net_agg_xyz[0,i,l_arm_mocap_idxs] 
                    # right hand
                    if out_handled_obj_idxs[1,i] != -1:
                        pred_refined_xyz[inp_length+i,r_arm_mocap_idxs] = pred_grab_net_agg_xyz[1,i,r_arm_mocap_idxs]
                
            # # # # # # # # # #
            # unprocess pose  #      
            # # # # # # # # # #
            
            true_xyz = unprocess_pose(true_xyz, table_pos, xyz_scale)   
            pred_xyz = unprocess_pose(pred_xyz, table_pos, xyz_scale)
            if has_pred_grab_net_agg_xyz == 1:
                pred_refined_xyz = unprocess_pose(pred_refined_xyz, table_pos, xyz_scale)
                                            
            # # # #
            #     #
            # kin #
            #     #
            # # # #
            
            true_kin = data["kin"][:,:44]
            true_kin = unprocess_pose(true_kin, None, kin_scale)
            pred_kin = None
                                       
            # # # # # # # # # # # #
            #                     #
            # process for fingers #
            #                     #
            # # # # # # # # # # # #
            
            keys = {}
            # inp keys
            keys["true_inp_finger"] = "true_free_net_inp_finger"
            keys["pred_inp_finger"] = "pred_free_net_inp_finger"
            # out keys
            keys["true_out_finger"] = "true_free_net_out_finger"
            keys["pred_out_finger"] = "pred_free_net_out_finger"
                
            # free_net data
            true_finger = np.concatenate([data[keys["true_inp_finger"]],data[keys["true_out_finger"]]],axis=0)  # [inp+out length, 2, 19]
            pred_finger = np.concatenate([data[keys["pred_inp_finger"]],data[keys["pred_out_finger"]]],axis=0)  # [inp+out length, 2, 19]
            # need to replace non-masked markers from the input sequence with the ground truth !
            # - replace all except right arm
            if p == 1:
                pred_finger[:inp_length,0] = np.copy(true_finger[:inp_length,0])
            if p == 0:
                pred_finger[:inp_length,1] = np.copy(true_finger[:inp_length,1])
            true_lfinger, true_rfinger = true_finger[:,0], true_finger[:,1]
            pred_lfinger, pred_rfinger = pred_finger[:,0], pred_finger[:,1]
            pred_refined_finger = np.copy(pred_finger)                                                          # [inp+out length, 2, 19]
            
            # # # # # # # # # # # # # # # #
            # update with refined inputs  #
            # # # # # # # # # # # # # # # # 
                            
            has_pred_grab_net_inp_finger = any(["pred_grab_net_inp_finger" in k for k in data.keys()])
            if has_pred_grab_net_inp_finger == 1:
            
                inp_length = data["inp_length"]                                 # 10
                inp_finger_mask_idx = data["inp_finger_mask_idxs"]              # 0 or 1
                true_inp_handled_obj_idxs = data["inp_handled_obj_idxs"]        # [out_length]
                #pred_inp_handled_obj_idxs = data["pred_inp_handled_obj_idxs"]   # [out_length]
                pred_grab_net_inp_finger = data["pred_grab_net_inp_finger"]     # [inp_length, 19]
                                
                # refined data
                for i in range(inp_length):
                
                    # either hands
                    if true_inp_handled_obj_idxs[i] != -1:
                        pred_refined_finger[i,inp_finger_mask_idx] = pred_grab_net_inp_finger[i]
                            
            # # # # # # # # # # # # # # # #
            # update with refined outputs #
            # # # # # # # # # # # # # # # # 
            
            has_pred_grab_net_agg_finger = any(["pred_grab_net_agg_finger" in k for k in data.keys()])
            if has_pred_grab_net_agg_finger == 1:
                
                inp_length = data["inp_length"]
                out_length = data["out_length"]
                out_handled_obj_idxs = data["out_handled_obj_idxs"]         # [2, out_length]
                pred_grab_net_agg_finger = data["pred_grab_net_agg_finger"] # [2, out_length, 19]
                                                
                # refined predictions
                for i in range(out_length):
                    
                    # left hand
                    if out_handled_obj_idxs[0,i] != -1:
                        pred_refined_finger[inp_length+i,0] = pred_grab_net_agg_finger[0,i]    
                    # right hand
                    if out_handled_obj_idxs[1,i] != -1:
                        pred_refined_finger[inp_length+i,1] = pred_grab_net_agg_finger[1,i]
                
            pred_refined_lfinger, pred_refined_rfinger = pred_refined_finger[:,0], pred_refined_finger[:,1]
                      
            # # # # # # # # # #
            #                 #
            # build xml file  #
            #                 #
            # # # # # # # # # #
                        
            m_encoding = 'UTF-8'
            
            # # # # # # # # # # # #
            # initialize root MMM #
            # # # # # # # # # # # #
            
            root = ET.Element("MMM")
            root.set("version", "2.0")
            root.set("name", sequence)
                                    
            # # # # # # # # # # # # # # # #
            # true and pred object nodes  #
            # # # # # # # # # # # # # # # #

            # create true object node
            create_object_node(root, true_obj_names,        obj_mocap_names,            true_obj_paths,
                                     obj_xyz=true_obj_xyz,  obj_pos=true_obj_pos, obj_rot=true_obj_rot, 
                                     frames=frames)
            
            # create pred object node
            create_object_node(root, pred_obj_names,        obj_mocap_names,            pred_obj_paths,
                                     obj_xyz=pred_obj_xyz,  obj_pos=None, obj_rot=None, 
                                     frames=frames)

            # create pred object node
            create_object_node(root, masked_obj_names,            obj_mocap_names,      masked_obj_paths,
                                     obj_xyz=inp_masked_obj_xyz,  obj_pos=None, obj_rot=None, 
                                     frames=frames[:inp_length])
            
            # # # # # # # # # # # # # # #
            # true and pred human nodes #
            # # # # # # # # # # # # # # #
            
            # create true human node
            create_human_node(root, "true_"+subject_id, "true_"+subject_id, subject_height, subject_mass, subject_hand_length,
                                    pos_data=[pos, rot], 
                                    kin_data=[kin_names, true_kin],
                                    xyz_data=[xyz_names, true_xyz], 
                                    lfinger_data=[lfinger_names, true_lfinger], 
                                    rfinger_data=[rfinger_names, true_rfinger], 
                                    frames=frames)
             
            # create pred human node
            create_human_node(root, "pred_"+subject_id, "pred_"+subject_id, subject_height, subject_mass, subject_hand_length, 
                                    pos_data=[pos, rot], 
                                    kin_data=[kin_names, None],
                                    xyz_data=[xyz_names, pred_xyz], 
                                    lfinger_data=[lfinger_names, pred_lfinger], 
                                    rfinger_data=[rfinger_names, pred_rfinger], 
                                    frames=frames)
            
            if has_pred_grab_net_agg_xyz == 1:
                pred_lfinger = pred_refined_lfinger if has_pred_grab_net_agg_finger else pred_lfinger
                pred_rfinger = pred_refined_rfinger if has_pred_grab_net_agg_finger else pred_rfinger
                create_human_node(root, "pred_grab_net_"+subject_id, "pred_"+subject_id, subject_height, subject_mass, subject_hand_length, 
                                        pos_data=[pos, rot], 
                                        kin_data=[kin_names, None],
                                        xyz_data=[xyz_names, pred_refined_xyz], 
                                        lfinger_data=[lfinger_names, pred_lfinger], 
                                        rfinger_data=[rfinger_names, pred_rfinger], 
                                        frames=frames)
            
            # convert to string
            dom = xml.dom.minidom.parseString(ET.tostring(root))
            xml_string = dom.toprettyxml()
            part1, part2 = xml_string.split('?>')

            # create folder
            path = Path(os.path.dirname(file.replace(args.result_root,"./results/"))) # results\action2pose\agraph\oh\Cut_files_motions_3021_Cut1_c_0_05cm_01
            path.mkdir(parents=True, exist_ok=True)

            filename = file.replace(args.result_root,"results/").replace(".json",".xml")
            with open(filename, 'w') as xfile:
                xfile.write(part1 + 'encoding=\"{}\"?>\n'.format(m_encoding) + part2)
                xfile.close()

            # # # # # # # # # # #
            # create text file  #
            # # # # # # # # # # #
            
            ubuntu_result_root = "/home/haziq/Forecasting-Bimanual-Object-Manipulation-Sequences-From-Unimanual-Observations/datasets/kit_mocap/my_scripts/result_processing/"
            
            # MMMMotionConverter
            MMMMotionConverter = "/home/haziq/MMMTools/build/bin/MMMMotionConverter"   
            
            # inputMotion and outputMotion
            inputMotion = outputMotion = os.path.join(ubuntu_result_root,filename)       
                        
            print(filename.replace("xml","sh"))
            with open(filename.replace("xml","sh"),"wb") as f:
                
                # sanity check for the true pose, to make sure that the noise has been added
                if 0:
                
                    # motionName
                    motionName = "true_"+subject_id
                    
                    # converterConfigFile
                    converterConfigFile = "/home/haziq/MMMTools/data/Model/Winter/NloptConverterVicon2MMM_WinterConfig.xml"
                    
                    # outputModelFile
                    outputModelFile = "/home/haziq/MMMTools/data/Model/Winter/mmm.xml"
                    
                    # outputModelProcessorConfigFile
                    outputModelProcessorConfigFile = os.path.join("/home/haziq/MMMTools/data/Model/Winter/config/",subject_id+".xml")
                    
                    # write
                    write = "{} --inputMotion \"{}\" --motionName {} --converterConfigFile \"{}\" --outputModelFile \"{}\" --outputModelProcessorConfigFile \"{}\" --outputMotion \"{}\" &&\\\nsleep 5 &&\\\n".format(MMMMotionConverter, inputMotion, motionName, converterConfigFile, outputModelFile, outputModelProcessorConfigFile, outputMotion)
                    #write = write.replace("\\","/")
                    f.write(write.encode())
                
                """
                xyz
                """
                
                # motionName
                motionName = "pred_"+subject_id
                
                # converterConfigFile
                converterConfigFile = "/home/haziq/MMMTools/data/Model/Winter/NloptConverterVicon2MMM_WinterConfig.xml"
                
                # outputModelFile
                outputModelFile = "/home/haziq/MMMTools/data/Model/Winter/mmm.xml"
                
                # outputModelProcessorConfigFile
                outputModelProcessorConfigFile = os.path.join("/home/haziq/MMMTools/data/Model/Winter/config/",subject_id+".xml")
                
                # write
                write = "{} --inputMotion \"{}\" --motionName \"{}\" --converterConfigFile \"{}\" --outputModelFile \"{}\" --outputModelProcessorConfigFile \"{}\" --outputMotion \"{}\" &&\\\nsleep 5 &&\\\n".format(MMMMotionConverter, inputMotion, motionName, converterConfigFile, outputModelFile, outputModelProcessorConfigFile, outputMotion)
                #write = write.replace("\\","/")
                f.write(write.encode())
            
                """
                agg
                """
            
                if (has_pred_grab_net_inp_xyz == 1 or has_pred_grab_net_agg_xyz == 1):
                                
                    # motionName
                    motionName = "pred_grab_net_"+subject_id
                    
                    # converterConfigFile
                    converterConfigFile = "/home/haziq/MMMTools/data/Model/Winter/NloptConverterVicon2MMM_WinterConfig.xml"
                    
                    # outputModelFile
                    outputModelFile = "/home/haziq/MMMTools/data/Model/Winter/mmm.xml"
                    
                    # outputModelProcessorConfigFile
                    outputModelProcessorConfigFile = os.path.join("/home/haziq/MMMTools/data/Model/Winter/config/",subject_id+".xml")
                    
                    # write
                    write = "{} --inputMotion \"{}\" --motionName \"{}\" --converterConfigFile \"{}\" --outputModelFile \"{}\" --outputModelProcessorConfigFile \"{}\" --outputMotion \"{}\" &&\\\nsleep 5 &&\\\n".format(MMMMotionConverter, inputMotion, motionName, converterConfigFile, outputModelFile, outputModelProcessorConfigFile, outputMotion)
                    #write = write.replace("\\","/")
                    f.write(write.encode())
            
                """
                obj
                """
            
                for obj_path,pred_obj_name in zip(obj_paths,pred_obj_names):
                    
                    #print("pred_obj_name = {}".format(pred_obj_name))
                    #print("obj_path = {}".format(obj_path))
                    #input()
                    
                    # motionName
                    motionName = pred_obj_name
                    
                    # converterConfigFile
                    converterConfigFile = converterConfigFile_dict[pred_obj_name]
                    
                    # outputModelFile
                    outputModelFile = obj_path
                    
                    # outputModelProcessorConfigFile
                    #outputModelProcessorConfigFile = outputModelProcessorConfigFile_dict[pred_obj_name]
                    
                    # write
                    write = "{} --inputMotion \"{}\" --motionName \"{}\" --converterConfigFile \"{}\" --outputModelFile \"{}\" --outputMotion \"{}\" &&\\\nsleep 5 &&\\\n".format(MMMMotionConverter, inputMotion, motionName, converterConfigFile, outputModelFile, outputMotion)
                    #write = write.replace("\\","/")
                    f.write(write.encode())
                                    
                for obj_path,pred_obj_name,masked_obj_name in zip(obj_paths,pred_obj_names,masked_obj_names):
                    
                    #print("pred_obj_name = {}".format(pred_obj_name))
                    #print("obj_path = {}".format(obj_path))
                    #input()
                    
                    # motionName
                    motionName = masked_obj_name
                    
                    # converterConfigFile
                    converterConfigFile = converterConfigFile_dict[pred_obj_name]
                    
                    # outputModelFile
                    outputModelFile = obj_path
                    
                    # outputModelProcessorConfigFile
                    #outputModelProcessorConfigFile = outputModelProcessorConfigFile_dict[pred_obj_name]
                    
                    # write
                    write = "{} --inputMotion \"{}\" --motionName \"{}\" --converterConfigFile \"{}\" --outputModelFile \"{}\" --outputMotion \"{}\" &&\\\nsleep 5 &&\\\n".format(MMMMotionConverter, inputMotion, motionName, converterConfigFile, outputModelFile, outputMotion)
                    #write = write.replace("\\","/")
                    f.write(write.encode())   
                
                write = "echo \"{} DONE\"".format(filename)
                write = write.replace("xml","sh")#.replace("\\","/")
                f.write(write.encode())
            os.chmod(filename.replace("xml","sh"),0o755)
    
    Path(os.path.join("results",args.result_name)).mkdir(parents=True,exist_ok=True)  
    print()
    print("Creating convert.sh",os.path.join("results",args.result_name,"convert.sh"))
    print()
    # main script that runs all the conversion scripts
    with open(os.path.join("results",args.result_name,"convert.sh"),"wb") as f: # results\action2pose\agraph\oh
        
        files = glob(os.path.join("results",args.result_name,"**","*.sh"), recursive=True)
        files = [os.path.abspath(file) for file in files if "convert" not in file]
        #print(files[-1]) # results/action2pose/kit_mocap/graph_v1 2022_06_23/noise_scale=0_noise_add_type=each_object_vel=1_kl=1e-3/Cut_files_motions_3023_Cut3_c_30_05cm_03/0000000000.sh
        #print(os.path.abspath(files[-1]))
        #files = ["./"+os.path.basename(args.result_name)+file.replace(os.path.join("results",args.result_name),"") for file in files if "convert" not in file]
        #print(files[-1])
        #sys.exit()
        for file in files:
            print(file)
            write = file+" &\\\n"
            #write = write.replace(" ","\ ")
            #write = file.replace("\\","/")+"\n"
            f.write(write.encode())
        write = ("\\\n")
        f.write(write.encode())
        write = ("wait")
        f.write(write.encode())
    os.chmod(os.path.join("results",args.result_name,"convert.sh"),0o755)
    
def create_human_node(root, subject_name, subject_id, subject_height, subject_mass, subject_hand_length, pos_data, kin_data, xyz_data, lfinger_data, rfinger_data, frames):

    # model pose
    pos = pos_data[0]
    rot = pos_data[1]
    
    # kin data
    kin_names = kin_data[0]
    kin = kin_data[1]
    
    # xyz data
    xyz_names = xyz_data[0]
    xyz = xyz_data[1]

    # # # # # # # # # # # # # # # # # # #
    # MMM                               #
    # - Motion name="1480"              # <-----
    # # # # # # # # # # # # # # # # # # #           
    
    motion = ET.SubElement(root, "Motion")
    motion.set("name",subject_name)
    motion.set("type","object")
    
    # # # # # # # # # # # # # # # # # # # # # #
    # MMM                                     #
    # - Motion name="1480"                    # 
    #   - Model path="mmm.xml"                # <-----
    #     - ModelProcessorConfig type="Winter # <-----
    #       - Height                          # <-----
    #       - Mass                            # <-----
    #       - HandLength                      # <-----
    # # # # # # # # # # # # # # # # # # # # # #   
        
    model = ET.SubElement(motion, "Model")
    model.set("path",outputModelFile_dict[subject_id])
    
    model_processor_config = ET.SubElement(model, "ModelProcessorConfig")
    model_processor_config.set("type","Winter")
    
    height = ET.SubElement(model_processor_config, "Height")
    height.text = str(subject_height)
    mass = ET.SubElement(model_processor_config, "Mass")
    mass.text = str(subject_mass)
    hand_length = ET.SubElement(model_processor_config, "HandLength")
    hand_length.text = str(subject_hand_length)
    
    # # # # # # # # # # # # # # # # # # # # # #
    # MMM                                     #
    # - Motion name="1480"                    #
    #   - Model path="mmm.xml"                #
    #   - Sensors                             # <-----
    # # # # # # # # # # # # # # # # # # # # # #                
    
    sensors = ET.SubElement(motion, "Sensors")
    
    # # # # # # # # # # # # # # # # # # # # # #
    # MMM                                     #
    # - Motion name="1480"                    #
    #   - Model path="mmm.xml"                #
    #   - Sensors                             #
    #     - Sensor type="ModelPose"           # <-----
    # # # # # # # # # # # # # # # # # # # # # # 
    
    sensor1 = ET.SubElement(sensors, "Sensor")
    sensor1.set("type","ModelPose")
    sensor1.set("version","1.0")           

    # # # # # # # # # # # # # # # # # # # # # #
    # MMM                                     #
    # - Motion name="1480"                    #
    #   - Model path="mmm.xml"                #
    #   - Sensors                             #
    #     - Sensor type="ModelPose"           #
    #       - Configuration                   # <-----
    #       - Data                            # <-----
    # # # # # # # # # # # # # # # # # # # # # # 
    
    configuration = ET.SubElement(sensor1, "Configuration")
    data = ET.SubElement(sensor1, "Data")
    
    # # # # # # # # # # # # # # # # # # # # # #
    # MMM                                     #
    # - Motion name="1480"                    #
    #   - Model path="mmm.xml"                #
    #   - Sensors                             #
    #     - Sensor type="ModelPose"           #
    #       - Configuration                   #
    #       - Data                            #
    #         - Measurement                   # <-----
    #           - RootPosition                # <-----
    #           - RootRotation                # <-----
    # # # # # # # # # # # # # # # # # # # # # # 
            
    # measurements
    for pos_t, rot_t, t in zip(pos, rot, frames):
        measurement = ET.SubElement(data, "Measurement")
        measurement.set("timestep",str(t))
        
        root_position = ET.SubElement(measurement, "RootPosition").text = " ".join([str(x) for x in pos_t])
        root_rotation = ET.SubElement(measurement, "RootRotation").text = " ".join([str(x) for x in rot_t])
    
    if kin is not None:
    
        # # # # # # # # # # # # # # # # # # # # # #
        # MMM                                     #
        # - Motion name="1480"                    #
        #   - Model path="mmm.xml"                #
        #   - Sensors                             #
        #     - Sensor type="Kinematic"           # <-----
        # # # # # # # # # # # # # # # # # # # # # # 
        
        sensor1 = ET.SubElement(sensors, "Sensor")
        sensor1.set("type","Kinematic")
        sensor1.set("version","1.0")           

        # # # # # # # # # # # # # # # # # # # # # #
        # MMM                                     #
        # - Motion name="1480"                    #
        #   - Model path="mmm.xml"                #
        #   - Sensors                             #
        #     - Sensor type="Kinematic"           #
        #       - Configuration                   # <-----
        #       - Data                            # <-----
        # # # # # # # # # # # # # # # # # # # # # # 
        
        configuration = ET.SubElement(sensor1, "Configuration")
        data = ET.SubElement(sensor1, "Data")
        
        # # # # # # # # # # # # # # # # # # # # # #
        # MMM                                     #
        # - Motion name="1480"                    #
        #   - Model path="mmm.xml"                #
        #   - Sensors                             #
        #     - Sensor type="Kinematic"           #
        #       - Configuration                   #
        #         - Joint name = "BLNx_joint"     # <-----
        #       - Data                            #
        # # # # # # # # # # # # # # # # # # # # # # 
        
        for kin_name in kin_names:
            joint = ET.SubElement(configuration, "Joint")
            joint.set("name",kin_name)
        
        # # # # # # # # # # # # # # # # # # # # # #
        # MMM                                     #
        # - Motion name="kitchen_sideboard"       #
        #   - Model path="kitchen_sideboard.xml"  #
        #   - Sensors                             #
        #     - Sensor type="ModelPose"           #
        #       - Configuration                   #
        #         - Joint name = "BLNx_joint"     #
        #       - Data                            #
        #         - Measurement                   # <-----
        #           - JointPosition               # <-----
        # # # # # # # # # # # # # # # # # # # # # # 
        
        # measurements
        for kin_t,t in zip(kin,frames):
            measurement = ET.SubElement(data, "Measurement")
            measurement.set("timestep",str(t))
            
            joint_position = ET.SubElement(measurement, "JointPosition").text = " ".join([str(x) for x in kin_t])
    
    if lfinger_data[1] is not None and rfinger_data[1] is not None:        
        for finger_data in [lfinger_data, rfinger_data]:
            
            finger_names = finger_data[0]
            finger = finger_data[1]
        
            # # # # # # # # # # # # # # # # # # # # # #
            # MMM                                     #
            # - Motion name="1480"                    #
            #   - Model path="mmm.xml"                #
            #   - Sensors                             #
            #     - Sensor type="Kinematic"           # <-----
            # # # # # # # # # # # # # # # # # # # # # # 
            
            sensor1 = ET.SubElement(sensors, "Sensor")
            sensor1.set("type","Kinematic")
            sensor1.set("version","1.0")    

            # # # # # # # # # # # # # # # # # # # # # #
            # MMM                                     #
            # - Motion name="1480"                    #
            #   - Model path="mmm.xml"                #
            #   - Sensors                             #
            #     - Sensor type="Kinematic"           #
            #       - Configuration                   # <-----
            #       - Data                            # <-----
            # # # # # # # # # # # # # # # # # # # # # #         
        
            configuration = ET.SubElement(sensor1, "Configuration")
            data = ET.SubElement(sensor1, "Data")
            
            # # # # # # # # # # # # # # # # # # # # # #
            # MMM                                     #
            # - Motion name="1480"                    #
            #   - Model path="mmm.xml"                #
            #   - Sensors                             #
            #     - Sensor type="Kinematic"           #
            #       - Configuration                   #
            #         - Joint name = "BLNx_joint"     # <-----
            #       - Data                            #
            # # # # # # # # # # # # # # # # # # # # # # 
            
            for finger_name in finger_names :
                joint = ET.SubElement(configuration, "Joint")
                joint.set("name",finger_name)

            # # # # # # # # # # # # # # # # # # # # # #
            # MMM                                     #
            # - Motion name="kitchen_sideboard"       #
            #   - Model path="kitchen_sideboard.xml"  #
            #   - Sensors                             #
            #     - Sensor type="ModelPose"           #
            #       - Configuration                   #
            #         - Joint name = "BLNx_joint"     #
            #       - Data                            #
            #         - Measurement                   # <-----
            #           - JointPosition               # <-----
            # # # # # # # # # # # # # # # # # # # # # # 
            
            # measurements
            for finger_t,t in zip(finger,frames):
                measurement = ET.SubElement(data, "Measurement")
                measurement.set("timestep",str(t))
                
                joint_position = ET.SubElement(measurement, "JointPosition").text = " ".join([str(x) for x in finger_t])
    
    if xyz is not None:
    
        # # # # # # # # # # # # # # # # # # # # # #
        # MMM                                     #
        # - Motion name="1480"                    #
        #   - Model path="mmm.xml"                #
        #   - Sensors                             #
        #     - Sensor type="MocapMarker"         # <-----
        # # # # # # # # # # # # # # # # # # # # # # 
        
        sensor1 = ET.SubElement(sensors, "Sensor")
        sensor1.set("type","MoCapMarker")
        sensor1.set("version","1.0")           

        # # # # # # # # # # # # # # # # # # # # # #
        # MMM                                     #
        # - Motion name="1480"                    #
        #   - Model path="mmm.xml"                #
        #   - Sensors                             #
        #     - Sensor type="MocapMarker"         #
        #       - Configuration                   # <-----
        #       - Data                            # <-----
        # # # # # # # # # # # # # # # # # # # # # # 
        
        configuration = ET.SubElement(sensor1, "Configuration")
        data = ET.SubElement(sensor1, "Data")
        
        # # # # # # # # # # # # # # # # # # # # # #
        # MMM                                     #
        # - Motion name="kitchen_sideboard"       #
        #   - Model path="kitchen_sideboard.xml"  #
        #   - Sensors                             #
        #     - Sensor type="ModelPose"           #
        #       - Configuration                   #
        #       - Data                            #
        #         - Measurement                   # <-----
        #           - MarkerPosition              # <-----
        # # # # # # # # # # # # # # # # # # # # # # 
                
        # measurements
        for xyz_t,t in zip(xyz,frames):
            measurement = ET.SubElement(data, "Measurement")
            measurement.set("timestep",str(t))
            
            for name,value_t in zip(xyz_names,xyz_t):
                marker_position = ET.SubElement(measurement, "MarkerPosition")
                marker_position.set("name",name)
                marker_position.text = " ".join([str(x) for x in value_t])
            
def create_object_node(root, obj_names, obj_mocap_names, obj_paths, obj_xyz, obj_pos, obj_rot, frames):

    #if obj_names is None or obj_pos is None or obj_rot is None:
    #    return

    if obj_xyz is None and obj_pos is None and obj_rot is None:
        return

    # = = = = = = = = = = = = = = = = = #
    # object motion                     #
    # = = = = = = = = = = = = = = = = = #
    
    # # # # # # # # # # # # # # # # # # #
    # MMM                               #
    # - Motion name="kitchen_sideboard" # <-----
    # - Motion name="salad_fork"        # <-----
    # - Motion ...                      # <-----
    # # # # # # # # # # # # # # # # # # #
        
    for i,(obj_name,obj_path) in enumerate(zip(obj_names,obj_paths)):
        motion = ET.SubElement(root, "Motion")
        motion.set("name",obj_name)
        motion.set("type","object")
        motion.set("synchronized","true")
    
        # # # # # # # # # # # # # # # # # # # # # #
        # MMM                                     #
        # - Motion name="kitchen_sideboard"       #
        #   - Model path="kitchen_sideboard.xml"  # <-----
        # # # # # # # # # # # # # # # # # # # # # #
                
        model = ET.SubElement(motion, "Model")
        model.set("path",obj_path)
        
        # # # # # # # # # # # # # # # # # # # # # #
        # MMM                                     #
        # - Motion name="kitchen_sideboard"       #
        #   - Model path="kitchen_sideboard.xml"  #
        #   - Sensors                             # <-----
        # # # # # # # # # # # # # # # # # # # # # #                
        
        sensors = ET.SubElement(motion, "Sensors")
        
        # # # # # # # # # # # # # # # # # # # # # #
        # MMM                                     #
        # - Motion name="kitchen_sideboard"       #
        #   - Model path="kitchen_sideboard.xml"  #
        #   - Sensors                             #
        #     - Sensor type="ModelPose"           # <-----
        # # # # # # # # # # # # # # # # # # # # # # 
        
        sensor1 = ET.SubElement(sensors, "Sensor")
        sensor1.set("type","ModelPose")
        sensor1.set("version","1.0")        
        sensor2 = ET.SubElement(sensors, "Sensor")
        sensor2.set("type","MoCapMarker")
        sensor2.set("version","1.0")
        
        # # # # # # # # # # # # # # # # # # # # # #
        # MMM                                     #
        # - Motion name="kitchen_sideboard"       #
        #   - Model path="kitchen_sideboard.xml"  #
        #   - Sensors                             #
        #     - Sensor type="ModelPose"           #
        #       - Configuration                   # <-----
        #       - Data                            # <-----
        # # # # # # # # # # # # # # # # # # # # # # 
        
        configuration1 = ET.SubElement(sensor1, "Configuration")
        data1 = ET.SubElement(sensor1, "Data")
        configuration2 = ET.SubElement(sensor2, "Configuration")
        data2 = ET.SubElement(sensor2, "Data")
        
        # # # # # # # # # # # # # # # # # # # # # #
        # MMM                                     #
        # - Motion name="kitchen_sideboard"       #
        #   - Model path="kitchen_sideboard.xml"  #
        #   - Sensors                             #
        #     - Sensor type="ModelPose"           #
        #       - Configuration                   #
        #       - Data                            #
        #         - Measurement                   # <-----
        #           - RootPosition                # <-----
        #           - RootRotation                # <-----
        # # # # # # # # # # # # # # # # # # # # # # 
        
        #debug=3
        #print(object_names[debug])
        #for object_pos_t, object_rot_t, t in zip(object_pos[debug], object_rot[debug], frames):
        #    print(object_pos_t, object_rot_t, t)
        #    print(" ".join([str(x) for x in object_pos_t]))
        #    input()
        
        if obj_pos is not None and obj_rot is not None:
            # measurements
            for obj_pos_t, obj_rot_t, t in zip(obj_pos[i], obj_rot[i], frames):
                measurement1 = ET.SubElement(data1, "Measurement")
                measurement1.set("timestep",str(t))
                
                root_position = ET.SubElement(measurement1, "RootPosition").text = " ".join([str(x) for x in obj_pos_t])
                root_rotation = ET.SubElement(measurement1, "RootRotation").text = " ".join([str(x) for x in obj_rot_t])
                
        if obj_xyz is not None and i < obj_xyz.shape[0]:
            # measurements
            for obj_xyz_t,t in zip(obj_xyz[i],frames):
                measurement2 = ET.SubElement(data2, "Measurement")
                measurement2.set("timestep",str(t))
                                
                for j,name in enumerate(obj_mocap_names[i]):
                    
                    mocap_value = ET.SubElement(measurement2, "MarkerPosition")
                    mocap_value.text = " ".join([str(x) for x in obj_xyz_t[j]])
                    mocap_value.set("name",name)
                    
if __name__ == "__main__":

    main()