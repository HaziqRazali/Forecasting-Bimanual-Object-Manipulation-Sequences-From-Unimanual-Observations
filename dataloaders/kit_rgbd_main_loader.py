import os
import re
import ast
import sys
import json
import copy
import torch
import logging
import inspect
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from bps import bps
from glob import glob
from natsort import natsorted
from collections import Counter

import kit_rgbd_variables as var
from utils import *

def parse_action_data(data):

    # even [0,2,4,...] key frames
    # odd  [1,3,5,...] action
    """
    {
      "right_hand": [0, 0, 1015],
      "left_hand": [0, 0, 1015]
    }
    """
    
    parsed_action_data = {}
    
    # parse
    for hand in ["left_hand","right_hand"]:
        parsed_action_data[hand] = []
        #print(hand)
        #print(data[hand])
        for i in range(0,len(data[hand])-2,2):
            start_frame = data[hand][i+0]
            action      = data[hand][i+1]
            end_frame   = data[hand][i+2]
            parsed_action_data[hand].extend([action for _ in range(start_frame,end_frame)])
        parsed_action_data[hand].append(parsed_action_data[hand][-1]) # this is because I ignored the final end_frame
    
    # convert to array
    for k,v in parsed_action_data.items():
        parsed_action_data[k] = np.array(parsed_action_data[k])
    return parsed_action_data

def most_confident_pose_idx(poses):

    if len(poses) == 1:
        return 0

    pose_confidences = []
    for pose in poses:
        joints = [x for x in list(pose.keys()) if x in ['LEar', 'LElbow', 'LEye', 'LHip', 'LShoulder', 'LWrist', 'MidHip', 'Neck', 'Nose', 'REar', 'RElbow', 'REye', 'RHip', 'RShoulder', 'RWrist']]
        total_confidence = np.sum([pose[joint]["confidence"] for joint in joints])
        pose_confidences.append(total_confidence)
    idx = np.argmax(pose_confidences)    
    return idx

def most_confident_hand_idx(hands):
    scores = [0]*len(hands)
    for i,hand in enumerate(hands):
    
        # make sure it detects both the left and right hands
        lhand = np.array([v["confidence"] for k,v in hand.items() if "L" in k])
        rhand = np.array([v["confidence"] for k,v in hand.items() if "R" in k])
        assert len(lhand) == 21
        assert len(rhand) == 21
        scores[i] = np.sum(lhand) + np.sum(rhand)
    return np.argmax(scores)

def remove_duplicates(sequence_data, key, main_3d_objects):

    """
    sequence_data["main_3d_objects"][instance_name] = {}
    sequence_data["main_3d_objects"][instance_name]["time"] = []
    sequence_data["main_3d_objects"][instance_name]["bbox"] = []   
    sequence_data["main_3d_objects"][instance_name]["colour"] = x["colour"] 
    """
    
    # for each object x in the ground truth
    for main_3d_object,count in main_3d_objects.items():
        
        # get object x in the sequence_data
        # duplicates for the current main_3d_object e.g.
        # - main_3d_object = Cup
        # - duplicates["main_3d_objects"].keys() = Cup_1, Cup_2
        duplicates = {}
        duplicates[key] = {}
        duplicates[key] = {k:sequence_data[key][k] for k in sequence_data[key].keys() if main_3d_object in k}
                
        # no duplicates, meaning I already have the exact amount as the ground truth
        if len(duplicates[key].keys()) == count:
            continue
            
        # contain duplicates, so I need to remove the excess detections
        elif len(duplicates[key].keys()) > count:
            #print("contain duplicates:")
            #print(key, duplicates[key].keys(), count)
            #print(sequence_data["main_3d_objects"].keys())
            #sys.exit()
            
            # find the top <count> keys with the longest lengths
            # - instead, sum confidences
                        
            by_num_detections = sorted([(len(v["bbox"]), k) for k, v in duplicates[key].items()], reverse=True)
            to_keep = by_num_detections[:count]
            to_delete = by_num_detections[count:]
            
            # update the sequence_data
            for length,k in to_delete:
                del sequence_data[key][k] 
        
        # contain fewer than the needed count
        # except for woodenwedge
        elif len(duplicates[key].keys()) < count and len(duplicates[key].keys()) > 0 and "woodenwedge" in main_3d_object:
            continue
            
        # contain fewer than the needed count
        else:
            logger.info("Error!")
            logger.info("{}".format(sequence_data["metadata"]["filename"]))
            logger.info("{}".format(key))
            logger.info("{} {}".format(key, main_3d_objects))
            logger.info("current key {}".format(main_3d_object))
            logger.info("sequence_data[key] {}".format(sequence_data[key].keys()))            
            sys.exit()

    #sequence_data["metadata"]["object_names"] = [x for x in sequence_data[key].keys()]
    return sequence_data

def get_tl_br(data):
    tlx, tly = np.min(data[:,0]), np.min(data[:,1])
    brx, bry = np.max(data[:,0]), np.max(data[:,1])
    return np.array([tlx,tly]), np.array([brx,bry])

def get_actions_with_no_similar_objects(objects, action_to_objects):
    
    objects = [x for x in objects if "Hand" not in x]
    return_actions = []
    for action,other_objects in action_to_objects.items():
        #print(objects, other_objects, set(objects) & set(other_objects))
        if len(list(set(objects) & set(other_objects))) == 0:
            return_actions.append(action)
    return return_actions

class main_loader(torch.utils.data.Dataset):
    def __init__(self, args, dtype):
    
        assert dtype == "train" or dtype == "val"    
        for key, value in args.__dict__.items():
            setattr(self, key, value)
        self.dtype = dtype
        
        # caller
        self.caller = sys.argv[0]
        
        # logger
        logger = logging.getLogger(os.path.join(args.log_root,args.log_name))
        
        # sequences to skip
        self.skip = var.skip
        
        # # # # # # # #
        #             #
        # human data  #
        #             #
        # # # # # # # # 
        
        if not hasattr(args, "hand_xyz_dims"):
            self.hand_xyz_dims = var.hand_xyz_dims
            setattr(args, "hand_xyz_dims", var.hand_xyz_dims)
        
        
        setattr(args, "num_body_joints", var.num_body_joints)
        setattr(args, "body_dim", var.body_dim)
        setattr(args, "num_obj_markers", var.num_obj_markers)
        setattr(args, "obj_dim", var.obj_dim)
        setattr(args, "num_hands", var.num_hands)
        setattr(args, "finger_dim", var.hand_dim)
        
        # # # # # # # #
        #             #
        # action data #
        #             #
        # # # # # # # # 
        
        # action        
        self.clf_actions = self.fine_actions
        setattr(args, "clf_actions", self.clf_actions)
        self.main_action_to_id = {a:i for i,a in enumerate(self.main_actions)}
        self.fine_action_to_id = {a:i for i,a in enumerate(self.fine_actions)}   
        self.clf_action_to_id = {i:a for i,a in enumerate(self.clf_actions)}    
        for k,v in self.clf_action_to_id.items():
            logger.info("{} {}".format(k,v))
            
        # list of all objects
        self.all_objects = var.all_objects
        
        # action -> objects
        self.action_to_objects = var.action_to_objects
        
        ## For each action, get all other actions with no similar objects
        #self.actions_with_no_similar_objects = {}
        #for action,objects in self.action_to_objects.items():
        #    action = re.sub(r'task_.*_[kw]_', '', action)
        #    self.actions_with_no_similar_objects[action] = get_actions_with_no_similar_objects(objects,self.action_to_objects)
        #    self.actions_with_no_similar_objects[action] = [x for x in self.actions_with_no_similar_objects[action] if "task_5_k_cereals" not in x]
        #    self.actions_with_no_similar_objects[action] = [re.sub(r'task_.*_[kw]_', '', x) for x in self.actions_with_no_similar_objects[action]]
        #for k,v in self.actions_with_no_similar_objects.items():
        #    print(k, v)
        
        # # # # # # # # #
        #               #
        # object labels #
        #               #
        # # # # # # # # # 
        
        # get the list of present objects given the main actions
        action_to_objects = []
        for main_action in self.main_actions:
            action_to_objects.extend(self.action_to_objects[main_action])
        # remove duplicates
        action_to_objects = list(set(action_to_objects))
        action_to_objects = sorted(action_to_objects)   
        # get ID
        self.object_name_to_id = {k:i+1 for i,k in enumerate(action_to_objects)}
        self.object_id_to_name = {i+1:k for i,k in enumerate(action_to_objects)}
        for i,action_to_object in enumerate(action_to_objects):
            print(i+1, action_to_object)
        print()
        setattr(args, "object_id_to_name", self.object_id_to_name)
        
        self.num_obj_classes = len(action_to_objects) + 1 - 2 # plus 1 for the padding minus 2 because action_to_objects include the left and right hand, here we only want the OBJECT (excluding wrist)
        self.num_obj_wrist_classes = self.num_obj_classes + 2 # plus 2 because we now want to include the left and right hands
        setattr(args, "num_obj_classes", self.num_obj_classes)
        setattr(args, "num_obj_wrist_classes", self.num_obj_wrist_classes)
        setattr(args, "obj_padded_length",       self.object_padded_length)
        setattr(args, "obj_body_padded_length",  self.object_padded_length+1)
        setattr(args, "obj_wrist_padded_length", self.object_padded_length+2) # the object padded length i specify in the config files do not take into account the left and right hands so we +2 here
        
        # l_arm_mocap_idxs and r_arm_mocap_idxs
        self.l_arm_mocap_idxs  = var.l_arm_mocap_idxs
        self.r_arm_mocap_idxs  = var.r_arm_mocap_idxs
        setattr(args, "l_arm_mocap_idxs", var.l_arm_mocap_idxs)
        setattr(args, "r_arm_mocap_idxs", var.r_arm_mocap_idxs)
        
        """        
        note that the variable t may not be the same
        
        - [metadata]
            - [action]   = task_2_k_cooking_with_bowls
            - [subject]  = subject_5
            - [filename] = /home_nfs/haziq/datasets/KIT-Bimanual-Actions/bimacs_derived_data/subject_5/task_2_k_cooking_with_bowls/take_7
            - [main_object_names]
            - [all_object_names]
        - [table_center]
        - [time]  
        
        - [person]
            - [person][pose_2d]
                - [person][pose_2d][t]
                - [person][pose_2d][xy]
                - [person][pose_2d][confidence]
            - [person][hand_2d]
                - [person][hand_2d][t]
                - [person][hand_2d][xy]
                - [person][hand_2d][confidence]
        
        - [main_3d_objects]
            - [main_3d_objects][object1]
                - [main_3d_objects][object1][time] # [t]
                - [main_3d_objects][object1][bbox] # [t, 2, 3]
            - [main_3d_objects][lhand]
            
        - [all_objects]
            - [all_objects][object1]
                - [all_objects][object1][time] # [t]
                - [all_objects][object1][bbox] # [t, 2, 3]
            - [all_objects][lhand]
            
        - [3d_distractor]
            - [3d_distractor][object1]
                - [3d_distractor][time] # [t]
                - [3d_distractor][bbox] # [t, 2, 3]
                
        - [segmentation]
            - [lhand]
            - [rhand]
        """
        
        # load metadata file contains the main object the person interacts with for each sequence
        metadata = pd.read_csv(os.path.join(self.data_root,"metadata.csv"),converters={'main_objects': ast.literal_eval})
        all_objects_unprocessed = metadata["all_objects_unprocessed"].tolist()
        all_objects = []
        for row in all_objects_unprocessed:
            
            if pd.isnull(row):        
                all_objects.append([""])
            else:
                row = row.split(",")
                count = Counter(row)
                entry = {k:v for k,v in count.items()}
                entry = {**entry, **{"LeftHand":1, "RightHand":1}}
                all_objects.append(entry)            
        metadata.insert(metadata.shape[1], column="all_objects", value=all_objects)    
        self.metadata = metadata.copy()
        
        # # # # # # # # # #   
        #                 #
        # load from cache #
        #                 #
        # # # # # # # # # #
        
        if args.cached_data_path == None:
            main_actions_truncated = [re.sub(r'task_.*_[kw]_', '', x) for x in self.main_actions]            
            data_folder = self.data_name + "_"
            split       = dtype+"="+str(eval("self."+dtype+"_samples")) + "_"
            actions     = str(main_actions_truncated)
            
            cached_data_path = os.path.join(self.data_root, "cached-data", \
                                            data_folder + \
                                            split + \
                                            actions)
                                            
        if os.path.isfile(cached_data_path):
            
            logger.info("Loading dataset from: {}".format(cached_data_path))
            with open(cached_data_path) as f:
                sequence_data = json.load(f)
                self.sequence_data = sequence_data
                self.sequence_data = [dict_list_to_arr(x,skip=["filename","main_action"]) for x in self.sequence_data]
            return
            
        # # # # # # # # # #    
        #                 #        
        # load from files #
        #                 #
        # # # # # # # # # #
        
        self.supplementary_data = []
        self.sequence_data = []
        
        # camera normalization data used only for visualization
        normdata = json.load(open(os.path.join(self.data_root,"bimacs_rgbd_data_cam_norm.json"),"r"))
                               
        cam_norm_idx = 0
        subjects = glob(os.path.join(self.data_root,"bimacs_derived_data","*"))
        subjects = [x for x in subjects if os.path.isdir(x)]
        subjects = natsorted(subjects)
        for subject in subjects:                                        #print(subject) # /home_nfs/haziq/datasets/KIT-Bimanual-Actions/bimacs_derived_data/subject_1
                        
            tasks = glob(os.path.join(subject,"*"))               
            tasks = [x for x in tasks if os.path.isdir(x)]
            tasks = natsorted(tasks)
            #for task,sample_ratio in zip(tasks,self.sample_ratio):      #print(task)    # /home_nfs/haziq/datasets/KIT-Bimanual-Actions/bimacs_derived_data/subject_1/task_1_k_cooking
            for task in tasks:
                                                                
                if not any([os.path.basename(task) == x for x in self.main_actions]):
                    continue
                                
                # take_0, take_1, ...
                takes = glob(os.path.join(task,"*"))              
                takes = [x for x in takes if os.path.isdir(x)]
                takes = natsorted(takes)
                
                # upsample takes
                # not really needed since all the actions have the same number of samples !
                #sample_ratio1 = int(sample_ratio)
                #sample_ratio2 = sample_ratio - sample_ratio1             
                #takes_sampled1 = takes * sample_ratio1
                #takes_sampled2 = takes[:int(sample_ratio2*len(takes))]
                #takes = takes_sampled1 + takes_sampled2
                
                for take in takes:                                      #print(take) # /home_nfs/haziq/datasets/KIT-Bimanual-Actions/bimacs_derived_data/subject_1/task_1_k_cooking/take_0
                    if not any([re.sub("take_","",os.path.basename(take)) == str(x) for x in eval("self."+dtype+"_samples")]):
                        cam_norm_idx += 1
                        continue
                    logger.info("{} {}".format(dtype, take))
                    
                    if any([(x[0] == os.path.basename(subject) and x[1] == os.path.basename(task) and x[2] == os.path.basename(take)) for x in self.skip]):
                        logger.info("Skipping {}".format(take))
                        continue
                    
                    # main objects the person interacts with
                    main_objects = metadata.loc[(metadata["subject"] == os.path.basename(subject)) & (metadata["task"] == os.path.basename(task)) & (metadata["take"] == os.path.basename(take))]["main_objects"]
                    assert len(main_objects) == 1
                    main_objects = main_objects.iloc[0]
                    
                    # all objects in the scene
                    all_objects = metadata.loc[(metadata["subject"] == os.path.basename(subject)) & (metadata["task"] == os.path.basename(task)) & (metadata["take"] == os.path.basename(take))]["all_objects"]
                    assert len(all_objects) == 1
                    all_objects = all_objects.iloc[0]
                
                    # initialize dictionary
                    sequence_data = {}
                    sequence_data = {k:{} for k in ["main_2d_objects","all_2d_objects","main_3d_objects","all_3d_objects",
                                                    "person"]}
                    
                    # metadata
                    sequence_data["metadata"] = {}
                    sequence_data["metadata"]["subject"]        = os.path.basename(subject) # subject_1
                    sequence_data["metadata"]["main_action"]    = os.path.basename(task)    # task_1_k_cooking
                    sequence_data["metadata"]["take"]           = os.path.basename(take)    # take_0
                    sequence_data["metadata"]["filename"]       = take                      # /home_nfs/haziq/datasets/KIT-Bimanual-Actions/bimacs_derived_data/subject_1/task_1_k_cooking_with_bowls/take_0
                    
                    # table center
                    key_idx = next(v for i,v in enumerate(normdata["key_indices"]) if v > cam_norm_idx)
                    sequence_data["table_center"] = np.array([normdata["map"][str(key_idx)]["offset_rl"],normdata["map"][str(key_idx)]["offset_h"],normdata["map"][str(key_idx)]["offset_d"]])
                    sequence_data["angle"] = normdata["map"][str(key_idx)]["angle"]
                    
                    # camera data
                    sequence_data["camera_data"] = {}
                    sequence_data["camera_data"]["cx"] = 320
                    sequence_data["camera_data"]["cy"] = 240
                    sequence_data["camera_data"]["fx"] = 628.0353617616482
                    sequence_data["camera_data"]["fy"] = 579.4112549695428
                    
                    # # # # # # # # # # # # # # # #
                    # read spatial relation data  #
                    # # # # # # # # # # # # # # # #
                    
                    relation_folder = os.path.join(take,"spatial_relations")
                    relation_files  = glob(os.path.join(relation_folder,"*"))
                    relation_files  = natsorted(relation_files)
                    
                    """for t,relation_file in enumerate(relation_files):
                                                           
                        # object data at current frame
                        relation_data = json.load(open(relation_file,"r"))
                        for relation_data_i in relation_data:
                            if relation_data_i["object_index"] == 12 or relation_data_i["subject_index"] == 12:
                                print(take)
                            
                            if relation_data_i["object_index"] > 12 or relation_data_i["subject_index"] > 12:
                                print("ERROR")
                                sys.exit()
                            #print(relation_data_i["object_index"])
                            #print(relation_data_i["object_index"] == 1)
                            #if relation_data_i["object_index"] == 14 or relation_data_i["subject_index"] == 15:
                            #    print("NICE")
                            #    sys.exit()
                            #sys.exit()
                            #if (relation_data_i["object_index"] == 14 and relation_data_i["relation_name"] == "contact") or (relation_data_i["subject_index"] == 15 and relation_data_i["relation_name"] == "contact"):
                            #    print(relation_data_i)"""
                    
                    # # # # # # # # # # # # # # #
                    # read bimanual action data #
                    # # # # # # # # # # # # # # #
                    
                    # even [0,2,4,...] key frames
                    # odd  [1,3,5,...] action
                    """
                    {
                      "right_hand": [0, 0, 1015],
                      "left_hand": [0, 0, 1015]
                    }
                    """

                    action_file = os.path.join(self.data_root,"bimacs_rgbd_data_ground_truth",os.path.basename(subject),os.path.basename(task),os.path.basename(take)+".json")
                    action_data = json.load(open(action_file,"r"))
                    action_data = parse_action_data(action_data)

                    sequence_data["segmentation"] = {}
                    sequence_data["segmentation"]["lhand"] = action_data["left_hand"]
                    sequence_data["segmentation"]["rhand"] = action_data["right_hand"]

                    # # # # # # # # # #
                    # read pose data  #
                    # # # # # # # # # #
                    
                    pose_folder = os.path.join(take,"body_pose")
                    pose_files = glob(os.path.join(pose_folder,"*"))
                    pose_files = natsorted(pose_files)
                    sequence_data["time"] = np.array([t for t in range(len(pose_files))])
                    
                    assert len(pose_files) == len(sequence_data["segmentation"]["lhand"])
                    assert len(pose_files) == len(sequence_data["segmentation"]["rhand"])
                    
                    for t,pose_file in enumerate(pose_files):
                    
                        # pose data at current frame
                        pose_data = json.load(open(pose_file,"r"))
                        pose_data = pose_data[most_confident_pose_idx(pose_data)]
                        
                        # pose joint xy coordinates
                        # l_arm_mocap_idxs = [1,5]
                        # r_arm_mocap_idxs = [10,14]
                        xy = np.array([[pose_data[joint]["x"], pose_data[joint]["y"]] for joint in ['LEar', 'LElbow', 'LEye', 'LHip', 'LShoulder', 'LWrist', 'MidHip', 'Neck', 'Nose', 'REar', 'RElbow', 'REye', 'RHip', 'RShoulder', 'RWrist']])
                        xy = xy / np.array([640,480])
                        confidence = np.array([pose_data[joint]["confidence"] for joint in ['LEar', 'LElbow', 'LEye', 'LHip', 'LShoulder', 'LWrist', 'MidHip', 'Neck', 'Nose', 'REar', 'RElbow', 'REye', 'RHip', 'RShoulder', 'RWrist']])
                        #print(xy[7] * np.array([640,480])) #to verify against visualize-pose-bbox-on-image.py
                        
                        # initialize entry
                        if "pose_2d" not in sequence_data["person"]:
                            sequence_data["person"]["pose_2d"] = {}
                            sequence_data["person"]["pose_2d"]["time"] = []
                            sequence_data["person"]["pose_2d"]["xy"] = []
                            sequence_data["person"]["pose_2d"]["confidence"] = []
                            
                        # collect
                        sequence_data["person"]["pose_2d"]["time"].append(t)                 # [t]    
                        sequence_data["person"]["pose_2d"]["xy"].append(xy)                  # [t, 15, 2]
                        sequence_data["person"]["pose_2d"]["confidence"].append(confidence)  # [t, 15]    
                    
                    # convert pose data to array
                    sequence_data["person"]["pose_2d"]["time"] = np.array(sequence_data["person"]["pose_2d"]["time"])
                    sequence_data["person"]["pose_2d"]["xy"] = np.array(sequence_data["person"]["pose_2d"]["xy"])
                    sequence_data["person"]["pose_2d"]["confidence"] = np.array(sequence_data["person"]["pose_2d"]["confidence"])
                    
                    # # # # # # # # # #
                    # read hand data  #
                    # # # # # # # # # #

                    hand_folder = os.path.join(take,"hand_pose")
                    hand_files  = glob(os.path.join(hand_folder,"*"))
                    hand_files  = natsorted(hand_files)
                    
                    assert len(hand_files) == len(sequence_data["segmentation"]["lhand"])
                    assert len(hand_files) == len(sequence_data["segmentation"]["rhand"])
                    
                    for t,hand_file in enumerate(hand_files):
                        
                        # hand data at current frame
                        hand_data = json.load(open(hand_file,"r"))
                        if len(hand_data) != 1:
                            # print("More than 1 detected hand at", os.path.join(args.root,"bimacs_derived_data_hand_pose",subject,action,take,"hand_pose"), "Frame", frame, flush=True)
                            # get hand with the highest score
                            hand_idx = most_confident_hand_idx(hand_data)
                            hand_data = hand_data[hand_idx:hand_idx+1]
                        hand_data = hand_data[0]
                        
                        # lhand xy and confidence
                        lhand_keys = ['LHand_0', 'LHand_1', 'LHand_10', 'LHand_11', 'LHand_12', 'LHand_13', 'LHand_14', 'LHand_15', 'LHand_16', 'LHand_17', 'LHand_18', 'LHand_19', 'LHand_2', 'LHand_20', 'LHand_3', 'LHand_4', 'LHand_5', 'LHand_6', 'LHand_7', 'LHand_8', 'LHand_9']
                        lhand_xy   = np.array([np.array([hand_data[key]["x"], hand_data[key]["y"]]) for key in lhand_keys])
                        lhand_conf = np.array([hand_data[key]["confidence"] for key in lhand_keys])
                        
                        # rhand xy and confidence
                        rhand_keys = ['RHand_0', 'RHand_1', 'RHand_10', 'RHand_11', 'RHand_12', 'RHand_13', 'RHand_14', 'RHand_15', 'RHand_16', 'RHand_17', 'RHand_18', 'RHand_19', 'RHand_2', 'RHand_20', 'RHand_3', 'RHand_4', 'RHand_5', 'RHand_6', 'RHand_7', 'RHand_8', 'RHand_9']
                        rhand_xy   = np.array([np.array([hand_data[key]["x"], hand_data[key]["y"]]) for key in rhand_keys])
                        rhand_conf = np.array([hand_data[key]["confidence"] for key in rhand_keys])
 
                        # initialize entry for lhand
                        if "lhand_2d" not in sequence_data["person"]:
                            sequence_data["person"]["lhand_2d"] = {}
                            sequence_data["person"]["lhand_2d"]["time"] = []
                            sequence_data["person"]["lhand_2d"]["xy"] = []
                            sequence_data["person"]["lhand_2d"]["confidence"] = []
                        sequence_data["person"]["lhand_2d"]["time"].append(t)                 # [t]    
                        sequence_data["person"]["lhand_2d"]["xy"].append(lhand_xy)            # [t, 15, 2]
                        sequence_data["person"]["lhand_2d"]["confidence"].append(lhand_conf)  # [t, 15] 
                        
                        # initialize entry for rhand
                        if "rhand_2d" not in sequence_data["person"]:
                            sequence_data["person"]["rhand_2d"] = {}
                            sequence_data["person"]["rhand_2d"]["time"] = []
                            sequence_data["person"]["rhand_2d"]["xy"] = []
                            sequence_data["person"]["rhand_2d"]["confidence"] = []
                        sequence_data["person"]["rhand_2d"]["time"].append(t)                 # [t]    
                        sequence_data["person"]["rhand_2d"]["xy"].append(rhand_xy)            # [t, 15, 2]
                        sequence_data["person"]["rhand_2d"]["confidence"].append(rhand_conf)  # [t, 15]   
                        
                    # convert hand data to array
                    sequence_data["person"]["lhand_2d"]["time"] = np.array(sequence_data["person"]["lhand_2d"]["time"])
                    sequence_data["person"]["lhand_2d"]["xy"] = np.array(sequence_data["person"]["lhand_2d"]["xy"])
                    sequence_data["person"]["lhand_2d"]["confidence"] = np.array(sequence_data["person"]["lhand_2d"]["confidence"])
                    
                    sequence_data["person"]["rhand_2d"]["time"] = np.array(sequence_data["person"]["rhand_2d"]["time"])
                    sequence_data["person"]["rhand_2d"]["xy"] = np.array(sequence_data["person"]["rhand_2d"]["xy"])
                    sequence_data["person"]["rhand_2d"]["confidence"] = np.array(sequence_data["person"]["rhand_2d"]["confidence"])
                                                
                    # # # # # # # # # # # # # # #
                    # read 3d object data       #
                    # - also contains the hands #
                    # # # # # # # # # # # # # # #
                                        
                    object_folder = os.path.join(take,"3d_objects")      
                    object_files = glob(os.path.join(object_folder,"*"))
                    object_files = natsorted(object_files)
                    sequence_data["time"] = np.array([t for t in range(len(object_files))])
                    
                    assert len(object_files) == len(sequence_data["segmentation"]["lhand"])
                    assert len(object_files) == len(sequence_data["segmentation"]["rhand"])
                    
                    for t,object_file in enumerate(object_files):
                                                           
                        # object data at current frame
                        object_data = json.load(open(object_file,"r"))
                             
                        for x in object_data:
                                                                          
                            # if person interacts with object
                            if any([main_object in x["instance_name"] for main_object in main_objects.keys()]):

                                # e.g. bowl_1, bowl_2
                                instance_name = x["instance_name"]        
                                bounding_box = np.array([[x["bounding_box"]["x0"],x["bounding_box"]["y0"],x["bounding_box"]["z0"]],[x["bounding_box"]["x1"],x["bounding_box"]["y1"],x["bounding_box"]["z1"]]]) # [2,3]
                                confidence = x["certainty"]
                                
                                # initialize entry
                                if instance_name not in sequence_data["main_3d_objects"]:
                                    sequence_data["main_3d_objects"][instance_name] = {}
                                    sequence_data["main_3d_objects"][instance_name]["time"] = []
                                    sequence_data["main_3d_objects"][instance_name]["bbox"] = []  
                                    sequence_data["main_3d_objects"][instance_name]["conf"] = []   
                                    
                                # collect
                                sequence_data["main_3d_objects"][instance_name]["time"].append(t)             # [t]
                                sequence_data["main_3d_objects"][instance_name]["bbox"].append(bounding_box)  # [t, 2, 3]
                                sequence_data["main_3d_objects"][instance_name]["conf"].append(confidence)    # [t]
                            
                            # all objects
                            if any([all_object in x["instance_name"] for all_object in all_objects.keys()]):

                                # e.g. bowl_1, bowl_2
                                instance_name = x["instance_name"]        
                                bounding_box = np.array([[x["bounding_box"]["x0"],x["bounding_box"]["y0"],x["bounding_box"]["z0"]],[x["bounding_box"]["x1"],x["bounding_box"]["y1"],x["bounding_box"]["z1"]]]) # [2,3]
                                confidence = x["certainty"]
                                
                                # initialize entry
                                if instance_name not in sequence_data["all_3d_objects"]:
                                    sequence_data["all_3d_objects"][instance_name] = {}
                                    sequence_data["all_3d_objects"][instance_name]["time"] = []
                                    sequence_data["all_3d_objects"][instance_name]["bbox"] = []  
                                    sequence_data["all_3d_objects"][instance_name]["conf"] = []    
                                    
                                # collect
                                sequence_data["all_3d_objects"][instance_name]["time"].append(t)             # [t]
                                sequence_data["all_3d_objects"][instance_name]["bbox"].append(bounding_box)  # [t, 2, 3]
                                sequence_data["all_3d_objects"][instance_name]["conf"].append(confidence)    # [t]
                    
                    # # # # # # # # # # # # # # #
                    # clean up main_3d_objects  # 
                    # # # # # # # # # # # # # # #
                    
                    # convert main objects to array
                    for k,v in sequence_data["main_3d_objects"].items():
                        sequence_data["main_3d_objects"][k]["time"] = np.array(sequence_data["main_3d_objects"][k]["time"])
                        sequence_data["main_3d_objects"][k]["bbox"] = np.array(sequence_data["main_3d_objects"][k]["bbox"])
                        sequence_data["main_3d_objects"][k]["conf"] = np.array(sequence_data["main_3d_objects"][k]["conf"])
                        
                    # remove duplicates based on length and true object count
                    sequence_data = remove_duplicates(sequence_data, "main_3d_objects", main_objects)
                                    
                    # make sure only got the right amount of each key
                    keys = [k.split("_")[0] for k in sequence_data["main_3d_objects"].keys()]
                    keys = Counter(keys)
                    for main_object,count in main_objects.items():
                        if keys[main_object] != count and "woodenwedge" not in main_object:
                            logger.info("take {}".format(take))
                            logger.info("main_objects {}".format(main_objects))
                            logger.info("keys {}".format(keys))
                            sys.exit()
                    
                    # # # # # # # # # # # # # #
                    # clean up all_3d_objects # 
                    # # # # # # # # # # # # # #
                    
                    # convert main objects to array
                    #sequence_data["metadata"]["all_object_names"] = [x for x in sequence_data["all_3d_objects"].keys()]
                    for k,v in sequence_data["all_3d_objects"].items():
                        sequence_data["all_3d_objects"][k]["time"] = np.array(sequence_data["all_3d_objects"][k]["time"])
                        sequence_data["all_3d_objects"][k]["bbox"] = np.array(sequence_data["all_3d_objects"][k]["bbox"])
                        sequence_data["all_3d_objects"][k]["conf"] = np.array(sequence_data["all_3d_objects"][k]["conf"])
                        
                    # remove duplicates based on length and true object count
                    sequence_data = remove_duplicates(sequence_data, "all_3d_objects", all_objects)
                                    
                    # make sure only got the right amount of each key
                    keys = [k.split("_")[0] for k in sequence_data["all_3d_objects"].keys()]
                    keys = Counter(keys)
                    for all_object,count in all_objects.items():
                        if keys[all_object] != count and "woodenwedge" not in all_object:
                            logger.info("take {}".format(take))
                            logger.info("all_objects {}".format(all_objects))
                            logger.info("keys {}".format(keys))
                            sys.exit()

                    # suppress their movements
                    main_3d_objects = list(sequence_data["main_3d_objects"].keys())
                    for k,v in sequence_data["all_3d_objects"].items():
                        # person does not interact with object
                        if k not in main_3d_objects:
                            sequence_data["all_3d_objects"][k]["bbox"][:] = sequence_data["all_3d_objects"][k]["bbox"][0]
                    
                    # # # # # # # # # # # # #
                    # read 2d object data   #
                    # # # # # # # # # # # # #
                    
                    # object data
                    object_folder = os.path.join(take,"tracked_2d_objects")      
                    object_files = glob(os.path.join(object_folder,"*"))
                    object_files = natsorted(object_files)
                    
                    # hand data
                    hand_folder = os.path.join(take,"hand_pose")
                    hand_files  = glob(os.path.join(hand_folder,"*"))
                    hand_files  = natsorted(hand_files)
                                        
                    assert len(object_files) == len(sequence_data["segmentation"]["lhand"])
                    assert len(object_files) == len(sequence_data["segmentation"]["rhand"])
                    assert len(hand_files) == len(sequence_data["segmentation"]["rhand"])
                    
                    for t,(object_file,hand_file) in enumerate(zip(object_files,hand_files)):
                                                           
                        # object data at current frame
                        object_data = json.load(open(object_file,"r")) 
                        
                        # hand data at current frame
                        hand_data   = json.load(open(hand_file,"r"))
                        if len(hand_data) != 1:
                            # print("More than 1 detected hand at", os.path.join(args.root,"bimacs_derived_data_hand_pose",subject,action,take,"hand_pose"), "Frame", frame, flush=True)
                            # get hand with the highest score
                            hand_idx = most_confident_hand_idx(hand_data)
                            hand_data = hand_data[hand_idx:hand_idx+1]

                        # get top left and bottom right bounding box coordinates for the hands
                        hand_data = hand_data[0]
                        lhand = np.array([np.array([v["x"],v["y"]]) for k,v in hand_data.items() if "L" in k])
                        rhand = np.array([np.array([v["x"],v["y"]]) for k,v in hand_data.items() if "R" in k])
                        lhand_tl, lhand_br = get_tl_br(lhand)
                        rhand_tl, rhand_br = get_tl_br(rhand)
                        
                        # form hand data
                        hand_data = [{"instance_name":"LeftHand",  "bounding_box":[lhand_tl[0],lhand_tl[1],lhand_br[0],lhand_br[1]]},
                                     {"instance_name":"RightHand", "bounding_box":[rhand_tl[0],rhand_tl[1],rhand_br[0],rhand_br[1]]}]
                        object_data = object_data + hand_data
                        for x in object_data:
                                       
                            if "instance_name" not in x:
                                continue
                                                                       
                            # if person interacts with object
                            if any([main_object in x["instance_name"] for main_object in main_objects.keys()]):

                                # e.g. bowl_1, bowl_2
                                instance_name = x["instance_name"]    
                                
                                if "Hand" not in instance_name:
                                    h_,w_,x_,y_ = x["bounding_box"]["h"], x["bounding_box"]["w"], x["bounding_box"]["x"], x["bounding_box"]["y"]                                
                                    tlx = (x_ - w_/2)
                                    tly = (y_ - h_/2)
                                    brx = (x_ + w_/2)
                                    bry = (y_ + h_/2)
                                    bounding_box = np.array([[tlx,tly],[brx,bry]]) # [2,2]
                                    #print(object_file)
                                    #print(x["instance_name"])
                                    #print(tlx, tly, brx, bry)
                                    #input()
                                else:
                                    tlx,tly,brx,bry = x["bounding_box"][0], x["bounding_box"][1], x["bounding_box"][2], x["bounding_box"][3]
                                    bounding_box    = np.array([[tlx,tly],[brx,bry]]) # [2,2]
                                
                                # initialize entry
                                if instance_name not in sequence_data["main_2d_objects"]:
                                    sequence_data["main_2d_objects"][instance_name] = {}
                                    sequence_data["main_2d_objects"][instance_name]["time"] = []
                                    sequence_data["main_2d_objects"][instance_name]["bbox"] = []    
                                    
                                # collect
                                sequence_data["main_2d_objects"][instance_name]["time"].append(t)             # [t]
                                sequence_data["main_2d_objects"][instance_name]["bbox"].append(bounding_box)  # [t, 2, 3]
                            
                            # all objects
                            if any([all_object in x["instance_name"] for all_object in all_objects.keys()]):

                                # e.g. bowl_1, bowl_2
                                instance_name = x["instance_name"]
                                
                                if "Hand" not in instance_name:   
                                    h,w,x,y = x["bounding_box"]["h"], x["bounding_box"]["w"], x["bounding_box"]["x"], x["bounding_box"]["y"]
                                    tlx = (x - w/2)
                                    tly = (y - h/2)
                                    brx = (x + w/2)
                                    bry = (y + h/2)
                                    bounding_box = np.array([[tlx,tly],[brx,bry]]) # [2,2]
                                else:
                                    tlx,tly,brx,bry = x["bounding_box"][0], x["bounding_box"][1], x["bounding_box"][2], x["bounding_box"][3]
                                    bounding_box    = np.array([[tlx,tly],[brx,bry]]) # [2,2]
                                
                                # initialize entry
                                if instance_name not in sequence_data["all_2d_objects"]:
                                    sequence_data["all_2d_objects"][instance_name] = {}
                                    sequence_data["all_2d_objects"][instance_name]["time"] = []
                                    sequence_data["all_2d_objects"][instance_name]["bbox"] = []    
                                    
                                # collect
                                sequence_data["all_2d_objects"][instance_name]["time"].append(t)             # [t]
                                sequence_data["all_2d_objects"][instance_name]["bbox"].append(bounding_box)  # [t, 2, 3]
                                
                    # # # # # # # # # # # # # # #
                    # clean up main_2d_objects  # 
                    # # # # # # # # # # # # # # #
                    
                    # convert main objects to array
                    for k,v in sequence_data["main_2d_objects"].items():
                        sequence_data["main_2d_objects"][k]["time"] = np.array(sequence_data["main_2d_objects"][k]["time"])
                        sequence_data["main_2d_objects"][k]["bbox"] = np.array(sequence_data["main_2d_objects"][k]["bbox"])
                        
                    # remove duplicates based on length and true object count
                    sequence_data = remove_duplicates(sequence_data, "main_2d_objects", main_objects)
                                    
                    # make sure only got the right amount of each key
                    keys = [k.split("_")[0] for k in sequence_data["main_2d_objects"].keys()]
                    keys = Counter(keys)
                    for main_object,count in main_objects.items():
                        if keys[main_object] != count and "woodenwedge" not in main_object:
                            logger.info("take {}".format(take))
                            logger.info("main_objects {}".format(main_objects))
                            logger.info("keys {}".format(keys))
                            sys.exit()

                    # # # # # # # # # # # # # #
                    # clean up all_2d_objects # 
                    # # # # # # # # # # # # # #
                    
                    # convert main objects to array
                    for k,v in sequence_data["all_2d_objects"].items():
                        sequence_data["all_2d_objects"][k]["time"] = np.array(sequence_data["all_2d_objects"][k]["time"])
                        sequence_data["all_2d_objects"][k]["bbox"] = np.array(sequence_data["all_2d_objects"][k]["bbox"])
                        # (WRONG!) suppress their movements
                        # sequence_data["all_2d_objects"][k]["bbox"][:] = sequence_data["all_2d_objects"][k]["bbox"][0]
                        
                    # remove duplicates based on length and true object count
                    sequence_data = remove_duplicates(sequence_data, "all_2d_objects", all_objects)
                                    
                    # make sure only got the right amount of each key
                    keys = [k.split("_")[0] for k in sequence_data["all_2d_objects"].keys()]
                    keys = Counter(keys)
                    for all_object,count in all_objects.items():
                        if keys[all_object] != count and "woodenwedge" not in all_object:
                            logger.info("take {}".format(take))
                            logger.info("all_objects {}".format(all_objects))
                            logger.info("keys {}".format(keys))
                            sys.exit()
                    
                    """
                    get hand-object contact
                    - bottle_1 (935,)
                    - whisk_2 (835,)
                    - bowl_3 (1015,)
                    - LeftHand (1015,)
                    - RightHand (1015,)
                    
                    do it here in main loader or at every iteration?
                    """
                    
                    #for k,v in sequence_data["all_2d_objects"].items():
                    #    print(k, sequence_data["all_2d_objects"][k]["time"].shape)
                    #print(sequence_data["person"]["pose_2d"]["time"].shape)
                    #print(sequence_data["person"]["pose_2d"]["xy"].shape)
                    #sys.exit()
                    
                    # # # # # #
                    # collect #  
                    # # # # # #
                                        
                    # sequence data
                    self.sequence_data.append(sequence_data)
                    cam_norm_idx += 1
                
        # # # # # # # # # # # # # # # # # # # # # #
        # convert all data to list then store it  #
        # # # # # # # # # # # # # # # # # # # # # #
        
        sequence_data = copy.deepcopy(self.sequence_data)
        sequence_data = [dict_arr_to_list(x) for x in sequence_data]
        
        main_actions_truncated = [re.sub(r'task_.*_[kw]_', '', x) for x in self.main_actions]            
        data_folder = self.data_name + "_"
        split       = dtype+"="+str(eval("self."+dtype+"_samples")) + "_"
        actions     = str(main_actions_truncated)
        
        cached_data_path = os.path.join(self.data_root, "cached-data", \
                                        data_folder + \
                                        split + \
                                        actions)
                                        
        with open(cached_data_path, "w") as fout:
            json.dump(sequence_data, fout)