import os
import sys
import json
import copy
import torch
import logging
import inspect
import numpy as np
import xml.etree.ElementTree as ET

from bps import bps
from glob import glob

import kit_mocap_variables as var
from utils import *

# get the actions with no similar objects
def get_actions_with_no_similar_objects(objects, action_to_objects):
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
        
        # # # # # # # #
        #             #
        # human data  #
        #             #
        # # # # # # # #
        
        # mocap names
        self.mocap_names        = var.mocap_names
        self.l_arm_mocap_names  = var.l_arm_mocap_names
        self.r_arm_mocap_names  = var.r_arm_mocap_names
        self.l_arm_mocap_idxs   = var.l_arm_mocap_idxs
        self.r_arm_mocap_idxs   = var.r_arm_mocap_idxs
        self.hip_mocap_idxs     = var.hip_mocap_idxs
        setattr(args, "mocap_names", var.mocap_names)
        setattr(args, "l_arm_mocap_names", var.l_arm_mocap_names)
        setattr(args, "r_arm_mocap_names", var.r_arm_mocap_names)
        setattr(args, "l_arm_mocap_idxs", var.l_arm_mocap_idxs)
        setattr(args, "r_arm_mocap_idxs", var.r_arm_mocap_idxs)
        setattr(args, "hip_mocap_idxs", var.hip_mocap_idxs)
        setattr(args, "hand_xyz_dims", var.hand_xyz_dims)
                                
        if not hasattr(args, "hand_xyz_dims"):
            self.hand_xyz_dims = var.hand_xyz_dims
            setattr(args, "hand_xyz_dims", var.hand_xyz_dims)
                        
        """# joint names
        self.joint_names       = var.joint_names
        self.l_arm_joint_names = var.l_arm_joint_names
        self.r_arm_joint_names = var.r_arm_joint_names
        self.l_arm_joint_idxs  = var.l_arm_joint_idxs
        self.r_arm_joint_idxs  = var.r_arm_joint_idxs
        self.extended_joint_names = var.extended_joint_names
        self.extended_joint_axis  = var.extended_joint_axis
        setattr(args, "extended_joint_idx_to_name", var.extended_joint_idx_to_name)
        setattr(args, "extended_joint_idx_axis", var.extended_joint_idx_axis)"""
        
        """# forward kinematics data
        self.link_idx_order_to_left_wrist  = var.link_idx_order_to_left_wrist
        self.link_idx_order_to_right_wrist = var.link_idx_order_to_right_wrist
        self.link_idx_direction_dict = var.link_idx_direction_dict
        self.link_idx_length_dict    = var.link_idx_length_dict
        self.extended_parent_idx_dict = var.extended_parent_idx_dict
        setattr(args, "link_idx_direction_dict", var.link_idx_direction_dict)
        setattr(args, "link_idx_length_dict", var.link_idx_length_dict)
        setattr(args, "extended_parent_idx_dict", var.extended_parent_idx_dict)"""
        
        setattr(args, "num_body_joints", var.num_body_joints)
        setattr(args, "body_dim", var.body_dim)
        setattr(args, "num_obj_markers", var.num_obj_markers)
        setattr(args, "obj_dim", var.obj_dim)
        setattr(args, "num_hands", var.num_hands)
        setattr(args, "finger_dim", var.hand_dim)
        
        # # # # # # # #
        #             #
        # object data #
        #             #
        # # # # # # # # 
        
        # object bps
        self.bps = {}
        vertex_folders = glob(os.path.join(os.path.expanduser("~"),"MMMTools/data/Model/Objects/","*"))
        for vertex_folder in vertex_folders:
            filenames = glob(os.path.join(vertex_folder,"*json"))
            for filename in filenames:
                #print(filename)
                data = json.load(open(filename,"r"))
                bps_random = np.array(data["bps"])
                self.bps[filename] = bps_random
                            
        # object mocap names
        self.object_mocap_names = var.object_mocap_names
        
        # load object mocap markers
        self.object_mocap_markers       = var.object_mocap_markers     
        self.object_mocap_marker_path   = var.object_mocap_marker_path
        
        # object names
        self.all_objects = var.all_objects
        #setattr(args, "all_objects", self.all_objects)
        
        # object dimensions
        setattr(self, "obj_dimensions", var.obj_dimensions)
        setattr(self, "tgt_ref_dict", var.tgt_ref_dict)
        
        # # # # # # # #
        #             #
        # action data #
        #             #
        # # # # # # # # 
        
        # sort with main_actions first followed by the other fine-grained actions
        self.clf_actions = self.main_actions + [x for x in self.fine_actions if x not in self.main_actions]
        setattr(args, "clf_actions", self.clf_actions)
        self.all_actions = self.main_actions + [x for x in self.fine_actions if x not in self.main_actions]
        self.action_to_id = {a:i for i,a in enumerate(self.all_actions)}
        self.id_to_action = {i:a for i,a in enumerate(self.all_actions)}
        for k,v in self.action_to_id.items():
            logger.info("{} {}".format(v, k))
        setattr(args, "id_to_action", self.id_to_action)
        self.main_action_ids = [self.action_to_id[x] for x in self.main_actions]
        self.all_action_ids  = [self.action_to_id[x] for x in self.all_actions]
        setattr(args, "main_action_ids", self.main_action_ids)
        setattr(args, "all_action_ids", self.all_action_ids)

        # action -> objects
        self.action_to_objects = var.action_to_objects
        
        # For each action, get all other actions with no similar objects
        self.actions_with_no_similar_objects = {}
        for action,objects in self.action_to_objects.items():
            #if self.allow_similar_objects == 1:
            #    x = list(self.action_to_objects.keys())
            #    x.remove(action)
            #    self.actions_with_no_similar_objects[action] = x
            #else:
            #    self.actions_with_no_similar_objects[action] = get_actions_with_no_similar_objects(objects,self.action_to_objects)
            self.actions_with_no_similar_objects[action] = get_actions_with_no_similar_objects(objects,self.action_to_objects)
        #for k,v in self.actions_with_no_similar_objects.items():
        #    print(k, v)
        #sys.exit()
        """
        Close ['Cut', 'Mix', 'Peel', 'RollOut', 'Scoop', 'Stir', 'Transfer', 'Wipe']
        Cut ['Close', 'Mix', 'Open', 'Pour', 'RollOut', 'Scoop', 'Stir']
        Mix ['Close', 'Cut', 'Open', 'Pour', 'RollOut']
        Open ['Cut', 'Mix', 'Peel', 'RollOut', 'Scoop', 'Stir', 'Transfer', 'Wipe']
        Peel ['Close', 'Open', 'Pour', 'RollOut']
        Pour ['Cut', 'Mix', 'Peel', 'RollOut', 'Transfer', 'Wipe']
        RollOut ['Close', 'Cut', 'Mix', 'Open', 'Peel', 'Pour', 'Scoop', 'Stir', 'Transfer', 'Wipe']
        Scoop ['Close', 'Cut', 'Open', 'RollOut']
        Stir ['Close', 'Cut', 'Open', 'RollOut']
        Transfer ['Close', 'Open', 'Pour', 'RollOut']
        Wipe ['Close', 'Open', 'Pour', 'RollOut']
        """
        
        
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
        # - 0 is for the padded objects
        self.object_name_to_id = {k:i+1 for i,k in enumerate(action_to_objects)}
        self.object_id_to_name = {i+1:k for i,k in enumerate(action_to_objects)}
        for i,action_to_object in enumerate(action_to_objects):
            logger.info("{} {}".format(i+1, action_to_object))
        setattr(args, "object_id_to_name", self.object_id_to_name)

        # padded = 0, actual objects = 1,2,3
        # +1 for the padded object
        # +2 for the wrists
        self.num_obj_classes        = len(action_to_objects) + 1
        self.num_obj_body_classes   = len(action_to_objects) + 1 + 1
        self.num_obj_wrist_classes  = len(action_to_objects) + 1 + 2
        setattr(args, "num_obj_classes",         self.num_obj_classes)
        setattr(args, "num_obj_wrist_classes",   self.num_obj_wrist_classes)
        setattr(args, "obj_padded_length",       self.object_padded_length)
        setattr(args, "obj_body_padded_length",  self.object_padded_length+1)
        setattr(args, "obj_wrist_padded_length", self.object_padded_length+2)
        
        # # # # # # # # # #   
        #                 #
        # load from cache #
        #                 #
        # # # # # # # # # #
                                
        # path to cached data if it was not provided
        if args.cached_data_path == None:        
            data_folder         = self.data_name + "_"                                                          # data-sorted-simpleGT-v4-xml-only
            split               = "kit_"+dtype+"="+str(eval("self."+dtype+"_samples")) + "_"                    # kit_train=[0,1]
            #resolution          = "res="+str(self.resolution) + "_"                                             # res=0.02
            #time_step_size      = "time-step-size="+str(self.time_step_size) + "_"                              # time-step-size=2.0
            actions             = str(self.main_actions)                                                        # ["Cut","Mix",...]
            cached_data_path = os.path.join(self.data_root, "cached_data", \
                                            data_folder + \
                                            split + \
                                            #resolution + \
                                            #time_step_size + \
                                            actions)
                                            
        if os.path.isfile(cached_data_path):
                
            logger.info("Loading dataset from {}".format(cached_data_path))
            with open(cached_data_path) as f:
                sequence_data = json.load(f)
                self.sequence_data = sequence_data
                self.sequence_data = [dict_list_to_arr(x,skip=["filename","object_names","object_paths","joint_names","mocap_names","action","object"]) for x in self.sequence_data]
            
            action_counter = {}
            for x in self.sequence_data:
                main_action = x["metadata"]["main_action"]
                if main_action not in action_counter.keys():
                    action_counter[main_action] = 1
                else:
                    action_counter[main_action] += 1
            for k,v in action_counter.items():
                logger.info("{} {}".format(k,v))
            return
            
        # # # # # # # # # #    
        #                 #        
        # load from files #
        #                 #
        # # # # # # # # # #
        
        self.sequence_data = []
        
        # each action_folder contains the motion folders
        action_folders = sorted(glob(os.path.join(self.data_root,self.data_name,"*")))
        action_folders = [x for x in action_folders if os.path.isdir(x)]
        action_folders = [x for x in action_folders if any([y == os.path.basename(x) for y in self.main_actions])]    # ['~/datasets/KIT-Dataset/data-sorted-simpleGT-v1/Cut', ...]
        
        # sort action_folders following main_actions
        action_folders_ = []
        for x in self.main_actions:
            for action_folder in action_folders:
                if x in action_folder:
                    action_folders_.append(action_folder)
        assert len(action_folders_) == len(action_folders)
        action_folders = action_folders_

        for x,y in zip(action_folders, self.sample_ratio):
            logger.info("{} {}".format(x,y))

        sequence_data_id = 0
        for action_folder,sample_ratio in zip(action_folders,self.sample_ratio):
            
            # each folder can contain multiple sequences
            motion_folders = sorted(glob(os.path.join(action_folder,"*")))
            motion_folders = [x for x in motion_folders if "comments" not in x and "txt" not in x]
            
            sample_ratio1 = int(sample_ratio)
            sample_ratio2 = sample_ratio - sample_ratio1            
            motion_folders_sampled1 = motion_folders * sample_ratio1
            motion_folders_sampled2 = motion_folders[:int(sample_ratio2*len(motion_folders))]
            motion_folders = motion_folders_sampled1 + motion_folders_sampled2
                        
            for motion_folder in motion_folders:
                
                # motion_files are only those with xml as the extension
                motion_files = sorted(glob(os.path.join(motion_folder,"*xml")))
                for m,motion_file in enumerate(motion_files):

                    # check if the current sample m is in the train/val set
                    if not m in eval("self."+dtype+"_samples"): #not any([m == sample for sample in eval("self."+dtype+"_samples")]):
                        continue
                                        
                    # check if the current sequence is in self.main_actions
                    sequence_filename = motion_file # /home_nfs/haziq/datasets/KIT/data/p1/files_motions_3060/Scoop17_salad_GB_pl_0_40_01.xml                    
                    if not any([main_action in sequence_filename for main_action in self.main_actions]):
                        logger.info("Skipping {} because it is not in self.main_actions: {}".format(sequence_filename, self.main_actions))
                        continue
                    
                    # # # # # # # # # # #    
                    #                   #        
                    # parse motion file #
                    #                   #
                    # # # # # # # # # # #                    
                    
                    logger.info("{} Processing {} {}".format(m, dtype, sequence_filename))
                    root = ET.parse(sequence_filename)
        
                    """
                    read the *.xml file into 
                    sequence_data that will contain
                    
                    note that the variable t may not be the same
                    
                    - [metadata]
                        - [object_names] = [object1,object2]
                        - [object_paths] = [...]
                        - [subject_id]  = 1040
                        - [filename] = files_motions_3072_Close1_aj_180_30h_03
        
                    - [root]
                        - [root][body]
                            - [root][body][time]             # [t]
                            - [root][body][root_position]    # [t,3]
                            - [root][body][root_rotation]    # [t,3]
                        - [root][object1]
                            - [root][object1][time]          # [t]
                            - [root][object1][root_position] # [t,3]
                            - [root][object1][root_rotation] # [t,3]
                        - [root][object2]
                            - [root][object1][time]          # [t]
                            - [root][object1][root_position] # [t,3]
                            - [root][object1][root_rotation] # [t,3]
                    
                    - [kinematics]
                        - [kinematics][body]
                            - [kinematics][body][joint_names]      # [44]
                            - [kinematics][body][time]             # [t]
                            - [kinematics][body][joint_values]     # [t, 44]
                        - [kinematics][lhand]
                            - [kinematics][lhand][joint_names]     # [19]
                            - [kinematics][lhand][time]            # [t]
                            - [kinematics][lhand][joint_values]    # [t, 19]
                        - [kinematics][rhand]
                            - [kinematics][rhand][joint_names]     # [19]
                            - [kinematics][rhand][time]            # [t]
                            - [kinematics][rhand][joint_values]    # [t, 19]
                    
                    - [mocap]
                        - [mocap][body]
                            - [mocap][body][mocap_names]          # [self.markers_in_use]
                            - [mocap][body][time]                 # [t]
                            - [mocap][body][mocap_values]         # [t, self.markers_in_use, 3] (instead of [self.markers_in_use, t, 3] since its easier to interpolate)
                            
                    - [segmentation]
                        - [segmentation][lhand]
                            - [segmentation][lhand][time]
                            - [segmentation][lhand][action]
                            - [segmentation][lhand][object]
                            
                    """   
                    
                    # initialize empty dictionary
                    sequence_data = {}
                    sequence_data = {k:{} for k in ["metadata","root","kinematics","mocap","segmentation"]}
                    sequence_data["metadata"]["filename"] = sequence_filename
                    #sequence_data["metadata"]["motion_folder_num"] = motion_folder_num
                    sequence_data["metadata"]["main_action"] = os.path.basename(action_folder)

                    # # # # # # # # # # #    
                    #                   #        
                    # get segmentation  #
                    #                   #
                    # # # # # # # # # # #
                    
                    sequence_data["segmentation"] = {}
                    
                    # get the left and right hand nodes
                    hand_nodes = root.findall("Segment/Segmentation")
                    for hand_node, hand in zip(hand_nodes,["lhand","rhand"]):
                        
                        # create sub dictionary for lhand/rhand
                        sequence_data["segmentation"][hand] = {}
                                                
                        # get the segmentation nodes that contain the timestep and annotation child node
                        # - action_list contains the level 1 segmentations 
                        time_list, action_list, action_id_list, object_list, object_id_list, target_object_list, target_object_id_list = [], [], [], [], [], [], []
                        segmentation_nodes = hand_node.findall("Segment")
                        previous_object = None
                        for segmentation_node in segmentation_nodes:
                                                    
                            # get [time]
                            start_time, end_time = float(segmentation_node.attrib["start"]), float(segmentation_node.attrib["end"])
                            time_list.append(np.array([start_time, end_time]))
                            
                            # get [action] and [action_id]
                            action = segmentation_node.find("Annotation").text
                            action_list.append(action)
                            action_id_list.append(self.action_to_id[action])
                            
                            # get main object i.e. object in contact
                            if "main" in segmentation_node.find("Annotation").attrib and any([segmentation_node.find("Annotation").attrib["main"] == x for x in action_to_objects]):
                                object_list.append(segmentation_node.find("Annotation").attrib["main"])
                                object_id_list.append(self.object_name_to_id[segmentation_node.find("Annotation").attrib["main"]])
                            
                            # i cannot find the main object in the xml file
                            else:      

                                # current action states that the person should not be holding anything
                                if any([action == x for x in ["Idle","Approach","Retreat"]]):
                                    object_list.append("None")
                                    object_id_list.append(-1)
                                
                                # current action states that the person should be holding something but is missing in the xml file
                                else:
                                
                                    # get previously held object
                                    if object_id_list[-1] != -1:
                                        object_list.append(object_list[-1])
                                        object_id_list.append(object_id_list[-1])
                                    
                                    # error
                                    else:
                                        print("Invalid segmentation in:", sequence_filename)
                                        print(hand, action, start_time, end_time)
                                        sys.exit()
                                
                            # get target object i.e. object being approached
                            if "target" in segmentation_node.find("Annotation").attrib and any([segmentation_node.find("Annotation").attrib["target"] == x for x in action_to_objects]):
                                target_object_list.append(segmentation_node.find("Annotation").attrib["target"])
                                target_object_id_list.append(self.object_name_to_id[segmentation_node.find("Annotation").attrib["target"]])
                            else:
                                target_object_list.append("None")
                                target_object_id_list.append(-1)
                            
                        # segmentation data for current hand
                        sequence_data["segmentation"][hand]["time"] = np.array(time_list)               # [t, 2]
                        sequence_data["segmentation"][hand]["action"] = action_list                     # [t]
                        sequence_data["segmentation"][hand]["action_id"] = action_id_list               # [t]
                        sequence_data["segmentation"][hand]["object"] = object_list                     # [t]
                        sequence_data["segmentation"][hand]["object_id"] = object_id_list               # [t]
                        sequence_data["segmentation"][hand]["target"] = target_object_list              # [t]
                        sequence_data["segmentation"][hand]["target_id"] = target_object_id_list        # [t]
                    
                        if len(list(set(sequence_data["segmentation"][hand]["object_id"]))) > 2:
                            logger.info("{} {} has incorrect number of interactions!".format(sequence_filename, hand))
                            logger.info(object_list)
                            sys.exit()
                    
                    # # # # # # # # # # # # #   
                    #                       #        
                    # get body data (root)  #
                    #                       #
                    # # # # # # # # # # # # #
                    
                    sequence_data["root"] = {}
                    sequence_data["root"]["body"] = {}
                    
                    # get subject id
                    body_node = root.find("Motion[@name='1723']") or root.find("Motion[@name='1480']")
                    sequence_data["metadata"]["subject_id"] = body_node.attrib["name"]
                    
                    # get subject height
                    sequence_data["metadata"]["subject_height"] = float(body_node.find("Model/ModelProcessorConfig/Height").text)
                    sequence_data["metadata"]["subject_mass"] = float(body_node.find("Model/ModelProcessorConfig/Mass").text)
                    sequence_data["metadata"]["subject_hand_length"] = float(body_node.find("Model/ModelProcessorConfig/HandLength").text)
                    
                    # get subject path
                    # - /home/haziq/MMMTools/data/Model/Objects/cucumber_attachment/cucumber_cut.xml
                    #sequence_data["metadata"]["subject_path"] = [body_node.find("Model").attrib["path"] for object_node in object_nodes]
                    
                    # get body ModelPose node
                    body_modelpose_node = root.find("Motion[@name='1723']/Sensors/Sensor[@type='ModelPose']") or root.find("Motion[@name='1480']/Sensors/Sensor[@type='ModelPose']")
                                        
                    # get measurement_nodes that contain the timestep attribute and root_position and root_rotation nodes
                    measurement_nodes = body_modelpose_node.findall("Data/Measurement")
                    
                    # get the nodes
                    time_list, root_position_list, root_rotation_list = [], [], []
                    for measurement_node in measurement_nodes:
                        
                        # get [time]
                        time = measurement_node.attrib["timestep"]
                        time_list.append(float(time))
                        
                        # get [root_position]
                        root_position = measurement_node.find("RootPosition").text.split()
                        root_position = [float(x) for x in root_position]
                        root_position_list.append(root_position)
                        
                        # get [root_rotation]
                        root_rotation = measurement_node.find("RootRotation").text.split()
                        root_rotation = [float(x) for x in root_rotation]
                        root_rotation_list.append(root_rotation) 
                    
                    sequence_data["root"]["body"]["time"] = np.array(time_list)                   # [t]
                    sequence_data["root"]["body"]["root_position"] = np.array(root_position_list) # [t, 3]
                    sequence_data["root"]["body"]["root_rotation"] = np.array(root_rotation_list) # [t, 3]
                    
                    # # # # # # # # # # # # #   
                    #                       #        
                    # get body data (mocap) #
                    #                       #
                    # # # # # # # # # # # # #                    

                    sequence_data["mocap"] = {}                    
                    sequence_data["mocap"]["body"] = {}
                    
                    # get body MoCapMarker node
                    body_mocap_node = root.find("Motion[@name='1723']/Sensors/Sensor[@type='MoCapMarker']") or root.find("Motion[@name='1480']/Sensors/Sensor[@type='MoCapMarker']")
                    
                    # get measurement_nodes that contain the timestep attribute and mocap child nodes
                    measurement_nodes = body_mocap_node.findall("Data/Measurement")
                    
                    # get [mocap_names]
                    mocap_names = measurement_nodes[0].findall("MarkerPosition")
                    mocap_names_list = [marker_name.attrib["name"] for marker_name in mocap_names]
                    sequence_data["mocap"]["body"]["mocap_names"] = mocap_names_list # [len(full_mocap_markers_names]
                                        
                    # get the nodes
                    time_list, mocap_values_list = [], [[] for _ in range(len(mocap_names))]
                    for measurement_node in measurement_nodes:
                    
                        # get [time]
                        time = measurement_node.attrib["timestep"]
                        time_list.append(float(time))
                                                
                        # get [mocap_values]
                        marker_value_nodes = measurement_node.findall("MarkerPosition")
                        for i,marker_value_node in enumerate(marker_value_nodes):
                            mocap_values = marker_value_node.text.split()
                            mocap_values = [float(x) for x in mocap_values]
                            mocap_values_list[i].append(mocap_values) # [len(mocap_names), t, 3]
                    
                    sequence_data["mocap"]["body"]["time"] = np.array(time_list) # [t]
                    sequence_data["mocap"]["body"]["mocap_values"] = np.transpose(np.array(mocap_values_list),axes=[1,0,2]) # [t, len(full_mocap_markers_names), 3]

                    # # # # # # # # # # # # # # # #  
                    #                             #        
                    # get body data (kinematics)  #
                    #                             #
                    # # # # # # # # # # # # # # # #

                    sequence_data["joint"] = {}
                    sequence_data["joint"]["body"]  = {}
                    sequence_data["joint"]["lhand"] = {}
                    sequence_data["joint"]["rhand"] = {}
                    
                    # get Kinematic nodes
                    body_kinematic_nodes = root.findall("Motion[@name='1723']/Sensors/Sensor[@type='Kinematic']") or root.findall("Motion[@name='1480']/Sensors/Sensor[@type='Kinematic']")
                    for bodypart_node,prefix,prefix_num_joints in zip(body_kinematic_nodes,["body","rhand","lhand"],[44,19,19]):
                                            
                        # get [prefix][joint_names]
                        joint_names = bodypart_node.findall("Configuration/Joint")
                        joint_names = [joint_name.attrib["name"] for joint_name in joint_names]
                        sequence_data["joint"][prefix]["joint_names"] = joint_names # [body=44, hand=19]

                        if prefix == "rhand":
                            assert "Right" in joint_names[0]
                        if prefix == "lhand":
                            assert "Left" in joint_names[0]

                        # get measurement_nodes that contain the timestep attribute and JointPosition child node
                        time_list, joint_values_list = [], []
                        measurement_nodes = bodypart_node.findall("Data/Measurement")
                        for measurement_node in measurement_nodes:
                        
                            # get [time]
                            time = measurement_node.attrib["timestep"]
                            time_list.append(float(time))
                            
                            # get [joint_values]
                            joint_values = measurement_node.find("JointPosition").text.split()
                            joint_values = [float(joint_value) for joint_value in joint_values] 
                            joint_values_list.append(joint_values)
                            assert len(joint_values) == prefix_num_joints
                            
                        sequence_data["joint"][prefix]["time"] = np.array(time_list)                 # [t]
                        sequence_data["joint"][prefix]["joint_values"] = np.array(joint_values_list) # [t,body=44 / hand=19]
                    
                    # # # # # # # # # # # # # #
                    #                         #        
                    # get object data (root)  #
                    #                         #
                    # # # # # # # # # # # # # #
                    
                    # get object nodes present in current scene
                    object_nodes = [root.find("Motion[@name='"+x+"']") for x in self.all_objects]
                    object_nodes = [x for x in object_nodes if x is not None]
                    assert len(object_nodes) != 0

                    # get [object_names]
                    sequence_data["metadata"]["object_names"] = [object_node.attrib["name"] for object_node in object_nodes]
                    
                    # get [object_paths]
                    # - /home/haziq/MMMTools/data/Model/Objects/cucumber_attachment/cucumber_cut.xml
                    sequence_data["metadata"]["object_paths"] = [object_node.find("Model").attrib["path"] for object_node in object_nodes]
                    
                    # make sure action_to_objects contain the object names
                    for x in sequence_data["metadata"]["object_names"]:
                        if x != "kitchen_sideboard":
                            if not(x in self.action_to_objects[os.path.basename(action_folder)]):
                                logger.info("Missing object {} for action {}".format(x, os.path.basename(action_folder)))
                                sys.exit()
                    for object_node in object_nodes:
                                                
                        # create sub dictionary for object
                        object_name = object_node.attrib["name"]
                        sequence_data["root"][object_name] = {}
                        
                        # get nodes for the current object
                        time_list, root_position_list, root_rotation_list = [],[],[]
                        measurement_nodes = object_node.findall("Sensors/Sensor[@type='ModelPose']/Data/Measurement")
                        for measurement_node in measurement_nodes:
                            
                            # get [time]
                            time = measurement_node.attrib["timestep"]
                            time_list.append(float(time))
                            
                            # get [root_position]
                            root_position = measurement_node.find("RootPosition").text.split()
                            root_position = [float(x) for x in root_position]
                            root_position_list.append(root_position)
                            
                            # get [root_rotation]
                            root_rotation = measurement_node.find("RootRotation").text.split()
                            root_rotation = [float(x) for x in root_rotation]
                            root_rotation_list.append(root_rotation) 

                        sequence_data["root"][object_name]["time"] = np.array(time_list)
                        sequence_data["root"][object_name]["root_position"] = np.array(root_position_list)
                        sequence_data["root"][object_name]["root_rotation"] = np.array(root_rotation_list)
                    
                    
                    # # # # # # # # # # # # # #
                    #                         #        
                    # get object data (mocap) #
                    #                         #
                    # # # # # # # # # # # # # #
                    
                    for object_node in object_nodes:
                        
                        # create sub dictionary for object
                        object_name = object_node.attrib["name"]
                        sequence_data["mocap"][object_name] = {}
                        sequence_data["mocap"][object_name]["mocap_names"] = var.object_mocap_names[object_name]
                        
                        # get nodes for the current object
                        time_list, mocap_values_list = [], [[] for _ in range(len(var.object_mocap_names[object_name]))]
                        measurement_nodes = object_node.findall("Sensors/Sensor[@type='MoCapMarker']/Data/Measurement")
                        for measurement_node in measurement_nodes:
                            
                            # get [time]
                            time = measurement_node.attrib["timestep"]
                            time_list.append(float(time))
                                                        
                            # get [mocap_values]
                            marker_value_nodes = measurement_node.findall("MarkerPosition")[:len(var.object_mocap_names[object_name])]
                            for i,marker_value_node in enumerate(marker_value_nodes):
                                mocap_values = marker_value_node.text.split()
                                mocap_values = [float(x) for x in mocap_values]
                                mocap_values_list[i].append(mocap_values) # [len(mocap_names), t, 3]                            

                        sequence_data["mocap"][object_name]["time"] = np.array(time_list)
                        sequence_data["mocap"][object_name]["mocap_values"] = np.transpose(np.array(mocap_values_list),axes=[1,0,2])
                        
                    
                    # sequence data
                    self.sequence_data.append(sequence_data)
                    
        # # # # # # # # # # # # # # # # # # # #
        # convert data to list then store it  #
        # # # # # # # # # # # # # # # # # # # #
        
        sequence_data = copy.deepcopy(self.sequence_data)
        sequence_data = [dict_arr_to_list(x) for x in sequence_data]
        
        data_folder         = self.data_name + "_"                                                          # data-sorted-simpleGT-v4-xml-only
        split               = "kit_"+dtype+"="+str(eval("self."+dtype+"_samples")) + "_"                    # kit_train=[0,1]
        #resolution          = "res="+str(self.resolution) + "_"                                             # res=0.02
        #time_step_size      = "time-step-size="+str(self.time_step_size) + "_"                              # time-step-size=2.0
        actions             = str(self.main_actions)                                                        # ["Cut","Mix",...]
        cached_data_path    = os.path.join(self.data_root, "cached_data", \
                                           data_folder + \
                                           split + \
                                           #resolution + \
                                           #time_step_size + \
                                           actions)
                                        
        with open(cached_data_path, "w") as fout:
            json.dump(sequence_data, fout)