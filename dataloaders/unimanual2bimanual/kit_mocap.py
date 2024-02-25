import os
import sys
import ast
import copy
import json
import time
import math
import torch 
import random
import numpy as np

from tqdm import tqdm
from scipy import interpolate
from collections import Counter, OrderedDict
   
sys.path.append(os.path.join(os.path.expanduser("~"),"Forecasting-Bimanual-Object-Manipulation-Sequences-From-Unimanual-Observations","dataloaders"))
from kit_mocap_main_loader import *
import kit_mocap_variables as var
            
class dataloader(torch.utils.data.Dataset):
    def __init__(self, args, dtype):
            
        # logger
        logger = logging.getLogger(os.path.join(args.log_root,args.log_name))
        
        """
        load data    
        """
        
        t1 = time.time() 
        data = main_loader(args, dtype)
        t2 = time.time() 
        logger.info("Loading time: {}".format(t2-t1))
        
        # pass all to self
        for key, value in data.__dict__.items():
            setattr(self, key, value)

        t1 = time.time()     
        
        """
        update object data      
        """

        for i in range(len(self.sequence_data)):
            filename = self.sequence_data[i]["metadata"]["filename"]
        
            for hand in ["lhand","rhand"]:                
                action_list     = self.sequence_data[i]["segmentation"][hand]["action"]
                object_list     = self.sequence_data[i]["segmentation"][hand]["object"]
                object_id_list  = self.sequence_data[i]["segmentation"][hand]["object_id"]
                target_list     = self.sequence_data[i]["segmentation"][hand]["target"]
                target_id_list  = self.sequence_data[i]["segmentation"][hand]["target_id"]
                
                previous_object_id = None
                for t,(current_action,current_object,current_object_id,current_target_id) in enumerate(zip(action_list,object_list,object_id_list,target_id_list)):
                    
                    # current action is such that hand should not be holding onto anything
                    # - Idle, Approach, Retreat
                    if current_action in ["Idle","Approach","Retreat"]:
                    
                        object_id_list[t]   = -1
                        object_id_list[t]   = -1
                        previous_object_id  = -1
                        
                        # look for target if hand approaching an object
                        if current_action == "Approach" and current_target_id != -1:
                            previous_object_id = current_target_id
                        
                        # (only happens if the ground truth is erroneous for the current action but correct later on)
                        #if current_object is not None:
                        if current_object_id != -1:
                            previous_object_id = current_object_id                            
                        
                    # current action is such that hand should be holding onto something
                    # - Hold + Cucumber, Cut + Knife
                    elif current_action in self.main_actions + ["Move", "Hold", "Place"]:
                    
                        # if ground truth is erroneous and says that the hand is not holding anything
                        if current_object_id == -1:
                            # get the object the hand was previously holding onto
                            if previous_object_id != -1:
                                object_list[t]      = self.object_id_to_name[previous_object_id]
                                object_id_list[t]   = previous_object_id
                            # assert if nothing
                            else:
                                logger.info("Hand is supposed to be holding onto something but the ground truth says otherwise and previous_object = {}!".format(previous_object))
                                logger.info(filename)
                                logger.info("{} {}".format(hand, object_list))
                                sys.exit()
                        
                        # if hand holding onto something
                        # - no need to update the list
                        elif current_object_id != -1:
                            previous_object_id = current_object_id
                            
                        else:
                            logger.info("Unknown case 1!")
                            logger.info(filename)
                            logger.info("{} {}".format(hand, object_list))
                            sys.exit()            
                
                # assert
                if len(list(set(object_id_list))) > 2:
                    logger.info("{} {} has incorrect number of interactions!".format(filename, hand))
                    logger.info(object_list)
                    sys.exit()
                    
                # update list
                self.sequence_data[i]["segmentation"][hand]["object"]           = object_list
                self.sequence_data[i]["segmentation"][hand]["object_id_list"]   = object_id_list
                
        """
        create indexer                    
        - this is where i discard sequences / subsequences
        - self.sequence_data remains unaffected
        """
        
        self.indexer = []
        for sequence_idx,sequence_data in enumerate(self.sequence_data):
        
            indexer           = []               
            filename          = sequence_data["metadata"]["filename"]                                                
            main_action       = sequence_data["metadata"]["main_action"]
            #motion_folder_num = sequence_data["metadata"]["motion_folder_num"]
                                            
            # check mocap variance
            skip = 0
            f = filename.split("/")
            f = "_".join([f[-3],f[-2],f[-1]])
            f = os.path.splitext(f)[0]          # Cut_files_motions_3021_Cut1_c_0_05cm_01
            sequence_mocap_markers = self.object_mocap_markers[f]
            for k,v in sequence_mocap_markers.items():
                if (v["var"] > self.mocap_var_limit).any() and k != "kitchen_sideboard":
                    logger.info("Skipping {} because it's mocap marker has variance greater than {}".format(filename, self.mocap_var_limit))
                    skip=1
                    break
                if skip==1:
                    continue
                    
            # max time given the input and output lengths
            max_time = np.max(np.concatenate([sequence_data["segmentation"][k]["time"] for k,v in sequence_data["segmentation"].items()],axis=0)) - (self.inp_length+self.out_length)*self.time_step_size
                                
            inp_frame = 0
            while inp_frame < max_time:            

                skip = 0
                
                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                # version to test "GrabNet"                                               #
                # - discard subsequences if at least 1 hand is not holding onto an object #
                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                
                if self.truncate_sequence == "object_in_both_hands":
                    frames = [inp_frame + self.resolution * i for i in range(self.inp_length)]
                    for hand in ["lhand","rhand"]:
                        # get objects being handled at every timestep
                        time_segmentations = sequence_data["segmentation"][hand]["time"]
                        handled_obj_ids = sequence_data["segmentation"][hand]["object_id"]  
                        handled_obj_ids = np.array([handled_obj_ids[np.where((frame >= time_segmentations[:,0]) & (frame < time_segmentations[:,1]))[0][0]] for frame in frames])
                        # skip if either hand is not holding onto an object
                        if any([x == -1 for x in handled_obj_ids]):
                            skip = 1
                            break
                
                if skip == 0:
                    indexer.append([sequence_idx, inp_frame, main_action, sequence_data["metadata"]["object_names"], filename])
                inp_frame += self.resolution
            
            # convert to dict
            indexer = [{"sequence_data_index":x[0], "inp_frame":x[1], "main_action":x[2], "object_names":x[3], "filename":x[4]} for x in indexer]
            self.indexer.extend(indexer)       
        self.data_len = len(self.indexer)
        t2 = time.time()

        logger.info("Processing time: {}".format(t2-t1))
        logger.info("Num {} samples: {}".format(self.dtype, self.data_len))
                
    def __len__(self):
        
        # 54 or 32
        return max(len(self.indexer),self.batch_size)
    
    def __getitem__(self, idx, is_distractor=0):

        t1 = time.time()

        #idx = idx % self.data_len
        # resample a random value if the sampled idx goes beyond data_len. This ensures that it does not matter how I augment the data
        if idx > self.data_len:
            idx = random.randint(0,self.__len__())
            
        # get the data
        indexer         = self.indexer[idx]
        inp_frame       = indexer["inp_frame"]
        main_action     = indexer["main_action"]
        sequence_data   = copy.deepcopy(self.sequence_data[indexer["sequence_data_index"]]) # 1.6689300537109375e-06 if equal operator, 0.003063678741455078 if deepcopy
                      
        # sequence filename
        filename = sequence_data["metadata"]["filename"]                # /home/haziq/datasets/kit_mocap/data-sorted-simpleGT-v4-xml-only/Cut/files_motions_3021/Cut1_c_0_05cm_01.xml
        filename = filename.split("/")
        filename = "_".join([filename[-3],filename[-2],filename[-1]])
        filename = os.path.splitext(filename)[0]                        # Cut_files_motions_3021_Cut1_c_0_05cm_01
        
        # # # # # # # # # # # # # #
        #                         #
        # get input and key frame #
        #                         #
        # # # # # # # # # # # # # #   
                
        # get frame data
        if is_distractor == 0:
            frames = np.array([inp_frame+i*self.time_step_size for i in range(self.inp_length+self.out_length)])
        else:
            frames = np.zeros(self.inp_length+self.out_length)
        frame_data = {"inp_frames":frames[:self.inp_length], "frames":frames}
        
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        #                                                             #
        # get object data                                             # 
        # - xyz, xyz_vel, pos, rot all do not contain the table data  #
        #                                                             #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #        

        # if self.normalize
        # - interactee also needs idle
        # - other objects
        #   - scale individually
        #   - scale using table min max

        obj_data = self.get_object_data(sequence_data, frame_data)
                
        # scale interactee
        # what about the other objects?
        
        # # # # # # # # # # # # # #
        #                         #
        # get human data          #  
        #                         #
        # # # # # # # # # # # # # #

        human_data = self.get_human_data(sequence_data, frame_data, obj_data)

        # # # # # # # # # # # # # #
        #                         #
        # get action data         #  
        #                         #
        # # # # # # # # # # # # # #
                
        # main action
        main_action_id = self.action_to_id[main_action]
        main_action_oh = one_hot(np.array([main_action_id]),len(self.main_actions))
        
        # fine action
        lhand_action_data = self.get_action_data(sequence_data, frame_data, obj_data, "lhand", "lhand")
        rhand_action_data = self.get_action_data(sequence_data, frame_data, obj_data, "rhand", "rhand")
        
        # merge
        hand_action_data = {}
        hand_action_data["action_ohs"]       = np.stack([lhand_action_data["lhand_action_ohs"],rhand_action_data["rhand_action_ohs"]],axis=0)
        hand_action_data["action_ids"]       = np.stack([lhand_action_data["lhand_action_ids"],rhand_action_data["rhand_action_ids"]],axis=0)
        hand_action_data["handled_obj_ids"]  = np.stack([lhand_action_data["lhand_handled_obj_ids"],rhand_action_data["rhand_handled_obj_ids"]],axis=0) # [2, inp_length + out_length]
        hand_action_data["handled_obj_idxs"] = np.stack([lhand_action_data["lhand_handled_obj_idxs"],rhand_action_data["rhand_handled_obj_idxs"]],axis=0)
        del lhand_action_data["lhand_action_ohs"]
        del lhand_action_data["lhand_action_ids"]
        del lhand_action_data["lhand_handled_obj_ids"]
        del lhand_action_data["lhand_handled_obj_idxs"]
        del rhand_action_data["rhand_action_ohs"]
        del rhand_action_data["rhand_action_ids"]
        del rhand_action_data["rhand_handled_obj_ids"]
        del rhand_action_data["rhand_handled_obj_idxs"]

        # # # # # # # #
        # distractors #
        # # # # # # # #
                
        # add distractor
        if is_distractor == 0 and self.add_distractors == 1:
                              
            # get the indices for sequence set with no similar objects
            actions_with_no_similar_objects = self.actions_with_no_similar_objects[main_action]
            idxs = [i for i,x in enumerate(self.indexer) if x["main_action"] in actions_with_no_similar_objects]
            
            # get the indices for sequences (not sequence set) with no similar objects
            if len(idxs) == 0:
            
                # get the object names for the current sequence
                # - note that indexer always contains "kitchen_sideboard"
                curr_seq_object_names = indexer["object_names"].copy()
                curr_seq_object_names.remove("kitchen_sideboard")
                
                # get the indices for sequences with no similar objects
                # - doing it this way is better compared to actions_with_no_similar_objects
                idxs = []
                for i,x in enumerate(self.indexer):
                    next_seq_object_names = x["object_names"].copy()
                    next_seq_object_names.remove("kitchen_sideboard")
                    if bool(set(curr_seq_object_names) & set(next_seq_object_names)) == 0:
                        #print(next_seq_object_names)
                        idxs.append(i)
            
            # filter list of distractors
            # - result of filter cannot be empty
            if hasattr(self,"distractor_filenames"):
                #print(filename)
                idxs_ = []
                for idx in idxs:
                    self_indexer = self.indexer[idx]
                    if any([distractor_filename in self_indexer["filename"] for distractor_filename in self.distractor_filenames]):
                        #print(x["filename"])
                        idxs_.append(idx)
                idxs = idxs_
                if len(idxs) == 0:
                    print(filename + " has empty filtered list of distractors!")
                    sys.exit()
                #input()
            
            # skip if still no valid distractor
            if len(idxs) == 0:
                pass
                
            # sample distractor and merge into obj_data
            else:
            
                # random sample from the list of valid sequences
                sequence_with_no_similar_objects_idx = random.choice(idxs)
            
                # get the distractors
                distractors = self.__getitem__(sequence_with_no_similar_objects_idx, is_distractor=1)
            
                # merge the obj_data from the actual sequence with the sampled distractors
                obj_data    = self.merge_distractors(indexer["main_action"], sequence_with_no_similar_objects_idx, obj_data, distractors, debug=0)
            
        # # # # # # # # # # #
        # form return data  #
        # # # # # # # # # # #
        
        # collate everything if not distractor
        if is_distractor == 0:
            return_data = {# filename
                           "sequence":filename, 
                           
                           # inp_frame for saving the json files, scaled by self.resolution to ensure the numbers are integers
                           # inp_frame increments in steps of resolution [0,resolution,resolution*2,...]
                           "inp_frame":inp_frame/self.resolution, "resolution":self.resolution,
                           "inp_length":self.inp_length,
                           
                           # main action
                           "main_action":main_action, "main_action_oh":main_action_oh,
                           
                           # scale
                           "xyz_scale":self.xyz_scale, "kin_scale":self.kin_scale,
            
                            # hand action data
                            **lhand_action_data, **rhand_action_data, **hand_action_data,
            
                           # frame data
                            **frame_data, 
                            
                           # object data
                            **obj_data, 
                           
                           # human data
                            **human_data}
            
            # so i know the sequence that was sampled
            #if self.add_distractors == 1:    
            #    return_data["distractor_sequence"] = self.sequence_data[self.indexer[sequence_with_no_similar_objects_idx]["sequence_data_index"]]["metadata"]["filename"]
            #    return_data["distractor_sequence_idx"] = sequence_with_no_similar_objects_idx
        
        # only return object data if distractor
        else:
            return_data = {**obj_data}
        
        # # # # # # # # # # # #
        # process return data #
        # - generate the mask #
        # # # # # # # # # # # #
        
        # do not process distractors
        if is_distractor == 0:
        
            # separate into inp, future, past
            return_data = self.split_data(return_data)
        
            # generate the masks
            return_data = self.generate_mask(return_data, p=self.p, prefix="")
                                
            # convert all list of strings into a single string
            for k,v in return_data.items():
                if type(v) == type([]) and type(v[0]) == type("string"):
                    return_data[k] = str(v)
                    
            # convert array type to float32
            for k,v in return_data.items():
                if type(v) == type(np.array(1.0)) and all([k != x for x in ["inp_xyz_mask_idxs","out_xyz_mask_idxs","kin_mask_idxs","link_idx_order_to_wrist","inp_handled_obj_idxs","out_handled_obj_idxs","out_finger_mask_idxs"]]):
                    return_data[k] = return_data[k].astype(np.float32)
        
        """
        # debug
        if is_distractor == 0:
            for k,v in return_data.items():
                if type(v) == type(np.array([1])):
                    print(k, v.shape, v.dtype)
            print()
            #sys.exit()
        """
        
        t2 = time.time()
        #print("Dataloader:", t2-t1)
        
        return return_data

    def merge_distractors(self, main_action, idx, real_obj_data_clone, distractor_obj_data_clone, debug=0):
                
        real_obj_data = dict(real_obj_data_clone)
        distractor_obj_data = dict(distractor_obj_data_clone)
        
        # merge_distractors
        # - (done) set distractor positions throughout time to its initial position
        # - (done) keep name of distractors (this is why the distractors cannot be objects relevant to the action)
        # - (done) unpad number objects before concatenating obj_ids, obj_ohs, obj_xyz, obj_xyz_vel, obj_pos, obj_rot
        # - randomize order
        # - (done) pad them, object_paded_length must satisfy the new total number of objects
                
        """
        set distractor positions throughout time to its initial position
        """
                
        distractor_obj_data["obj_xyz"]     = np.repeat(distractor_obj_data["obj_xyz"][0:1,:,:,:],repeats=self.inp_length+self.out_length,axis=0)        # [t, m, 4, 3]
        distractor_obj_data["obj_xyz_vel"] = np.repeat(distractor_obj_data["obj_xyz_vel"][0:1,:,:,:],repeats=self.inp_length+self.out_length,axis=0)    # [t, m, 4, 3]
        distractor_obj_data["obj_pos"]     = np.repeat(distractor_obj_data["obj_pos"][0:1,:,:],repeats=self.inp_length+self.out_length,axis=0)          # [t, m, 3]
        distractor_obj_data["obj_rot"]     = np.repeat(distractor_obj_data["obj_rot"][0:1,:,:],repeats=self.inp_length+self.out_length,axis=0)          # [t, m, 3]
                
        # limit number of extra distractors
        if self.num_extra_distractors != -1:
            """print("num_extra_distractors",self.num_extra_distractors)
            print(distractor_obj_data["obj_names"])
            print(ast.literal_eval(distractor_obj_data["obj_mocap_names"]))
            print(distractor_obj_data["obj_ids"], distractor_obj_data["obj_ids_unpadded_length"])
            print(distractor_obj_data["obj_ohs"], distractor_obj_data["obj_ohs_unpadded_length"])
            print(distractor_obj_data["obj_xyz"].shape, distractor_obj_data["obj_xyz_unpadded_objects"], np.sum(distractor_obj_data["obj_xyz"][:,:distractor_obj_data["obj_xyz_unpadded_objects"]]), np.sum(distractor_obj_data["obj_xyz"][:,distractor_obj_data["obj_xyz_unpadded_objects"]:]))
            print(distractor_obj_data["obj_xyz_vel"].shape, distractor_obj_data["obj_xyz_vel_unpadded_objects"])
            print(distractor_obj_data["obj_pos"].shape, distractor_obj_data["obj_pos_unpadded_objects"])
            print(distractor_obj_data["obj_rot"].shape, distractor_obj_data["obj_rot_unpadded_objects"])"""
                        
            distractor_obj_data["obj_names"] = distractor_obj_data["obj_names"][:self.num_extra_distractors]
            distractor_obj_data["obj_mocap_names"] = ast.literal_eval(distractor_obj_data["obj_mocap_names"])
            distractor_obj_data["obj_mocap_names"] = distractor_obj_data["obj_mocap_names"][:self.num_extra_distractors]
            distractor_obj_data["obj_mocap_names"] = str(distractor_obj_data["obj_mocap_names"])
            distractor_obj_data["obj_ids"][self.num_extra_distractors:] = 0
            distractor_obj_data["obj_ohs"][self.num_extra_distractors:] = 0
            distractor_obj_data["obj_xyz"][:,self.num_extra_distractors:] = 0
            distractor_obj_data["obj_xyz_vel"][:,self.num_extra_distractors:] = 0
            distractor_obj_data["obj_pos"][:,self.num_extra_distractors:] = 0
            distractor_obj_data["obj_rot"][:,self.num_extra_distractors:] = obj_ids
            
            # if the number of extra distractors is lesser than the number of objects currently in the distractor data
            if self.num_extra_distractors < distractor_obj_data["obj_ids_unpadded_length"]:
                distractor_obj_data["obj_ids_unpadded_length"] = self.num_extra_distractors
                distractor_obj_data["obj_ohs_unpadded_length"] = self.num_extra_distractors
                distractor_obj_data["obj_xyz_unpadded_objects"] = self.num_extra_distractors
                distractor_obj_data["obj_xyz_vel_unpadded_objects"] = self.num_extra_distractors
                distractor_obj_data["obj_pos_unpadded_objects"] = self.num_extra_distractors
                distractor_obj_data["obj_rot_unpadded_objects"] = self.num_extra_distractors
            
            """print("num_extra_distractors",self.num_extra_distractors)
            print(distractor_obj_data["obj_names"])
            print(ast.literal_eval(distractor_obj_data["obj_mocap_names"]))
            print(distractor_obj_data["obj_ids"], distractor_obj_data["obj_ids_unpadded_length"])
            print(distractor_obj_data["obj_ohs"], distractor_obj_data["obj_ohs_unpadded_length"])
            print(distractor_obj_data["obj_xyz"].shape, distractor_obj_data["obj_xyz_unpadded_objects"], np.sum(distractor_obj_data["obj_xyz"][:,:distractor_obj_data["obj_xyz_unpadded_objects"]]), np.sum(distractor_obj_data["obj_xyz"][:,distractor_obj_data["obj_xyz_unpadded_objects"]:]))
            print(distractor_obj_data["obj_xyz_vel"].shape, distractor_obj_data["obj_xyz_vel_unpadded_objects"])
            print(distractor_obj_data["obj_pos"].shape, distractor_obj_data["obj_pos_unpadded_objects"])
            print(distractor_obj_data["obj_rot"].shape, distractor_obj_data["obj_rot_unpadded_objects"])
            print()"""
                
        # merged_data dictionary
        merged_data = {}
        
        # # # # # # # # # # # # # # # # # # # # # # #
        # obj_names, obj_mocap_names, and obj_paths #
        # # # # # # # # # # # # # # # # # # # # # # #

        merged_data["obj_names"] = real_obj_data["obj_names"] + distractor_obj_data["obj_names"]
        merged_data["obj_paths"] = str(ast.literal_eval(real_obj_data["obj_paths"]) + ast.literal_eval(distractor_obj_data["obj_paths"]))
        merged_data["obj_mocap_names"] = str(ast.literal_eval(real_obj_data["obj_mocap_names"]) + ast.literal_eval(distractor_obj_data["obj_mocap_names"]))
        
        # # # # # #
        # obj_ids #
        # # # # # #
                
        # merge
        real_obj_ids        = real_obj_data["obj_ids"][:real_obj_data["obj_ids_unpadded_length"]]               # [n]
        distractor_obj_ids  = distractor_obj_data["obj_ids"][:distractor_obj_data["obj_ids_unpadded_length"]]   # [m]
        merged_obj_ids      = np.concatenate((real_obj_ids,distractor_obj_ids),axis=0)                          # [n+m]
        padded_merged_obj_ids = pad(merged_obj_ids,self.object_padded_length)                                   # [10]
        merged_obj_ids_unpadded_length = real_obj_data["obj_ids_unpadded_length"] + distractor_obj_data["obj_ids_unpadded_length"]
        
        # update dictionary
        merged_data["obj_ids"] = padded_merged_obj_ids
        merged_data["obj_ids_unpadded_length"] = merged_obj_ids_unpadded_length
        merged_data["obj_human_ids"] = np.concatenate([padded_merged_obj_ids,[0]])
        
        # to make sure the objects in the distractor sequence does not exist in the main sequence
        # does not apply when im adding twice or more number of distractors
        if len(set(merged_obj_ids)) != len(merged_obj_ids) and self.num_extra_distractors < 8:
            print(main_action, idx, self.indexer[idx]["main_action"])
            print("real_obj_ids:", [self.object_id_to_name[x] for x in real_obj_ids])
            print("distractor_obj_ids:", [self.object_id_to_name[x] for x in distractor_obj_ids])
            print("merged_obj_ids:", [self.object_id_to_name[x] for x in merged_obj_ids])
            sys.exit()
            
        #print("obj_ids")
        #print(real_obj_ids, distractor_obj_ids, merged_obj_ids, padded_merged_obj_ids)
        #print(merged_obj_ids_unpadded_length)
        #print()
        
        # # # # # #
        # obj_ohs #
        # # # # # #
        
        # merge
        real_obj_ohs        = real_obj_data["obj_ohs"][:real_obj_data["obj_ohs_unpadded_length"]]               # [n, self.num_obj_wrist_classes]
        distractor_obj_ohs  = distractor_obj_data["obj_ohs"][:distractor_obj_data["obj_ohs_unpadded_length"]]   # [m, self.num_obj_wrist_classes]
        merged_obj_ohs      = np.concatenate((real_obj_ohs,distractor_obj_ohs),axis=0)                          # [n+m, self.num_obj_wrist_classes]
        padded_merged_obj_ohs = pad(merged_obj_ohs,self.object_padded_length)     
        merged_obj_ohs_unpadded_length = real_obj_data["obj_ohs_unpadded_length"] + distractor_obj_data["obj_ohs_unpadded_length"]
        
        # update dictionary
        merged_data["obj_ohs"] = padded_merged_obj_ohs
        merged_data["obj_ohs_unpadded_length"] = merged_obj_ohs_unpadded_length
        
        #print("obj_ohs")
        #print(real_obj_ohs.shape, distractor_obj_ohs.shape, merged_obj_ohs.shape, padded_merged_obj_ohs.shape)
        #print(merged_obj_ohs_unpadded_length)
        #print()
        
        # # # # # #
        # obj_xyz #
        # # # # # #
        
        # merge
        real_obj_xyz            = real_obj_data["obj_xyz"][:,:real_obj_data["obj_xyz_unpadded_objects"]]
        distractor_obj_xyz      = distractor_obj_data["obj_xyz"][:,:distractor_obj_data["obj_xyz_unpadded_objects"]]
        merged_obj_xyz          = np.concatenate((real_obj_xyz,distractor_obj_xyz),axis=1)
        padded_merged_obj_xyz   = np.transpose(pad(np.transpose(merged_obj_xyz,[1,0,2,3]),self.object_padded_length),[1,0,2,3])
        merged_obj_xyz_unpadded_objects = real_obj_data["obj_xyz_unpadded_objects"] + distractor_obj_data["obj_xyz_unpadded_objects"]
        
        # update dictionary
        merged_data["obj_xyz"] = padded_merged_obj_xyz
        #merged_data["obj_xyz_unpadded_length"]  = real_obj_data["obj_xyz_unpadded_length"]
        merged_data["obj_xyz_unpadded_objects"] = merged_obj_xyz_unpadded_objects
        
        """print("obj_xyz")
        print(real_obj_xyz.shape, distractor_obj_xyz.shape, merged_obj_xyz.shape)
        print(padded_merged_obj_xyz.shape)
        print(merged_obj_xyz_unpadded_objects)
        print()"""
        
        # # # # # # # # # # #
        # adjacency matrix  #
        # # # # # # # # # # #
        
        if hasattr(self,"adjacency_matrix_type"):
            if self.adjacency_matrix_type == "mtgcn":
            
                # object adjacency matrix
                obj_adjacency_matrix = 1 - np.eye(self.object_padded_length)
                for i,obj_id in enumerate(merged_obj_ids):
                    # if obj_id is zero, detach it from all
                    if obj_id == 0:
                        obj_adjacency_matrix = detach(obj_adjacency_matrix, i)
                
                # object and human adjacency matrix
                adjacency_matrix = np.zeros([self.object_padded_length + 53, self.object_padded_length + 53])
                
                # object adjacency matrix padded
                # - we pad AFTER the obj_adjacency matrix
                obj_adjacency_matrix = np.pad(obj_adjacency_matrix,((0,53),(0,53)))
                                
                # human adjacency matrix padded
                # - we pad BEFORE the mocap_adjacency_matrix
                human_adjacency_matrix = np.pad(var.mocap_adjacency_matrix,((self.object_padded_length,0),(self.object_padded_length,0)))
                
                # object and human adjacency matrix
                # - data is in the order [object,human] because of how we padded it above
                adjacency_matrix = obj_adjacency_matrix + human_adjacency_matrix
                merged_data["adjacency_matrix"] = adjacency_matrix
                
                #np.set_printoptions(threshold=sys.maxsize)
                #print(var.mocap_adjacency_matrix)
                #sys.exit()
                #print(merged_obj_ids)
                #print(adjacency_matrix)
                
                #  0  1  2  3  4  5  6  7
                # [0. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
                #  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 
                #  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
                #sys.exit()
        
        else:
        
            merged_data["adjacency_matrix"] = np.zeros([self.object_padded_length + 53, self.object_padded_length + 53])
        
        # # # # # # # # # #
        # initial_obj_xyz #
        # # # # # # # # # #
        
        # merge
        real_initial_obj_xyz            = real_obj_data["initial_obj_xyz"][:,:real_obj_data["initial_obj_xyz_unpadded_objects"]]                                # [1, n1,                   4, 3]
        distractor_initial_obj_xyz      = distractor_obj_data["initial_obj_xyz"][:,:distractor_obj_data["initial_obj_xyz_unpadded_objects"]]                    # [1, n2,                   4, 3]
        merged_initial_obj_xyz          = np.concatenate((real_initial_obj_xyz, distractor_initial_obj_xyz),axis=1)                                             # [1, n1+n2,                4, 3]
        padded_merged_initial_obj_xyz   = np.transpose(pad(np.transpose(merged_initial_obj_xyz,[1,0,2,3]),self.object_padded_length),[1,0,2,3])                 # [1, object_padded_length, 4, 3]
        merged_initial_obj_xyz_unpadded_objects = real_obj_data["initial_obj_xyz_unpadded_objects"] + distractor_obj_data["initial_obj_xyz_unpadded_objects"]   # n1+n2
        
        # update dictionary
        merged_data["initial_obj_xyz"] = padded_merged_initial_obj_xyz
        merged_data["initial_obj_xyz_unpadded_objects"] = merged_initial_obj_xyz_unpadded_objects
        
        # # # # # # # # # # # #
        # initial_obj_xyz_vel #
        # # # # # # # # # # # #
        
        real_initial_obj_xyz_vel                    = real_obj_data["initial_obj_xyz_vel"][:,:real_obj_data["initial_obj_xyz_vel_unpadded_objects"]]                                # [1, n1,                   4, 3]
        distractor_initial_obj_xyz_vel              = distractor_obj_data["initial_obj_xyz_vel"][:,:distractor_obj_data["initial_obj_xyz_vel_unpadded_objects"]]                    # [1, n2,                   4, 3]
        merged_initial_obj_xyz_vel                  = np.concatenate((real_initial_obj_xyz_vel, distractor_initial_obj_xyz_vel),axis=1)                                             # [1, n1+n2,                4, 3]
        padded_merged_initial_obj_xyz_vel           = np.transpose(pad(np.transpose(merged_initial_obj_xyz_vel,[1,0,2,3]),self.object_padded_length),[1,0,2,3])                 # [1, object_padded_length, 4, 3]
        merged_initial_obj_xyz_vel_unpadded_objects = real_obj_data["initial_obj_xyz_vel_unpadded_objects"] + distractor_obj_data["initial_obj_xyz_vel_unpadded_objects"]   # n1+n2
        
        # update dictionary
        merged_data["initial_obj_xyz_vel"] = padded_merged_initial_obj_xyz_vel
        merged_data["initial_obj_xyz_vel_unpadded_objects"] = merged_initial_obj_xyz_vel_unpadded_objects
        
        # # # # # # # #
        # obj_xyz_vel #
        # # # # # # # #
        
        # merge
        real_obj_xyz_vel        = real_obj_data["obj_xyz_vel"][:,:real_obj_data["obj_xyz_vel_unpadded_objects"]]
        distractor_obj_xyz_vel  = distractor_obj_data["obj_xyz_vel"][:,:distractor_obj_data["obj_xyz_vel_unpadded_objects"]]
        merged_obj_xyz_vel      = np.concatenate((real_obj_xyz_vel,distractor_obj_xyz_vel),axis=1)
        padded_merged_obj_xyz_vel = np.transpose(pad(np.transpose(merged_obj_xyz_vel,[1,0,2,3]),self.object_padded_length),[1,0,2,3])
        merged_obj_xyz_vel_unpadded_objects = real_obj_data["obj_xyz_vel_unpadded_objects"] + distractor_obj_data["obj_xyz_vel_unpadded_objects"]
        
        # update dictionary
        merged_data["obj_xyz_vel"] = padded_merged_obj_xyz_vel
        #merged_data["obj_xyz_vel_unpadded_length"]  = real_obj_data["obj_xyz_vel_unpadded_length"]
        merged_data["obj_xyz_vel_unpadded_objects"] = merged_obj_xyz_vel_unpadded_objects
        
        #print("obj_xyz_vel")
        #print(distractor_obj_xyz_vel.shape, real_obj_xyz_vel.shape, merged_obj_xyz_vel.shape)
        #print(merged_obj_xyz_vel_unpadded_objects)
        #print()
        
        # # # # # #
        # obj_pos #
        # # # # # #
        
        # merge
        real_obj_pos        = real_obj_data["obj_pos"][:,:real_obj_data["obj_pos_unpadded_objects"]]
        distractor_obj_pos  = distractor_obj_data["obj_pos"][:,:distractor_obj_data["obj_pos_unpadded_objects"]]
        merged_obj_pos      = np.concatenate((real_obj_pos,distractor_obj_pos),axis=1)
        padded_merged_obj_pos = np.transpose(pad(np.transpose(merged_obj_pos,[1,0,2]),self.object_padded_length),[1,0,2])
        merged_obj_pos_unpadded_objects = real_obj_data["obj_pos_unpadded_objects"] + distractor_obj_data["obj_pos_unpadded_objects"]
        
        # update dictionary
        merged_data["obj_pos"] = padded_merged_obj_pos
        #merged_data["obj_pos_unpadded_length"]  = real_obj_data["obj_pos_unpadded_length"]
        merged_data["obj_pos_unpadded_objects"] = merged_obj_pos_unpadded_objects
        
        #print("obj_pos")
        #print(distractor_obj_pos.shape, real_obj_pos.shape, merged_obj_pos.shape)
        #print(merged_obj_pos_unpadded_objects)
        #print()
        
        # # # # # # # # # #
        # initial_obj_pos #
        # # # # # # # # # #
        
        # merge
        real_initial_obj_pos        = real_obj_data["initial_obj_pos"][:,:real_obj_data["initial_obj_pos_unpadded_objects"]]
        distractor_initial_obj_pos  = distractor_obj_data["initial_obj_pos"][:,:distractor_obj_data["initial_obj_pos_unpadded_objects"]]
        merged_initial_obj_pos      = np.concatenate((real_initial_obj_pos,distractor_initial_obj_pos),axis=1)
        padded_merged_initial_obj_pos = np.transpose(pad(np.transpose(merged_initial_obj_pos,[1,0,2]),self.object_padded_length),[1,0,2])
        merged_initial_obj_pos_unpadded_objects = real_obj_data["initial_obj_pos_unpadded_objects"] + distractor_obj_data["initial_obj_pos_unpadded_objects"]
        
        # update dictionary
        merged_data["initial_obj_pos"] = padded_merged_initial_obj_pos
        #merged_data["obj_pos_unpadded_length"]  = real_obj_data["obj_pos_unpadded_length"]
        merged_data["initial_obj_pos_unpadded_objects"] = merged_initial_obj_pos_unpadded_objects
        
        # # # # # #
        # obj_rot #
        # # # # # #
        
        # merge
        real_obj_rot        = real_obj_data["obj_rot"][:,:real_obj_data["obj_rot_unpadded_objects"]]
        distractor_obj_rot  = distractor_obj_data["obj_rot"][:,:distractor_obj_data["obj_rot_unpadded_objects"]]
        merged_obj_rot      = np.concatenate((real_obj_rot,distractor_obj_rot),axis=1)
        padded_merged_obj_rot = np.transpose(pad(np.transpose(merged_obj_rot,[1,0,2]),self.object_padded_length),[1,0,2])
        merged_obj_rot_unpadded_objects = real_obj_data["obj_rot_unpadded_objects"] + distractor_obj_data["obj_rot_unpadded_objects"]
        
        # update dictionary
        merged_data["obj_rot"] = padded_merged_obj_rot
        #merged_data["obj_rot_unpadded_length"]  = real_obj_data["obj_rot_unpadded_length"]
        merged_data["obj_rot_unpadded_objects"] = merged_obj_rot_unpadded_objects
        
        #print("obj_rot")
        #print(distractor_obj_rot.shape, real_obj_rot.shape, merged_obj_rot.shape)
        #print(merged_obj_rot.shape)
        
        # # # # # # # # # #
        # initial_obj_rot #
        # # # # # # # # # #
        
        # merge
        real_initial_obj_rot        = real_obj_data["initial_obj_rot"][:,:real_obj_data["initial_obj_rot_unpadded_objects"]]
        distractor_initial_obj_rot  = distractor_obj_data["initial_obj_rot"][:,:distractor_obj_data["initial_obj_rot_unpadded_objects"]]
        merged_initial_obj_rot      = np.concatenate((real_initial_obj_rot,distractor_initial_obj_rot),axis=1)
        padded_merged_initial_obj_rot = np.transpose(pad(np.transpose(merged_initial_obj_rot,[1,0,2]),self.object_padded_length),[1,0,2])
        merged_initial_obj_rot_unpadded_objects = real_obj_data["initial_obj_rot_unpadded_objects"] + distractor_obj_data["initial_obj_rot_unpadded_objects"]
        
        # update dictionary
        merged_data["initial_obj_rot"] = padded_merged_initial_obj_rot
        #merged_data["obj_pos_unpadded_length"]  = real_obj_data["obj_pos_unpadded_length"]
        merged_data["initial_obj_rot_unpadded_objects"] = merged_initial_obj_rot_unpadded_objects
        
        # # # # # # # # # # # # # # # #
        # table data (no distractors  #
        # # # # # # # # # # # # # # # #
        
        merged_data["obj_table_pos"]                    = real_obj_data["obj_table_pos"]
        #merged_data["obj_table_pos_unpadded_length"]    = real_obj_data["obj_table_pos_unpadded_length"]
        merged_data["obj_table_rot"]                    = real_obj_data["obj_table_rot"]
        #merged_data["obj_table_rot_unpadded_length"]    = real_obj_data["obj_table_rot_unpadded_length"]
        merged_data["obj_table_xyz"]                    = real_obj_data["obj_table_xyz"]
        
        
        merged_data["initial_obj_table_xyz"] = real_obj_data["initial_obj_table_xyz"]
        merged_data["initial_obj_table_pos"] = real_obj_data["initial_obj_table_pos"]
        merged_data["initial_obj_table_rot"] = real_obj_data["initial_obj_table_rot"]
        
        # # # # # #
        # lqk rqk #
        # # # # # #
        if "lqk" in real_obj_data.keys() and "rqk" in real_obj_data.keys():
            merged_data["lqk"]  = real_obj_data["lqk"]
            merged_data["rqk"]  = real_obj_data["rqk"]
        
        if debug == 1:
            print(merged_data["obj_xyz_unpadded_objects"])
        
        return merged_data
           
    def flip_data(self, data):
        
        # flip masked
        # input to model will be the normal & flipped masked human and objects
        # output will be the missing joints
        # they will then be merged to the masked data
        # and prediction will occur
        
        masked_kin      = data["masked_kin"]        # [t, 44]
        masked_finger   = data["masked_finger"]     # [t, 2, 19]
        masked_obj_xyz  = data["masked_obj_xyz"]    # [t, n, markers, 3]
        
        reversed_masked_kin     = np.copy(masked_kin)
        reversed_masked_finger  = np.copy(masked_finger)
        reversed_masked_obj_xyz = np.copy(masked_obj_xyz)
        
        reversed_masked_kin     = reversed_masked_kin[::-1]
        reversed_masked_finger  = reversed_masked_finger[::-1]
        reversed_masked_obj_xyz = reversed_masked_obj_xyz[::-1]
        
        data["reversed_masked_kin"]     = reversed_masked_kin
        data["reversed_masked_finger"]  = reversed_masked_finger
        data["reversed_masked_obj_xyz"] = reversed_masked_obj_xyz
        return data
   
    def split_data(self, data):
    
        # split the relevant keys into inp, future, past
        keys = ["kin","finger","obj_xyz", "frames","xyz", "obj_pos", "obj_rot"]
        for k in keys:
            data["inp_"+k] = data[k][:self.inp_length]
            data["out_"+k] = data[k][self.inp_length:self.inp_length+self.out_length]
        keys = ["obj_xyz_unpadded_objects"]
        for k in keys:
            data["inp_"+k] = data[k]
            data["out_"+k] = data[k]
                                
        # need to split separately since the 1st dimension is for the left and right hands
        data["inp_handled_obj_ids"] = data["handled_obj_ids"][:,:self.inp_length]                                  # [2, inp_length]
        data["out_handled_obj_ids"] = data["handled_obj_ids"][:,self.inp_length:self.inp_length+self.out_length]   # [2, out_length]
        data["inp_handled_obj_idxs"] = data["handled_obj_idxs"][:,:self.inp_length]                                # [2, inp_length]
        data["out_handled_obj_idxs"] = data["handled_obj_idxs"][:,self.inp_length:self.inp_length+self.out_length] # [2, out_length]
        return data
   
    def generate_mask(self, data, p=None, prefix=""):
                                           
        # during the inp frame
        # (done) probability of masking either the left or right
        # mask object if hand was in contact with object
        # check if using exp or xyz
                
        # mask out l arm if p < 0.5
        # mask out r arm if p > 0.5
        p = np.random.uniform(low=0.0,high=1.0,size=1)[0] if p == None else p
        inp_masked_hand = "Left" if p < 0.5 else "Right"
        inp_masked_hand_idx = 0 if p < 0.5 else 1
                
        """
        generate inp and out xyz mask
        - out needed for teacher forcing
        """
        
        # inp xyz_mask_idxs
        inp_xyz_mask_idxs = self.l_arm_mocap_idxs if p < 0.5 else self.r_arm_mocap_idxs
        inp_xyz_mask_idxs = np.array(inp_xyz_mask_idxs)
        
        # inp_masked_xyz - mocap markers but where the ones that are masked are set to 0
        inp_masked_xyz = np.copy(data["inp_xyz"])   # [inp_length, 53, 3]
        inp_masked_xyz[:,inp_xyz_mask_idxs] = 0
        
        # out_xyz_mask_idxs
        out_xyz_mask_idxs = np.stack([self.l_arm_mocap_idxs,self.r_arm_mocap_idxs],axis=0)
        
        # out_masked_xyz
        out_masked_xyz = np.copy(data["out_xyz"])                       #    [inp_length, 53, 3]
        out_masked_xyz = np.repeat(out_masked_xyz[None,:,:,:],2,axis=0) # [2, inp_length, 53, 3]
        for i,x in enumerate(out_xyz_mask_idxs):
            out_masked_xyz[i,:,x] = 0
                
        # missing_xyz - mocap markers that were masked
        inp_missing_xyz = np.copy(data["inp_xyz"][:,inp_xyz_mask_idxs])
        
        # remaining_xyz - remaining mocap markers that were not masked
        inp_remaining_xyz = np.copy(data["inp_xyz"])[:,[i for i in range(data["inp_xyz"].shape[1]) if i not in inp_xyz_mask_idxs]] #[inp_length, 45, 3]
                     
        """
        generate kin mask
        """
        
        # # # # # # # # # # # # # # #
        # get FK for masked joints  #
        # # # # # # # # # # # # # # #
        """
        link_idx_order_to_wrist = self.link_idx_order_to_left_wrist if p < 0.5 else self.link_idx_order_to_right_wrist
        link_idx_order_to_wrist = np.array(link_idx_order_to_wrist) # [9, 2]
        """
        # # # # # # # # # # # # #
        # mask kinematic joints # 
        # # # # # # # # # # # # #
        """
        kin_mask_idxs = self.l_arm_joint_idxs if p < 0.5 else self.r_arm_joint_idxs   
        kin_mask_idxs = np.array(kin_mask_idxs)
        
        # masked_kin - kin joints but where the ones that are masked are set to 0
        masked_kin = np.copy(data["inp_kin"])           # [inp_length, 44]
        masked_kin[:,kin_mask_idxs] = 0     
        
        # missing_kin - kin joints that were masked 
        missing_kin = np.copy(data["inp_kin"])[:,kin_mask_idxs]  # [inp_length, 7]
                
        # remaining_kin - remaining kin joints that were not masked
        remaining_kin = np.copy(data["inp_kin"])[:,[i for i in range(data["inp_kin"].shape[1]) if i not in kin_mask_idxs]]
        """
        # # # # # # # # # # # #
        # mask finger joints  #
        # # # # # # # # # # # #
        
        # mask input finger joints
        inp_finger_mask_idxs = 0 if p < 0.5 else 1
        inp_masked_finger = np.copy(data["inp_finger"])                             # [inp_length, 2, 19]
        inp_masked_finger[:,inp_finger_mask_idxs] = 0                               
        inp_missing_finger = np.copy(data["inp_finger"])[:,inp_finger_mask_idxs]    # [inp_length, 19]
        
        # mask output finger joints
        out_masked_finger = np.copy(data["out_finger"])                         #    [out_length, 2, 19]
        out_masked_finger = np.repeat(out_masked_finger[None,:,:,:],2,axis=0)   # [2, out_length, 2, 19]
        for i in range(2):
            out_masked_finger[i,:,i] = 0
        out_finger_mask_idxs = np.array([0,1])
                                
        """
        generate object mask
        # determine if handled_obj_ids = [0 0 0 ...], [0 1 1 ...], [1 0 0 ...], [1 1 1 ...]
        # [0 0 0 0 0] means the person approaches the object.                   No change so we set the object's position to t[0]
        # [0 1 1 ...] means the person approaches and picks up object.          But the model should not have seen this and so we set the object's position to t[0]
        # [1 0 0 ...] means the person puts the object back onto the table.     But the model should not have seen this and so we set the object's position to t[where the first 0 is]
        # [1 1 1 1 1] means the person is performing an action with the object. But the model should not have seen this and so we set the object's position to t[initial]
        # [0 1 0 ...] is invalid
        # [1 0 1 ...] is invalid
        """
        
        # initialize masked_obj_xyz
        masked_obj_xyz  = np.copy(data["inp_obj_xyz"])      # [inp_length, n, num_mocap_markers, 3]
        initial_obj_xyz = np.copy(data["initial_obj_xyz"])  # [1,          n, num_mocap_markers, 3]
        
        #logger.info(masked_obj_xyz.shape, initial_obj_xyz.shape)
        #logger.info(np.sum(masked_obj_xyz[0,1] - masked_obj_xyz[5,1]))
        #masked_obj_xyz[:self.inp_length,1] = initial_obj_xyz[:,0]
        #logger.info(np.sum(masked_obj_xyz[0,1] - masked_obj_xyz[5,1]))
        #logger.info(masked_obj_xyz.shape, initial_obj_xyz.shape)
        #logger.info("NICE")
        #sys.exit()
        
        # list of obj_ids
        # - note that obj_ids and obj_xyz, etc are ordered together
        obj_ids = data["obj_ids"]
        #print("obj_ids", obj_ids) #logger.info("obj_ids",obj_ids)
        
        # get the handled_obj_ids and handled_obj_idxs at every timestep
        handled_obj_ids  = data["inp_handled_obj_ids"][0] if p < 0.5 else data["inp_handled_obj_ids"][1]
        handled_obj_idxs = data["inp_handled_obj_idxs"][0] if p < 0.5 else data["inp_handled_obj_idxs"][1]
                
        """
        handled_obj_idxs = []
        for x in handled_obj_ids:
            handled_obj_idx = np.where(obj_ids == x)[0]
            handled_obj_idx = -1 if len(handled_obj_idx) == 0 else handled_obj_idx[0]
            handled_obj_idxs.append(handled_obj_idx)
        handled_obj_idxs = np.array(handled_obj_idxs)    
        #print("handled_obj_ids", handled_obj_ids) #logger.info("handled_obj_ids",handled_obj_ids)
        #print("handled_obj_idxs", handled_obj_idxs) #logger.info("handled_obj_idxs",handled_obj_idxs)
        """
        
        # get the grasp probability at every timestep
        grasp_probability = np.copy(handled_obj_ids)
        grasp_probability[grasp_probability != -1] = 1
        grasp_probability[grasp_probability == -1] = 0
        
        # assert for invalid cases
        # - person picks up object then places it down in a sequence
        # WE NO LONGER HAVE SUCH CASES
        """x = handled_obj_ids[:-1] != handled_obj_ids[1:]
        #logger.info("x",x)
        if np.sum(x) > 1:
            print("Invalid sequence in process_data() {}".format(handled_obj_ids))
            print("Sequence name: {}".format(data["sequence"]))
            print("Masked hand: {}".format(inp_masked_hand))
            print("inp_frames: {}".format(data["inp_frames"]))
            sys.exit()"""

        # assert for invalid cases
        # - person's one hand handles more than 1 object
        x = set(handled_obj_ids)
        x.discard(-1)
        if len(x) > 1:
            print("Invalid sequence in process_data() {}".format(handled_obj_ids))
            print("Sequence name: {}".format(data["sequence"]))
            print("Masked hand: {}".format(inp_masked_hand))
            print("inp_frames: {}".format(data["inp_frames"]))
            sys.exit()
            
        # determine case
        # if [-1 -1 -1 -1 -1], not handling anything
        case = None
        if np.sum(handled_obj_ids == -1) == self.inp_length:
            case = 1
            #logger.info("pass")
            handled_obj_id  = -1
            handled_obj_idx = -1
            missing_obj_xyz = np.zeros(data["inp_obj_xyz"][:,0].shape)
            pass
            
        # if [1 1 1 1 1], person currently handling object
        # - set handled object to t[initial]
        elif np.sum(handled_obj_ids > -1) == self.inp_length:
            case = 2
            #logger.info("if [1 1 1 1 1], set handled object to t[initial]")
            #logger.info(handled_obj_ids)
            handled_obj_id = [x for x in list(set(handled_obj_ids)) if x != -1] # ignore -1 (hand not handling anything)
            if len(handled_obj_id) > 1:
                print("More than 1 handled_obj_ids detected in process_data() 1")
                print("sequence: {}".format(data["sequence"]))
                print("frames: {}".format(data["frames"]))
                print("object_ids: {}".format(handled_obj_id))
                print("object_names: {}".format([self.object_id_to_name[x] for x in handled_obj_id]))
                sys.exit()
            handled_obj_id  = handled_obj_id[0]
            handled_obj_idx = np.where(obj_ids == handled_obj_id)[0][0]
            #logger.info("handled_obj_idx",handled_obj_idx)
            masked_obj_xyz[:,handled_obj_idx] = np.copy(initial_obj_xyz[:,handled_obj_idx]) # [inp_length, obj_padded_length, 4, 3]
            missing_obj_xyz = np.copy(data["inp_obj_xyz"][:,handled_obj_idx])               # [inp_length,                    4, 3]
            #logger.info(masked_obj_xyz.shape, missing_obj_xyz.shape)
        
        # if [-1 1 1 ...], person picked object up before performing some action
        # - set the handled object throughout inp_length to t[0]
        elif np.sum(handled_obj_ids > -1) > 0 and np.sum(handled_obj_ids > -1) < self.inp_length and handled_obj_ids[0] == -1 and handled_obj_ids[-1] != -1:
            case = 3
            #logger.info("if [-1 1 1 ...], set the handled object throughout inp_length to t[0]")
            #logger.info(handled_obj_ids)
            handled_obj_id = [x for x in list(set(handled_obj_ids)) if x != -1] # ignore -1
            if len(handled_obj_id) > 1:
                print("More than 1 handled_obj_ids detected in process_data() 2 {}".format(handled_obj_id))
                sys.exit()
            handled_obj_id  = handled_obj_id[0]
            handled_obj_idx = np.where(obj_ids == handled_obj_id)[0][0]
            #logger.info("handled_obj_idx",handled_obj_idx)
            masked_obj_xyz[:,handled_obj_idx] = np.copy(masked_obj_xyz[0:1,handled_obj_idx]) # [inp_length, obj_padded_length, 4, 3]
            missing_obj_xyz = np.copy(data["inp_obj_xyz"][:,handled_obj_idx])                # [inp_length,                    4, 3]
            #logger.info(masked_obj_xyz.shape, missing_obj_xyz.shape)
        
        # if [1 -1 -1 ...], person placed object down after performing some action
        # - set the handled object to where the first -1 is
        elif np.sum(handled_obj_ids > -1) > 0 and np.sum(handled_obj_ids > -1) < self.inp_length and handled_obj_ids[0] != -1 and handled_obj_ids[-1] == -1:
            case = 4
            #logger.info("if [1 -1 -1 ...], set the handled object to where the first -1 is")
            #logger.info(handled_obj_ids)
            handled_obj_id = [x for x in list(set(handled_obj_ids)) if x != -1] # ignore -1
            if len(handled_obj_id) > 1:
                print("More than 1 handled_obj_ids detected in process_data() 3 {}".format(handled_obj_id))
                sys.exit()
            handled_obj_id  = handled_obj_id[0]
            handled_obj_idx = np.where(obj_ids == handled_obj_id)[0][0]
            #logger.info("handled_obj_idx",handled_obj_idx)
            t = np.where(x)[0][0]+1
            #logger.info("t",t)
            masked_obj_xyz[:,handled_obj_idx] = np.copy(masked_obj_xyz[t:t+1,handled_obj_idx])
            missing_obj_xyz = np.copy(data["inp_obj_xyz"][:,handled_obj_idx])
            #logger.info(masked_obj_xyz.shape, missing_obj_xyz.shape)
        
        # if [-1 1 1 ... -1], person picked object up, performed action, then put it down
        # - same as case 3: set the handled object throughout inp_length to t[0]
        elif np.sum(handled_obj_ids > -1) > 0 and np.sum(handled_obj_ids > -1) < self.inp_length and handled_obj_ids[0] == -1 and handled_obj_ids[-1] == -1:
            case = 5
            #logger.info("if [-1 1 1 ...], set the handled object throughout inp_length to t[0]")
            #logger.info(handled_obj_ids)
            handled_obj_id = [x for x in list(set(handled_obj_ids)) if x != -1] # ignore -1
            if len(handled_obj_id) > 1:
                print("More than 1 handled_obj_ids detected in process_data() 2 {}".format(handled_obj_id))
                sys.exit()
            handled_obj_id  = handled_obj_id[0]
            handled_obj_idx = np.where(obj_ids == handled_obj_id)[0][0]
            #logger.info("handled_obj_idx",handled_obj_idx)
            masked_obj_xyz[:,handled_obj_idx] = np.copy(masked_obj_xyz[0:1,handled_obj_idx]) # [inp_length, obj_padded_length, 4, 3]
            missing_obj_xyz = np.copy(data["inp_obj_xyz"][:,handled_obj_idx])                # [inp_length,                    4, 3]
            #logger.info(masked_obj_xyz.shape, missing_obj_xyz.shape)
        
        # if [1 -1 -1 ...], person placed object down after performing some action then picked something up
        # - same as case 4: set the handled object to where the first -1 is
        elif np.sum(handled_obj_ids > -1) > 0 and np.sum(handled_obj_ids > -1) < self.inp_length and handled_obj_ids[0] != -1 and handled_obj_ids[-1] != -1:
            case = 6
            #logger.info("if [1 -1 -1 ...], set the handled object to where the first -1 is")
            #logger.info(handled_obj_ids)
            handled_obj_id = [x for x in list(set(handled_obj_ids)) if x != -1] # ignore -1
            if len(handled_obj_id) > 1:
                print("More than 1 handled_obj_ids detected in process_data() 3 {}".format(handled_obj_id))
                sys.exit()
            handled_obj_id  = handled_obj_id[0]
            handled_obj_idx = np.where(obj_ids == handled_obj_id)[0][0]
            #logger.info("handled_obj_idx",handled_obj_idx)
            t = np.where(x)[0][0]+1
            #logger.info("t",t)
            masked_obj_xyz[:,handled_obj_idx] = np.copy(masked_obj_xyz[t:t+1,handled_obj_idx])
            missing_obj_xyz = np.copy(data["inp_obj_xyz"][:,handled_obj_idx])
            #logger.info(masked_obj_xyz.shape, missing_obj_xyz.shape)
        
        else:
            print(handled_obj_ids)
            print("Unknown case")
            sys.exit()
        #input()
        
        """
        data["kin"]           - full kin joints                                               [44,3]
        data["kin_mask_idxs"] - indexes of the masked joints in data["masked_kin"]
        data["masked_kin"]    - kin joints but where the ones that are masked are set to 0    [44,3]
        data["missing_kin"]   - kin joints that were masked                                   [? ,3]

        data["finger"]
        data["finger_masked_idxs"]
        data["masked_finger"]
        data["missing_finger"]
        
        data["obj_mask_id"]       - id of the object that was masked
        data["obj_mask_idx"]      - index of the masked object in data["obj_xyz"]
        data["masked_obj_xyz"]    - obj xyz but where the one that was masked is set to 0
        data["missing_obj_xyz"]   - obj xyz that were masked
        """
        
        # correct
        """if case == 2:
            print("masked_obj_xyz")
            print(masked_obj_xyz[:,handled_obj_idx])
            print("inp_obj_xyz")
            print(data["inp_obj_xyz"][:,handled_obj_idx])
            print(handled_obj_idx)
            print(case)
            sys.exit()"""
        
        # correct
        """if case == 3:
            print("masked_obj_xyz")
            print(masked_obj_xyz[:,handled_obj_idx])
            print("inp_obj_xyz")
            print(data["inp_obj_xyz"][:,handled_obj_idx])
            print(handled_obj_idx)
            print(case)
            sys.exit()"""
        
        # correct
        """if case == 4:
            print("masked_obj_xyz")
            print(masked_obj_xyz[:,handled_obj_idx])
            print("inp_obj_xyz")
            print(data["inp_obj_xyz"][:,handled_obj_idx])
            print(handled_obj_idx)
            print(case)
            sys.exit()"""
                     
        return_data = {"inp_masked_hand":inp_masked_hand, "inp_masked_hand_idx":inp_masked_hand_idx,
        
                       # handled_obj_ids and grasp_probability
                       "inp_handled_obj_ids":handled_obj_ids, "inp_handled_obj_idxs":handled_obj_idxs, "grasp_probability":grasp_probability,
        
                       # mocap markers
                       "inp_xyz_mask_idxs":inp_xyz_mask_idxs, "inp_masked_xyz":inp_masked_xyz, "inp_missing_xyz":inp_missing_xyz, "inp_remaining_xyz":inp_remaining_xyz,
                       "out_xyz_mask_idxs":out_xyz_mask_idxs, "out_masked_xyz":out_masked_xyz,
        
                       # kinematic joints
                       #"kin_mask_idxs":kin_mask_idxs, "inp_masked_kin":masked_kin, "inp_missing_kin":missing_kin, "inp_remaining_kin":remaining_kin,
        
                       # FK order for the masked arm
                       #"link_idx_order_to_wrist":link_idx_order_to_wrist,
        
                       # finger joints
                       "inp_finger_mask_idxs":inp_finger_mask_idxs, "inp_masked_finger":inp_masked_finger, "inp_missing_finger":inp_missing_finger,
                       "out_finger_mask_idxs":out_finger_mask_idxs, "out_masked_finger":out_masked_finger,
                       
                       # object
                       "obj_mask_id":handled_obj_id, "obj_mask_idx":handled_obj_idx, "inp_masked_obj_xyz":masked_obj_xyz, "inp_missing_obj_xyz":missing_obj_xyz}        
        
        #for k,v in return_data.items():
        #    return_data[prefix+k] = return_data[k]
        #    del return_data[k]
        
        return {**data, **return_data}

    def get_frame_data(self, inp_frame):
        
        inp_out_frames  = [inp_frame+i*self.time_step_size for i in range(self.inp_length+self.out_length)]
        inp_frames      = inp_out_frames[:self.inp_length]
        out_frames      = inp_out_frames[self.inp_length:]
                       
        return_data = {"inp_frames":inp_frames, "out_frames":out_frames}
        return return_data
        
    # # # # # # # # # # # # # # # #
    #                             #
    # Action Processing Functions #
    #                             #
    # # # # # # # # # # # # # # # #
    
    # get action data
    # - action_ids, handled_obj_ids, semantic variation
    def get_action_data(self, sequence_data, frame_data, obj_data, hand, prefix):
    
        # time segmentations
        time_segmentations = sequence_data["segmentation"][hand]["time"]
                
        # get unpadded frames
        frames = frame_data["frames"]
       
        # get fine grained actions at every timestep
        action_ids = sequence_data["segmentation"][hand]["action_id"] # [num_action_segments]
        action_ids = np.array([action_ids[np.where((frame >= time_segmentations[:,0]) & (frame < time_segmentations[:,1]))[0][0]] for frame in frames])
        action_ohs = one_hot(action_ids,len(self.all_actions))
        
        # get object ids being handled at every timestep
        # -1 if hand not holding anything
        # 0 was reserved for padding
        handled_obj_ids = sequence_data["segmentation"][hand]["object_id"]  
        handled_obj_ids = np.array([handled_obj_ids[np.where((frame >= time_segmentations[:,0]) & (frame < time_segmentations[:,1]))[0][0]] for frame in frames])
                     
        # get object idxs being handled at every timestep
        # - note that obj_ids and obj_xyz, etc are ordered together
        # - -1 if hand not handling any object
        obj_ids = obj_data["obj_ids"]
        handled_obj_idxs = []
        for x in handled_obj_ids:
            handled_obj_idx = np.where(obj_ids == x)[0]
            handled_obj_idx = -1 if len(handled_obj_idx) == 0 else handled_obj_idx[0]
            handled_obj_idxs.append(handled_obj_idx)
        handled_obj_idxs = np.array(handled_obj_idxs)
        
        return_data = {                       
                       # action at every timestep
                       "action_ids":action_ids, "action_ohs":action_ohs,
                       
                       # handled object at every timestep
                       "handled_obj_ids":handled_obj_ids, "handled_obj_idxs":handled_obj_idxs,
                       }        
        return_data = {prefix+"_"+k:v for k,v in return_data.items()} if len(prefix) > 0 else {k:v for k,v in return_data.items()}
        return return_data
    
    # # # # # # # # # # # # # # # #
    #                             #
    # Object Processing Functions #
    #                             #
    # # # # # # # # # # # # # # # #
    
    # get object data
    # - get object meta
    # - object position and rotation
    def get_object_data(self, sequence_data, frame_data):
        
        # # # # # # # # # # # # # # #
        # get object data at frames #
        # # # # # # # # # # # # # # #
        
        # object meta data
        obj_meta  = self.get_object_meta(sequence_data, frame_data)
        
        # unpad frames
        frames = frame_data["frames"]
        
        # # # # # # # # # # # # # # # # #
        # object position and rotation  #
        # # # # # # # # # # # # # # # # #
        
        # object position and rotation
        # - need to unpad for some sequences e.g. Roll as obj_xyz = [t, num_markers, 3] instead of [n, t, num_markers, 3] since squeezing removes the dim
        obj_xyz = sample_readings(sequence_data, category="mocap", items=obj_meta["obj_names"], x_name="time", y_name="mocap_values",  timesteps=frames, return_dict=False) # [n, t, num markers, 3]
        obj_pos = sample_readings(sequence_data, category="root",  items=obj_meta["obj_names"], x_name="time", y_name="root_position", timesteps=frames, return_dict=False) # [n, t, 3]
        obj_rot = sample_readings(sequence_data, category="root",  items=obj_meta["obj_names"], x_name="time", y_name="root_rotation", timesteps=frames, return_dict=False) # [n, t, 3]
        #obj_xyz = np.expand_dims(obj_xyz,axis=0) if len(obj_xyz.shape) == 3 else obj_xyz
        #obj_pos = np.expand_dims(obj_pos,axis=0) if len(obj_pos.shape) == 2 else obj_pos
        #obj_rot = np.expand_dims(obj_rot,axis=0) if len(obj_rot.shape) == 2 else obj_rot
                  
        # get table data
        table_xyz = sample_readings(sequence_data, category="mocap", items=["kitchen_sideboard"], x_name="time", y_name="mocap_values",  timesteps=frames, return_dict=False) # [1, t, num_markers, 3]
        table_pos = sample_readings(sequence_data, category="root",  items=["kitchen_sideboard"], x_name="time", y_name="root_position", timesteps=frames, return_dict=False) # [1, t, 3]
        table_rot = sample_readings(sequence_data, category="root",  items=["kitchen_sideboard"], x_name="time", y_name="root_rotation", timesteps=frames, return_dict=False) # [1, t, 3]
        table_xyz = table_xyz[0] # [t, num_markers, 3]
        table_pos = table_pos[0] # [t, num_markers, 3]
        table_rot = table_rot[0] # [t, num_markers, 3]
        
        # process object
        obj_data = self.process_obj(obj_xyz, obj_pos, obj_rot, table_xyz=table_xyz, table_pos=table_pos, table_rot=table_rot, prefix="obj")
            
        # # # # # # # # # # # # # #
        # get initial object data #
        # # # # # # # # # # # # # #
        
        # initial object position and rotation
        initial_obj_xyz = sample_readings(sequence_data, category="mocap", items=obj_meta["obj_names"], x_name="time", y_name="mocap_values",  timesteps=[0], return_dict=False) # [n, t=1, num markers, 3]
        initial_obj_pos = sample_readings(sequence_data, category="root",  items=obj_meta["obj_names"], x_name="time", y_name="root_position", timesteps=[0], return_dict=False) # [n, t=1, 3]
        initial_obj_rot = sample_readings(sequence_data, category="root",  items=obj_meta["obj_names"], x_name="time", y_name="root_rotation", timesteps=[0], return_dict=False) # [n, t=1, 3]
                
        # get table data
        initial_table_xyz = sample_readings(sequence_data, category="mocap", items=["kitchen_sideboard"], x_name="time", y_name="mocap_values",  timesteps=[0], return_dict=False) # [1, t=1, num_markers, 3]
        initial_table_pos = sample_readings(sequence_data, category="root",  items=["kitchen_sideboard"], x_name="time", y_name="root_position", timesteps=[0], return_dict=False) # [1, t=1, 3]
        initial_table_rot = sample_readings(sequence_data, category="root",  items=["kitchen_sideboard"], x_name="time", y_name="root_rotation", timesteps=[0], return_dict=False) # [1, t=1, 3]
        initial_table_xyz = initial_table_xyz[0] # [t=1, num_markers, 3]
        initial_table_pos = initial_table_pos[0] # [t=1, num_markers, 3]
        initial_table_rot = initial_table_rot[0] # [t=1, num_markers, 3]
                                     
        # process initial object data
        initial_obj_data = self.process_obj(initial_obj_xyz, initial_obj_pos, initial_obj_rot, table_xyz=initial_table_xyz, table_pos=initial_table_pos, table_rot=initial_table_rot, prefix="initial_obj") 
        
        # object adjacency matrix
        obj_adjacency_matrix = 1 - np.eye(self.object_padded_length)
        for i,obj_id in enumerate(obj_meta["obj_ids"]):
            # if obj_id is zero, detach it from all
            if obj_id == 0:
                obj_adjacency_matrix = detach(obj_adjacency_matrix, i)
                
        # object and body adjacency matrix
        if hasattr(self,"adjacency_matrix_type"):
            if self.adjacency_matrix_type == "mtgcn":
                
                # object and human adjacency matrix
                adjacency_matrix = np.zeros([self.object_padded_length + 53, self.object_padded_length + 53])
                
                # object adjacency matrix padded
                obj_adjacency_matrix = np.pad(obj_adjacency_matrix,((0,53),(0,53)))
                
                # human adjacency matrix padded
                human_adjacency_matrix = np.pad(var.mocap_adjacency_matrix,((self.object_padded_length,0),(self.object_padded_length,0)))
                
                # object and human adjacency matrix
                adjacency_matrix = obj_adjacency_matrix + human_adjacency_matrix
                            
        else:
            adjacency_matrix = np.zeros([self.object_padded_length + 53, self.object_padded_length + 53])
                            
        # return data
        return_data = {**obj_meta, **obj_data, **initial_obj_data, "adjacency_matrix":adjacency_matrix}
                
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # object mocap marker position                          #
        # - transformation of obj_mocap_markers must = obj_xyz  #
        # !!! no longer in use. We are using the BPS now        #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        
        """if "train" in self.caller or "test" in self.caller:
            # get object mocap markers
            # - they are already centered and derotated so no need to process
            filename = sequence_data["metadata"]["filename"]
            filename = filename.split("/")
            filename = "_".join([filename[-3],filename[-2],filename[-1]])
            filename = os.path.splitext(filename)[0]                
            obj_mocap_markers = self.object_mocap_markers[filename]
            obj_mocap_markers = np.stack([obj_mocap_markers[obj_name]["xyz"] for obj_name in obj_meta["obj_names"]])    # [num_objects, num_mocap_markers, 3]
            obj_mocap_markers_padded = pad(obj_mocap_markers, self.object_padded_length)                                # [padded n, num_mocap_markers, 3]  
            return_data = {**return_data, "obj_mocap_markers":obj_mocap_markers_padded}"""
        
        return return_data

    # get object meta
    # - object names
    # - id
    # - one hot vectors
    def get_object_meta(self, sequence_data, frame_data):
        
        # object names
        obj_names = sequence_data["metadata"]["object_names"]
        
        # get the table index and remove the table from the object names
        table_idx = obj_names.index("kitchen_sideboard")
        obj_names.pop(table_idx)
        assert "kitchen_sideboard" not in obj_names
        
        # object xml paths
        obj_paths = sequence_data["metadata"]["object_paths"]
        obj_paths.pop(table_idx)
        obj_paths = str(obj_paths)
        
        # object mocap marker names
        obj_mocap_names = [sequence_data["mocap"][obj_name]["mocap_names"] for obj_name in obj_names]
        obj_mocap_names = str(obj_mocap_names)
        
        # object ids and one hots
        obj_ids = np.array([self.object_name_to_id[obj_name] for obj_name in obj_names])    # ignore table
        obj_ids_padded = pad(obj_ids,self.object_padded_length).astype(int)
        obj_ohs_padded = one_hot(obj_ids_padded,self.num_obj_classes)
        
        # object and human id (human id = padded id)
        obj_human_ids = np.concatenate([obj_ids_padded,[0]])
           
        if obj_ids_padded[-1] != 0:
            logger.info("Error! No placeholder node in get_object_meta()!")
            logger.info("Filename: {}".format(sequence_data["metadata"]["filename"]))
            logger.info("obj_ids_padded: {}".format(obj_ids_padded))
            sys.exit()
       
        """
        # object bps
        obj_bps = []
        for object_path in sequence_data["metadata"]["object_paths"]:
            object_path = object_path.replace("xml","json").replace("/home/haziq",os.path.expanduser("~"))
            bps = self.bps[object_path]
            obj_bps.append(bps)
        obj_bps = np.concatenate(obj_bps,axis=0)                                                             #           [n,        1024]
        obj_bps_padded = pad(obj_bps, self.object_padded_length)                                             #           [padded n, 1024]
        #obj_bps_padded = np.expand_dims(obj_bps_padded,axis=0).repeat(self.inp_length+self.out_length,1,1)  # [padded t, padded n, 1024]
        """
       
        return_data = {"obj_names":obj_names,    "obj_mocap_names":obj_mocap_names, 
                       "obj_paths":obj_paths,
                       "obj_ids":obj_ids_padded, "obj_ids_unpadded_length":obj_ids.shape[0], 
                       "obj_ohs":obj_ohs_padded, "obj_ohs_unpadded_length":obj_ids.shape[0],
                       "obj_human_ids":obj_human_ids
                       #"table_idx":table_idx
                       }                   
        return return_data
        
    # process all objects
    # - subtract by table_pos and maybe by the reference object
    # - scale
    # - pad time and number of objects
    def process_obj(self, xyz, pos, rot, table_xyz, table_pos, table_rot, prefix):

        # xyz = [n, t, num_markers, 3]
        # pos = [n, t, 3]
        # rot = [n, t, 3]
        # table_pos = [t, 3]

        # remove table center from pos and rot
        #xyz = np.delete(xyz, table_idx, axis=0) # [n-1, t, num_markers, 3]
        #pos = np.delete(pos, table_idx, axis=0) # [n-1, t, 3]
        #rot = np.delete(rot, table_idx, axis=0) # [n-1, t, 3]

        pos = np.expand_dims(pos,1) if len(pos.shape) == 2 else pos # [n-1, t, 3]
        rot = np.expand_dims(rot,1) if len(rot.shape) == 2 else rot # [n-1, t, 3]

        # convert rotation angles to rotation matrix
        #rot = np.stack([compute_rotation_matrix(obj_thetas) for obj_thetas in rot]) # [n, t, 3, 3]

        # subtract by unscaled table_pos at every timestep
        xyz         = xyz - np.expand_dims(table_pos,axis=1) if table_pos is not None else xyz                  # [n, t, num_markers=4, 3]
        pos         = pos - table_pos if table_pos is not None else pos                                         # [n, t, 3]
        table_xyz   = table_xyz - np.expand_dims(table_pos,axis=1) if table_pos is not None else table_xyz      # [t, num_markers=4, 3]
                
        """
        # transform data such that object and table are oriented wrt global axes
        if self.normalize == 1:
            # logger.info(table_rot.shape)    # [107, 3]
            table_rot_t0 = table_rot[0] # [3]
            rx = compute_rotation_matrix(table_rot_t0[0],"x")
            ry = compute_rotation_matrix(table_rot_t0[1],"y")
            rz = compute_rotation_matrix(table_rot_t0[2],"z")
            r = rz @ ry @ rx
            
            # rotation matrix to view the object position wrt 
            # to the object's coordinate system
            r = r.T # [3x3]
            
            # transform object pos to view it wrt 
            # to the object's coordinate system
            xyz = r @ np.transpose(xyz,[3,0,1,2])
            xyz = np.transpose(xyz,[1,2,3,0])
            #pos_t = r @ np.expand_dims(pos_t,1)
            #transformed_pos.append(pos_t)
        """
        
        # scale
        xyz         = xyz * self.xyz_scale          # [n-1, t, num_markers, 3]
        pos         = pos * self.xyz_scale          # [n-1, t,              3]
        table_xyz   = table_xyz * self.xyz_scale    # [t,      num_markers, 3]
                               
        # pad objects
        xyz_padded = pad(xyz, self.object_padded_length) # [padded n, padded t, num_markers, 3]
        pos_padded = pad(pos, self.object_padded_length)
        rot_padded = pad(rot, self.object_padded_length)
        
        # transpose
        xyz_padded = np.transpose(xyz_padded, (1,0,2,3))    # [padded t, padded n, num_markers, 3]
        pos_padded = np.transpose(pos_padded, (1,0,2))      # [padded t, padded n, 3]
        rot_padded = np.transpose(rot_padded, (1,0,2))      # [padded t, padded n, 3]
        #logger.info(xyz_padded.shape)
        #logger.info(xyz.shape)
        #logger.info(np.sum(xyz_padded[93:]))
        #logger.info(np.sum(xyz_padded[94:]))
        #sys.exit()
                                
        # object velocities
        xyz_vel_padded = np.zeros(xyz_padded.shape)
        xyz_vel_padded[1:] = xyz_padded[1:] - xyz_padded[:-1]
        #xyz_vel_padded[xyz.shape[1]] = xyz_vel_padded[0]
        
        prefix = prefix+"_"
        return_data = {
                       prefix+"xyz":xyz_padded,         prefix+"xyz_unpadded_objects":xyz.shape[0],        # obj_xyz,     obj_xyz_unpadded_length
                       prefix+"xyz_vel":xyz_vel_padded, prefix+"xyz_vel_unpadded_objects":xyz.shape[0],    # obj_xyz_vel, obj_xyz_unpadded_length
                       
                       prefix+"pos":pos_padded,         prefix+"pos_unpadded_objects":pos.shape[0],        # obj_pos, obj_pos_unpadded_length
                       prefix+"rot":rot_padded,         prefix+"rot_unpadded_objects":rot.shape[0],        # obj_rot, obj_rot_unpadded_length
                       prefix+"table_xyz":table_xyz,
                       prefix+"table_pos":table_pos,                                                       # table pos is not at origin
                       prefix+"table_rot":table_rot
                      }
        return return_data

    # # # # # # # # # # # # # # # #
    #                             #
    # Human Processing Functions  #
    #                             #
    # # # # # # # # # # # # # # # #
    
    def get_human_data(self, sequence_data, frame_data, obj_data):
        
        # metadata
        subject_id          = sequence_data["metadata"]["subject_id"]
        subject_height      = np.float32(sequence_data["metadata"]["subject_height"])
        subject_mass        = np.float32(sequence_data["metadata"]["subject_mass"])
        subject_hand_length = np.float32(sequence_data["metadata"]["subject_hand_length"])
        kin_names       = sequence_data["joint"]["body"]["joint_names"]
        xyz_names       = sequence_data["mocap"]["body"]["mocap_names"]
        finger_names    = sequence_data["joint"]["lhand"]["joint_names"] + sequence_data["joint"]["rhand"]["joint_names"]

        # unpad frames
        frames = frame_data["frames"]
        
        # get centers
        table_pos = obj_data["obj_table_pos"]
        
        # get xyz and joint data
        # ========================================
        xyz = sample_readings(sequence_data, category="mocap", items=["body"], x_name="time", y_name="mocap_values", timesteps=frames, return_dict=False)
        kin = sample_readings(sequence_data, category="joint", items=["body"], x_name="time", y_name="joint_values", timesteps=frames, return_dict=False)
        xyz = xyz[0]
        kin = kin[0]
        
        # process xyz and joint data
        xyz = self.process_pose(xyz, table_pos=table_pos, scale=self.xyz_scale, pad_length=self.pose_padded_length, prefix="xyz")
        kin = self.process_pose(kin, table_pos=None,      scale=self.kin_scale, pad_length=self.pose_padded_length, prefix="kin")
                
        # get global position and rotation
        # ========================================
        pos = sample_readings(sequence_data, category="root", items=["body"], x_name="time", y_name="root_position", timesteps=frames, return_dict=False)
        rot = sample_readings(sequence_data, category="root", items=["body"], x_name="time", y_name="root_rotation", timesteps=frames, return_dict=False)
        pos = pos[0]
        rot = rot[0]
        
        # process global position and rotation
        pos = self.process_pose(pos, table_pos=table_pos, scale=1, pad_length=self.pose_padded_length, prefix="pos")
        rot = self.process_pose(rot, table_pos=None,      scale=1, pad_length=self.pose_padded_length, prefix="rot")

        # concatenate [kin,rot,[0,0,0]]
        column_segment = np.zeros([len(frames),3])
        kin["kin"] = np.concatenate([kin["kin"],rot["rot"],column_segment],axis=-1)

        # get finger joint data
        # ========================================
        finger = sample_readings(sequence_data, category="joint", items=["lhand","rhand"], x_name="time", y_name="joint_values", timesteps=frames, return_dict=False) # [2, t, 19]
        finger = self.process_pose(np.transpose(finger,axes=[1,0,2]), table_pos=None, scale=1, pad_length=self.pose_padded_length, prefix="finger")                   # [t, 2, 19]

        # get hand data
        #wrist_ids = np.array([0 for _ in range(2)])         # we use zero (reserved for padding) for the wrist    
        #wrist_ohs = one_hot(wrist_ids,self.num_obj_classes) 
        #wrist_xyz = xyz["xyz"][:,self.hand_xyz_dims]        # [t, hand_xyz_dims, 3]
        
        #logger.info(xyz["xyz"].shape) # (150, 53, 3)
        #logger.info(wrist_xyz.shape)  # (150, 10, 3)
        
        # # # # # # # #
        # velocities  #
        # # # # # # # #
        
        # wrist velocity
        #wrist_xyz_vel = np.zeros(wrist_xyz.shape)                       # [t, hand_xyz_dims, 3]
        #wrist_xyz_vel[1:] = wrist_xyz[1:] - wrist_xyz[:-1]              # [t, hand_xyz_dims, 3]
        #wrist_xyz_vel[xyz["xyz_unpadded_length"]] = wrist_xyz_vel[0]    # [t, hand_xyz_dims, 3]
        
        """
        logger.info(xyz[hand+"_xyz_unpadded_length"])
        logger.info(np.sum(wrist_xyz[xyz[hand+"_xyz_unpadded_length"]:]))
        logger.info(np.sum(wrist_xyz_vel[xyz[hand+"_xyz_unpadded_length"]:]))
        sys.exit()
        """     
    
        # body velocity
        xyz_vel = np.zeros(xyz["xyz"].shape)
        xyz_vel[1:] = xyz["xyz"][1:] - xyz["xyz"][:-1]
        #xyz_vel[xyz["xyz_unpadded_length"]] = xyz_vel[0]
    
        # finger velocity
        finger_vel = np.zeros(finger["finger"].shape)
        finger_vel[1:] = finger["finger"][1:] - finger["finger"][:-1]
        #finger_vel[finger["finger_unpadded_length"]] = finger_vel[0]

        # subject id and joint names
        return_data = {"subject_id":subject_id, "subject_height":subject_height, "subject_mass":subject_mass, "subject_hand_length":subject_hand_length,
                       "kin_names":kin_names, "xyz_names":xyz_names, "finger_names":finger_names,
        
                       # wrist data
                       #"wrist_ohs":wrist_ohs,
                       #"wrist_xyz":wrist_xyz,         
                       #"wrist_xyz_vel":wrist_xyz_vel,
                       
                       # finger data
                       **finger,
                       "finger_vel":finger_vel,
                       
                       # pose data
                       **xyz, **kin,
                       "xyz_vel":xyz_vel,
                       **pos, **rot}

        """
                       # subject id and joint names
        return_data = {"subject_id":subject_id, "subject_height":subject_height, "subject_mass":subject_mass, "subject_hand_length":subject_hand_length,
                       prefix+"kin_names":kin_names, prefix+"xyz_names":xyz_names, prefix+"finger_names":finger_names,
        
                       # wrist data
                       prefix+"wrist_ohs":wrist_ohs,
                       prefix+"wrist_xyz":wrist_xyz,         prefix+"wrist_xyz_unpadded_length":xyz[prefix+"xyz_unpadded_length"],
                       prefix+"wrist_xyz_vel":wrist_xyz_vel, prefix+"wrist_xyz_vel_unpadded_length":xyz[prefix+"xyz_unpadded_length"],
                       
                       # finger data
                       **finger,
                       
                       # pose data
                       **xyz, **kin,
                       **pos, **rot}
        """        
        return return_data

    def process_pose(self, data, table_pos, scale, pad_length, prefix):
            
        # center
        if table_pos is not None:        
            # table_pos.shape = [t,3]
            # xyz.shape = [t,53,3]
            # pos.shape = [t,3]
            table_pos = np.expand_dims(table_pos,1) if len(data.shape) == 3 else table_pos
            data = data - table_pos
        
        # scale
        data = data * scale
                    
        return {prefix:data}

# # # # # # # # # # # # #
#                       #
# processing functions  #
#                       #
# # # # # # # # # # # # #

def normalize(data, a, b):
    return (data - a) / (b - a)

def one_hot(labels, max_label=None):

    one_hot_labels = np.zeros((labels.size, labels.max()+1)) if max_label is None else np.zeros((labels.size, max_label))
    one_hot_labels[np.arange(labels.size),labels] = 1
    
    return one_hot_labels

# get the id of the reference object
# - will be none for approach action
def get_reference_object_name(object_names, action):
    return None

# compute the rotation matrix given thetas in radians and axes
def compute_rotation_matrix(theta, axis):

    assert axis == "x" or axis == "y" or axis == "z"
    
    # form n 3x3 identity arrays
    #n = theta.shape[0] if type(theta) == type(np.array(1)) else 1
    #r = np.zeros((n,3,3),dtype=np.float32)
    #r[:,0,0] = 1
    #r[:,1,1] = 1
    #r[:,2,2] = 1
    
    r = np.zeros(shape=[theta.shape[0],3,3], dtype=np.float32) # [len(theta), 3, 3]

    if axis == "x":
        #r = np.array([[1, 0,              0],
        #              [0, np.cos(theta), -np.sin(theta)],
        #              [0, np.sin(theta),  np.cos(theta)]])
        r[:,0,0] =  1
        r[:,1,1] =  np.cos(theta)
        r[:,1,2] = -np.sin(theta)
        r[:,2,1] =  np.sin(theta)
        r[:,2,2] =  np.cos(theta)
                     
    if axis == "y":
        #r = np.array([[ np.cos(theta), 0,  np.sin(theta)],
        #              [ 0,             1,  0],
        #              [-np.sin(theta), 0,  np.cos(theta)]])
        r[:,1,1] =  1
        r[:,0,0] =  np.cos(theta)
        r[:,0,2] =  np.sin(theta)
        r[:,2,0] = -np.sin(theta)
        r[:,2,2] =  np.cos(theta)

    if axis == "z":
        #r = np.array([[np.cos(theta), -np.sin(theta), 0],
        #              [np.sin(theta),  np.cos(theta), 0],
        #              [0,              0,             1]])
        r[:,2,2] =  1
        r[:,0,0] =  np.cos(theta)
        r[:,0,1] = -np.sin(theta)
        r[:,1,0] =  np.sin(theta)
        r[:,1,1] =  np.cos(theta)
    
    return r

# pad data
def pad(data, pad_length, return_unpadded_length=0):

    #logger.info(data.shape)
    # data must be [t, ...]
    
    unpadded_length = data.shape[0]
    if pad_length < unpadded_length:
        print("pad_length too short !")
        print("Pad Length = {}".format(pad_length))
        print("Unpadded Sequence Length = {}".format(data.shape))
        a=b
        sys.exit()

    new_shape = [pad_length] + list(data.shape[1:])
    new_shape = tuple(new_shape)
    data_padded = np.zeros(shape=new_shape)
    data_padded[:unpadded_length] = data    
    
    assert np.array_equal(data_padded[:unpadded_length],data)
    assert np.all(data_padded[unpadded_length:] == 0)    
    
    if return_unpadded_length:
        return unpadded_length, data_padded.astype(np.float32) 
    return data_padded.astype(np.float32)

# compute data velocities
def compute_velocities(data, discard_t0=1):
    
    data_vel = np.zeros(data.shape, dtype=data.dtype)
    
    if len(data.shape) == 2:
        
        # data must be [t, num_joints]
        data_vel[1:] = data[1:] - data[:-1]
        if discard_t0 == 1:
            data_vel = data_vel[1:]
        
    elif len(data.shape) == 3:     
    
        # data must be [t, num_joints]
        data_vel[:,1:] = data[:,1:] - data[:,:-1]
        if discard_t0 == 1:
            data_vel = data_vel[:,1:]
        
    else:
        sys.exit("Wrong number of dimensions")
    
    return data_vel

# # # # # # # # # # # #
#                     #
# sampling functions  #
#                     #
# # # # # # # # # # # #

# get the first and final frame given the input frame
def get_row(segmentation_timesteps, inp_frame):

    row = np.where(np.logical_and(segmentation_timesteps[:,0] <= inp_frame, segmentation_timesteps[:,1] >= inp_frame))[0]
    assert row.shape[0] == 1
    return row[0]
     
# sample the readings at timestep by interpolating the left and right values
def sample_readings(data, category, items, x_name, y_name, timesteps, return_dict=True):

    sampled_readings = {}
    for item in items:
        
        time = data[category][item][x_name]    # [n]
        values = data[category][item][y_name]  # [n,3] or [n,m,3]
        
        yt_list = []
        for t in timesteps:
            
            # get interpolating index
            # https://stackoverflow.com/questions/36275459/find-the-closest-elements-above-and-below-a-given-number
                        
            if t <= np.min(time):
                yt_list.append(values[0])
                
            elif t >= np.max(time):
                yt_list.append(values[-1])
                
            else:
                #logger.info("time",time)
                #logger.info("t",t)
                #logger.info("time[time < t]",time[time < t])
                try:
                    i1 = np.where(time == time[time <= t].max())[0][0]
                    i2 = np.where(time == time[time > t].min())[0][0]                    
                except:
                    logger.info(time)
                    logger.info("filename = {}".format(data["metadata"]["filename"]))
                    logger.info("t = {}".format(t))
                    logger.info("time.shape = {}".format(time.shape))
                    logger.info("values.shape = {}".format(values.shape))
                    logger.info("Error here")
                    logger.info(np.where(time == time[time < t].max())[0][0])
                    logger.info(np.where(time == time[time > t].min())[0][0])
                    sys.exit()
        
                # time.shape    [t]
                # values.shape  [t, num_mocap / num_joint /, 3]        
                # y = joint values
                # x = timestep
                x = np.array([time[i1],time[i2]])               
                y = np.stack([values[i1],values[i2]],axis=-1)
                
                # interpolate 
                f = interpolate.interp1d(x, y)
                yt = f(t)
                yt_list.append(yt)
        sampled_readings[item] = np.array(yt_list)
    
    if return_dict:
        return sampled_readings
    return np.stack([v for k,v in sampled_readings.items()])
    #return np.squeeze(np.stack([v for k,v in sampled_readings.items()]))
