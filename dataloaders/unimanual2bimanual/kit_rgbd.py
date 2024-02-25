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

from scipy import interpolate
from collections import Counter, OrderedDict
       
sys.path.append(os.path.join(os.path.expanduser("~"),"Forecasting-Bimanual-Object-Manipulation-Sequences-From-Unimanual-Observations","dataloaders"))
from kit_rgbd_main_loader import *
from utils import *
import kit_rgbd_variables as var
           
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
        print("Loading time: {}".format(t2-t1))
        
        # pass all to self
        for key, value in data.__dict__.items():
            setattr(self, key, value)
                          
        """
        create indexer                    
        - this is where i discard sequences / subsequences
        - self.sequence_data remains unaffected
        """

        """self.indexer = []
        for sequence_idx,sequence_data in enumerate(self.sequence_data):
            # indexer to index sequence_data
            indexer = [{"sequence_idx":sequence_idx, "main_action":sequence_data["metadata"]["main_action"]}]
            self.indexer.extend(indexer)
        self.data_len = len(self.indexer)"""
                
        self.indexer = []
        for sequence_idx,sequence_data in enumerate(self.sequence_data):
        
            """indexer           = []               
            filename          = sequence_data["metadata"]["filename"]                                                
            main_action       = sequence_data["metadata"]["main_action"]
                        
            # max time given the input and output lengths
            assert len(sequence_data["segmentation"]["lhand"]) == len(sequence_data["segmentation"]["rhand"])
            max_time = len(sequence_data["segmentation"]["lhand"]) - (self.inp_length+self.out_length)*self.time_step_size
                                
            inp_frame = 0
            while inp_frame < max_time:      
                indexer = [{"sequence_idx":sequence_idx, "inp_frame":inp_frame}]
                self.indexer.extend(indexer)
                inp_frame += self.resolution"""
                
            """
            because the person is moving, i need to start the inp_frame and max_time by computing the distance to the table
            """
            
            key = "main_3d_objects" # must always be main_3d_objects or all_3d_objects because we want the z dimension
        
            # get start stop frames
            table_pos = sequence_data["table_center"]
            table_z   = table_pos[2]
                        
            # get hand bbox
            # note that timesteps[-1] may not != sequence_data["time"][-1]
            timesteps = sequence_data["time"]
            timesteps = np.arange(timesteps[0], timesteps[-1], 1)
            hand_bbox = sample_readings(sequence_data, category=key, items=[x for x in sequence_data[key].keys() if "Hand" in x], x_name="time", y_name="bbox", timesteps=timesteps, return_dict=False) # [n, t, 2, 3]
            
            # compute distance from hand to table
            z = []
            for t in range(hand_bbox.shape[1]):
                l = np.mean(hand_bbox[0,t],axis=0)[2]
                r = np.mean(hand_bbox[1,t],axis=0)[2]
                hand_closest_to_table_z = max(l,r)
                z.append(hand_closest_to_table_z - table_z)
            z = np.array(z)
            
            # start and end frame
            start_frame_idx = np.argmax(z>-500)
            end_frame_idx   = len(z) - np.argmax(z[::-1]>-500)
            inp_frame = sequence_data["time"][start_frame_idx]
            end_frame = sequence_data["time"][end_frame_idx]
            
            # update end frame
            end_frame = end_frame - (self.inp_length+self.out_length)*self.time_step_size
            if end_frame - inp_frame <= 10:
                continue
            
            # collect
            while inp_frame < end_frame:
                indexer = [{"sequence_idx":sequence_idx, "inp_frame":inp_frame, "main_action":sequence_data["metadata"]["main_action"]}]
                self.indexer.extend(indexer)
                inp_frame += self.resolution                
        self.data_len = len(self.indexer)
        logger.info("{} has {} samples".format(dtype,len(self.indexer)))
                
    def __len__(self):
        
        # X or 32
        return max(len(self.indexer),self.batch_size)
    
    def __getitem__(self, idx, is_distractor=0):

        # resample a random value if the sampled idx goes beyond data_len. This ensures that it does not matter how I augment the data
        if idx > self.data_len:
            idx = random.randint(0,self.__len__())
            
        # data
        indexer = self.indexer[idx]
        sequence_data = self.sequence_data[indexer["sequence_idx"]]
        
        # path to take
        subject     = sequence_data["metadata"]["subject"]
        main_action = sequence_data["metadata"]["main_action"]
        take        = sequence_data["metadata"]["take"]
        
        # sequence name
        sequence = sequence_data["metadata"]["filename"]
        sequence = sequence.split("/")
        sequence = "_".join([sequence[-3],sequence[-2],sequence[-1]])
                
        # # # # # # # # # # # # # #
        #                         #
        # get input and key frame #
        #                         #
        # # # # # # # # # # # # # #   
                
        # get frame data
        inp_frame = indexer["inp_frame"]
        frames = np.array([inp_frame+i*self.time_step_size for i in range(self.inp_length+self.out_length)])
        frame_data = {"inp_frames":frames[:self.inp_length], "frames":frames}
                        
        # # # # # # # # # # # # #
        # denormalization data  #  
        # # # # # # # # # # # # #
        
        table_center = sequence_data["table_center"]
        angle = sequence_data["angle"] 
        rx    = compute_rotation_matrix(angle * np.pi / 180, "x")
        
        # # # # # # # #
        # camera data #
        # # # # # # # #
        
        cx, cy = np.expand_dims(np.array(sequence_data["camera_data"]["cx"]),axis=0), np.expand_dims(np.array(sequence_data["camera_data"]["cy"]),axis=0)
        fx, fy = np.expand_dims(np.array(sequence_data["camera_data"]["fx"]),axis=0), np.expand_dims(np.array(sequence_data["camera_data"]["fy"]),axis=0)
        
        # # # # # # # # # # # # # #
        # get action data         #  
        # # # # # # # # # # # # # #
        
        lhand_action_data = self.get_action_data(sequence_data, frame_data, "lhand")
        rhand_action_data = self.get_action_data(sequence_data, frame_data, "rhand")
        hand_action_data = {}
        hand_action_data["hand_action_ids"] = np.stack([lhand_action_data["lhand_action_ids"],rhand_action_data["rhand_action_ids"]],axis=1)
                
        # # # # # # # # # # # # # #
        # get pose and hand data  #
        # # # # # # # # # # # # # #

        # get pose data
        pose_data = self.get_pose_data(sequence_data, frame_data)
        #print(pose_data["xyz"].shape) # [pose_padded_length, num_joints, xy] [150, 15, 2]
        
        # get hand data
        hand_data = self.get_hand_data(sequence_data, frame_data)
        #print(hand_data["finger"].shape) # [pose_padded_length, hands, joints*xy] [150, 2, 42]
                
        # # # # # # # # # # # # # #
        # get object data         #  
        # # # # # # # # # # # # # #
                
        # get object data
        obj_data = self.get_object_data(sequence_data, frame_data) 
        
        # # # # # # # # # # # # # # #
        # get get_handled_obj_data  #  
        # # # # # # # # # # # # # # #
        
        contact_data = self.get_handled_obj_data(obj_data, lhand_action_data, hand_action_data, frame_data)
                
        # # # # # # # #
        # return data #
        # # # # # # # #
        
        return_data = {# metadata
                       "sequence":sequence, "inp_frame":inp_frame,
                       "subject":subject, "main_action":main_action, "take":take,
                       
                       # denormalization data
                       "xyz_scale":np.array(self.xyz_scale),
                       "table_center":table_center, "angle":angle, "rx":rx,
                       
                       # camera data
                       "cx":cx, "cy":cy,
                       "fx":fx, "fy":fy,
                                                       
                       # frame data
                       **frame_data,
                                
                       # action data
                       **lhand_action_data, **rhand_action_data,
                       **hand_action_data,
                       **contact_data,
                       
                       # pose and hand data
                       **pose_data,
                       **hand_data,
                       
                       # object data
                       **obj_data}
                
        # separate into inp, future, past
        return_data = self.split_data(return_data)
    
        # generate the masks
        return_data = self.generate_mask(return_data, p=self.p, prefix="")
        
        # convert all list of strings into a single string
        for k,v in return_data.items():
            if type(v) == type([]) and type(v[0]) == type("string"):
                return_data[k] = str(v)
                
        # convert all array type to float32
        #for k,v in return_data.items():
        #    if type(v) == type(np.array(1)):
        #        return_data[k] = return_data[k].astype(np.float32)
        
        # convert array type to float32
        for k,v in return_data.items():
            if type(v) == type(np.array(1.0)) and all([k != x for x in ["inp_xyz_mask_idxs","out_xyz_mask_idxs","kin_mask_idxs","inp_handled_obj_idxs","out_handled_obj_idxs","out_finger_mask_idxs"]]):
                return_data[k] = return_data[k].astype(np.float32)
                
        """
        for k,v in return_data.items():
            if type(v) == type(np.array([1])):
                print(k, v.shape)
        print()
        """  
        
        """
        print(sequence_data["metadata"]["filename"])
        print(obj_data["obj_xyz_unpadded_length"], obj_data["obj_xyz"].shape)
        print(obj_data["wrist_xyz_unpadded_length"], obj_data["wrist_xyz"].shape)
        for i in range(obj_data["wrist_xyz_unpadded_length"]):
            print(i)
            print()
            print(obj_data["obj_xyz"][i])
            print()
            print(obj_data["wrist_xyz"][i])
            input()
        sys.exit()
        """
                
        return return_data

    def generate_mask(self, data, p=None, prefix=""):
                            
        # during the inp frame
        # (done) probability of masking either the left or right
        # mask object if hand was in contact with object
        # check if using exp or xyz
                
        # mask out l arm if p < 0.5
        # mask out r arm if p > 0.5
        p = np.random.uniform(low=0.0,high=1.0,size=1)[0] if p == None else p
        inp_masked_hand = "Left" if p < 0.5 else "Right"
        masked_hand_idx = 0 if p < 0.5 else 1
                
        """
        mask wrist
        """
                
        inp_masked_wrist_xyz = np.copy(data["inp_wrist_xyz"])   # [inp_length, 2, 3]
        inp_masked_wrist_xyz[:,masked_hand_idx] = 0
                
        """
        generate inp and out xyz mask
        - out needed for teacher forcing
        """
        
        # inp xyz_mask_idxs
        inp_xyz_mask_idxs = self.l_arm_mocap_idxs if p < 0.5 else self.r_arm_mocap_idxs
        inp_xyz_mask_idxs = np.array(inp_xyz_mask_idxs)
        
        # inp_masked_xyz - mocap markers but where the ones that are masked are set to 0
        inp_masked_xyz = np.copy(data["inp_xyz"])   # [inp_length, 15, 2]
        inp_masked_xyz[:,inp_xyz_mask_idxs] = 0
        
        # out_xyz_mask_idxs
        out_xyz_mask_idxs = np.stack([self.l_arm_mocap_idxs,self.r_arm_mocap_idxs],axis=0)
        
        # out_masked_xyz
        out_masked_xyz = np.copy(data["out_xyz"])                       #    [inp_length, 15, 2]
        out_masked_xyz = np.repeat(out_masked_xyz[None,:,:,:],2,axis=0) # [2, inp_length, 15, 2]
        for i,x in enumerate(out_xyz_mask_idxs):
            out_masked_xyz[i,:,x] = 0
                
        # missing_xyz - mocap markers that were masked
        inp_missing_xyz = np.copy(data["inp_xyz"][:,inp_xyz_mask_idxs])
        
        # remaining_xyz - remaining mocap markers that were not masked
        inp_remaining_xyz = np.copy(data["inp_xyz"])[:,[i for i in range(data["inp_xyz"].shape[1]) if i not in inp_xyz_mask_idxs]] #[inp_length, 13, 2]
                     
        # # # # # # # # # # # #
        # mask finger joints  #
        # # # # # # # # # # # #
        
        # mask input finger joints
        inp_finger_mask_idxs = 0 if p < 0.5 else 1
        inp_masked_finger = np.copy(data["inp_finger"])                             # [inp_length, 2, 42]
        inp_masked_finger[:,inp_finger_mask_idxs] = 0                               
        inp_missing_finger = np.copy(data["inp_finger"])[:,inp_finger_mask_idxs]    # [inp_length, 42]
        
        # mask output finger joints
        out_masked_finger = np.copy(data["out_finger"])                         #    [out_length, 2, 42]
        out_masked_finger = np.repeat(out_masked_finger[None,:,:,:],2,axis=0)   # [2, out_length, 2, 42]
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
        #print(initial_obj_xyz[0,:,0])
                
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
        #grasp_probability = np.copy(handled_obj_ids)
        #grasp_probability[grasp_probability != -1] = 1
        #grasp_probability[grasp_probability == -1] = 0
        
        # assert for invalid cases
        # - person picks up object then places it down in a sequence
        # WE NO LONGER HAVE SUCH CASES
        """x = handled_obj_ids[:-1] != handled_obj_ids[1:]
        #logger.info("x",x)
        if np.sum(x) > 1:
            print("Invalid sequence in process_data() {}".format(handled_obj_ids))
            print("Sequence name: {}".format(data["sequence"]))
            print("Masked hand: {}".format(masked_hand))
            print("inp_frames: {}".format(data["inp_frames"]))
            sys.exit()"""

        # assert for invalid cases
        # - person's one hand handles more than 1 object
        x = set(handled_obj_ids)
        x.discard(-1)
        if len(x) > 1:
            print("Invalid sequence in process_data() {}".format(handled_obj_ids))
            print("handled_obj_ids".format(data["inp_handled_obj_ids"]))
            print("Sequence name: {}".format(data["sequence"]))
            print("Masked hand: {}".format(masked_hand))
            print("inp_frames: {}".format(data["inp_frames"]))
            sys.exit()
            
        # determine case
        # if [-1 -1 -1 -1 -1], not handling anything
        case = None
        if np.sum(handled_obj_ids == -1) == self.inp_length:
            case = 1
            #print("pass")
            handled_obj_id  = -1
            handled_obj_idx = -1
            missing_obj_xyz = np.zeros(data["inp_obj_xyz"][:,0].shape)
            pass
            
        # if [1 1 1 1 1], person currently handling object
        # - set handled object to t[initial]
        elif np.sum(handled_obj_ids > -1) == self.inp_length:
            case = 2
            #print("if [1 1 1 1 1], set handled object to t[initial]")
            #print(handled_obj_ids)
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
            #print("handled_obj_idx",handled_obj_idx)
            masked_obj_xyz[:,handled_obj_idx] = np.copy(initial_obj_xyz[:,handled_obj_idx]) # [inp_length, obj_padded_length, 4, 3]
            missing_obj_xyz = np.copy(data["inp_obj_xyz"][:,handled_obj_idx])               # [inp_length,                    4, 3]
            #print(masked_obj_xyz.shape, missing_obj_xyz.shape)
        
        # if [-1 1 1 ...], person picked object up before performing some action
        # - set the handled object throughout inp_length to t[0]
        elif np.sum(handled_obj_ids > -1) > 0 and np.sum(handled_obj_ids > -1) < self.inp_length and handled_obj_ids[0] == -1 and handled_obj_ids[-1] != -1:
            case = 3
            #print("if [-1 1 1 ...], set the handled object throughout inp_length to t[0]")
            #print(handled_obj_ids)
            handled_obj_id = [x for x in list(set(handled_obj_ids)) if x != -1] # ignore -1
            if len(handled_obj_id) > 1:
                print("More than 1 handled_obj_ids detected in process_data() 2 {}".format(handled_obj_id))
                sys.exit()
            handled_obj_id  = handled_obj_id[0]
            handled_obj_idx = np.where(obj_ids == handled_obj_id)[0][0]
            #print("handled_obj_idx",handled_obj_idx)
            masked_obj_xyz[:,handled_obj_idx] = np.copy(masked_obj_xyz[0:1,handled_obj_idx]) # [inp_length, obj_padded_length, 4, 3]
            missing_obj_xyz = np.copy(data["inp_obj_xyz"][:,handled_obj_idx])                # [inp_length,                    4, 3]
            #print(masked_obj_xyz.shape, missing_obj_xyz.shape)
        
        # if [1 -1 -1 ...], person placed object down after performing some action
        # - set the handled object to where the first -1 is
        elif np.sum(handled_obj_ids > -1) > 0 and np.sum(handled_obj_ids > -1) < self.inp_length and handled_obj_ids[0] != -1 and handled_obj_ids[-1] == -1:
            case = 4
            #print("if [1 -1 -1 ...], set the handled object to where the first -1 is")
            #print(handled_obj_ids)
            handled_obj_id = [x for x in list(set(handled_obj_ids)) if x != -1] # ignore -1
            if len(handled_obj_id) > 1:
                print("More than 1 handled_obj_ids detected in process_data() 3 {}".format(handled_obj_id))
                sys.exit()
            handled_obj_id  = handled_obj_id[0]
            handled_obj_idx = np.where(obj_ids == handled_obj_id)[0][0]
            #print("handled_obj_idx",handled_obj_idx)
            t = np.where(x)[0][0]+1
            #print("t",t)
            masked_obj_xyz[:,handled_obj_idx] = np.copy(masked_obj_xyz[t:t+1,handled_obj_idx])
            missing_obj_xyz = np.copy(data["inp_obj_xyz"][:,handled_obj_idx])
            #print(masked_obj_xyz.shape, missing_obj_xyz.shape)
        
        # if [-1 1 1 ... -1], person picked object up, performed action, then put it down
        # - same as case 3: set the handled object throughout inp_length to t[0]
        elif np.sum(handled_obj_ids > -1) > 0 and np.sum(handled_obj_ids > -1) < self.inp_length and handled_obj_ids[0] == -1 and handled_obj_ids[-1] == -1:
            case = 5
            #print("if [-1 1 1 ...], set the handled object throughout inp_length to t[0]")
            #print(handled_obj_ids)
            handled_obj_id = [x for x in list(set(handled_obj_ids)) if x != -1] # ignore -1
            if len(handled_obj_id) > 1:
                print("More than 1 handled_obj_ids detected in process_data() 2 {}".format(handled_obj_id))
                sys.exit()
            handled_obj_id  = handled_obj_id[0]
            handled_obj_idx = np.where(obj_ids == handled_obj_id)[0][0]
            #logger.info("handled_obj_idx",handled_obj_idx)
            masked_obj_xyz[:,handled_obj_idx] = np.copy(masked_obj_xyz[0:1,handled_obj_idx]) # [inp_length, obj_padded_length, 4, 3]
            missing_obj_xyz = np.copy(data["inp_obj_xyz"][:,handled_obj_idx])                # [inp_length,                    4, 3]
            #print(masked_obj_xyz.shape, missing_obj_xyz.shape)
        
        # if [1 -1 -1 ...], person placed object down after performing some action then picked something up
        # - same as case 4: set the handled object to where the first -1 is
        elif np.sum(handled_obj_ids > -1) > 0 and np.sum(handled_obj_ids > -1) < self.inp_length and handled_obj_ids[0] != -1 and handled_obj_ids[-1] != -1:
            case = 6
            #print("if [1 -1 -1 ...], set the handled object to where the first -1 is")
            #print(handled_obj_ids)
            handled_obj_id = [x for x in list(set(handled_obj_ids)) if x != -1] # ignore -1
            if len(handled_obj_id) > 1:
                print("More than 1 handled_obj_ids detected in process_data() 3 {}".format(handled_obj_id))
                sys.exit()
            handled_obj_id  = handled_obj_id[0]
            handled_obj_idx = np.where(obj_ids == handled_obj_id)[0][0]
            #print("handled_obj_idx",handled_obj_idx)
            t = np.where(x)[0][0]+1
            #print("t",t)
            masked_obj_xyz[:,handled_obj_idx] = np.copy(masked_obj_xyz[t:t+1,handled_obj_idx])
            missing_obj_xyz = np.copy(data["inp_obj_xyz"][:,handled_obj_idx])
            #print(masked_obj_xyz.shape, missing_obj_xyz.shape)
        
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
        
        """# correct
        if case == 2:
            print("masked_obj_xyz")
            print(masked_obj_xyz[:,handled_obj_idx])
            print("inp_obj_xyz")
            print(data["inp_obj_xyz"][:,handled_obj_idx])
            print(handled_obj_idx)
            print(case)
            
        # correct
        if case == 3:
            print("masked_obj_xyz")
            print(masked_obj_xyz[:,handled_obj_idx])
            print("inp_obj_xyz")
            print(data["inp_obj_xyz"][:,handled_obj_idx])
            print(handled_obj_idx)
            print(case)
            
        # correct
        if case == 4:
            print("masked_obj_xyz")
            print(masked_obj_xyz[:,handled_obj_idx])
            print("inp_obj_xyz")
            print(data["inp_obj_xyz"][:,handled_obj_idx])
            print(handled_obj_idx)
            print(case)
            
        # correct
        if case == 5:
            print("masked_obj_xyz")
            print(masked_obj_xyz[:,handled_obj_idx])
            print("inp_obj_xyz")
            print(data["inp_obj_xyz"][:,handled_obj_idx])
            print(handled_obj_idx)
            print(case)
            
        # correct
        if case == 6:
            print("masked_obj_xyz")
            print(masked_obj_xyz[:,handled_obj_idx])
            print("inp_obj_xyz")
            print(data["inp_obj_xyz"][:,handled_obj_idx])
            print(handled_obj_idx)
            print(case)
            
        input()"""
        
        return_data = {"inp_masked_hand":inp_masked_hand, "masked_hand_idx":masked_hand_idx,
        
                       # wrist
                       "inp_masked_wrist_xyz":inp_masked_wrist_xyz,
        
                       # handled_obj_ids and grasp_probability
                       "inp_handled_obj_ids":handled_obj_ids, "inp_handled_obj_idxs":handled_obj_idxs, #"grasp_probability":grasp_probability,
        
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

    def split_data(self, data):
        
        # split the relevant keys into inp, future, past
        keys = ["finger","obj_xyz", "frames", "xyz", "obj_pos","finger_confidence","wrist_xyz"]
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
    
    def get_handled_obj_data(self, obj_data, lhand_action_data, hand_action_data, frame_data):
                
        # wrist centers
        wrist = obj_data["wrist_xyz"]           # [t, 2, 3]
        
        # hand centers
        #finger = hand_data["finger"]                                        # [t, 2, 42]
        #finger = np.reshape(finger,[finger.shape[0],finger.shape[1],-1,2])  # [t, 2, 21, 2]
        #finger = np.mean(finger,axis=2)                                     # [t, 2, 2]
        
        # object coordinates
        #print(obj_data["obj_ids"])
        #print(obj_data["obj_xyz_unpadded_objects"])
        obj_xyz_unpadded_objects = obj_data["obj_xyz_unpadded_objects"]
        obj_xyz = obj_data["obj_xyz"][:,:obj_xyz_unpadded_objects]  # [t, obj_xyz_unpadded_objects, 1, 3]
        obj_xyz = obj_xyz[:,:,0]
        
        # compute distance from left and right hand to every object in the scene
        lhand_dist = np.sqrt(np.sum((wrist[:,0:1] - obj_xyz)**2,axis=-1))  # [t, obj_xyz_unpadded_objects]
        rhand_dist = np.sqrt(np.sum((wrist[:,1:2] - obj_xyz)**2,axis=-1))  # [t, obj_xyz_unpadded_objects]
        
        # get idx of min
        lhand_obj_idx = np.argmin(lhand_dist,axis=-1) # [t]
        rhand_obj_idx = np.argmin(rhand_dist,axis=-1) # [t]
        handled_obj_idxs = np.stack([lhand_obj_idx,rhand_obj_idx],axis=-1) # [t, 2]
        #print(handled_obj_idxs)
        #print(handled_obj_idxs.shape)
        
        # update idx, set to -1 if action = idle/approach/...
        hand_action_ids = hand_action_data["hand_action_ids"] # [t, 2]
        for t,hand_action_ids_t in enumerate(hand_action_ids):
            handled_obj_idxs[t,0] = -1 if hand_action_ids_t[0] in [0,1,2] else handled_obj_idxs[t,0]
            handled_obj_idxs[t,1] = -1 if hand_action_ids_t[1] in [0,1,2] else handled_obj_idxs[t,1]
        
        # update idx, set to -1 if action = idle/approach/...
        obj_ids = obj_data["obj_ids"]
        handled_obj_ids = np.zeros(handled_obj_idxs.shape)
        for t in range(handled_obj_idxs.shape[0]):
            handled_obj_ids[t,0] = obj_ids[handled_obj_idxs[t,0]] if handled_obj_idxs[t,0] != -1 else -1
            handled_obj_ids[t,1] = obj_ids[handled_obj_idxs[t,1]] if handled_obj_idxs[t,1] != -1 else -1
        
        # transpose
        handled_obj_ids  = np.transpose(handled_obj_ids,[1,0])  # [2, t]
        handled_obj_idxs = np.transpose(handled_obj_idxs,[1,0]) # [2, t]
        
        # !!!!! WE CURRENTLY ASSUME THERE TO BE ONLY ONE MASKED OBJECT FOR THE KIT RGBD DATASET
        for i in range(2):
            val = int(Counter(handled_obj_ids[i]).most_common(1)[0][0])
            for t in range(handled_obj_ids.shape[1]):
                handled_obj_ids[i,t] = val if handled_obj_ids[i,t] != -1 else -1
            val = int(Counter(handled_obj_idxs[i]).most_common(1)[0][0])
            for t in range(handled_obj_idxs.shape[1]):
                handled_obj_idxs[i,t] = val if handled_obj_idxs[i,t] != -1 else -1
                                
        return {"handled_obj_idxs":handled_obj_idxs, "handled_obj_ids":handled_obj_ids}
    
    # # # # # # # # # # # # # # # #
    #                             #
    # Action Processing Functions #
    #                             #
    # # # # # # # # # # # # # # # #
    
    # get action data
    def get_action_data(self, sequence_data, frame_data, hand):
                                                     
        # # # # # # # #
        # fine action #
        # # # # # # # #
        
        frames = frame_data["frames"]
        
        # get fine action at timesteps
        fine_action_ids = sequence_data["segmentation"][hand][frames]                                   # some entries may contain None 
        fine_action_ids_mask = np.array([0 if x is None else 1 for x in fine_action_ids]).astype(int)   # mask for None. 0 means ignore. 1 means do not ignore.
        fine_action_ids = np.array([0 if x is None else x for x in fine_action_ids]).astype(int)        # convert None to 0 for the one_hot function
        fine_action_ohs = one_hot(fine_action_ids, len(self.fine_actions))
                                          
        # pad
        #fine_action_ids_padded = pad(fine_action_ids, self.pose_padded_length)
        #fine_action_ids_mask_padded = pad(fine_action_ids_mask, self.pose_padded_length)
        #fine_action_ohs_padded = pad(fine_action_ohs, self.pose_padded_length)
                
        return_data = {# fine actions
                       "action_ids":fine_action_ids,
                       "action_ids_mask":fine_action_ids_mask,
                       "action_ohs":fine_action_ohs
                       }        
        return_data = {hand+"_"+k:v for k,v in return_data.items()}
        return return_data

    # # # # # # # # # # # # # # # #
    #                             #
    # Frame Processing Functions  #
    #                             #
    # # # # # # # # # # # # # # # #
    
    def get_frame_data(self, sequence_data):
        
        key = "main_3d_objects" # must always be main_3d_objects or all_3d_objects because we want the z dimension
        
        # get start stop frames
        table_pos = sequence_data["table_center"]
        table_z   = table_pos[2]
        
        # get hand bbox
        # note that timesteps[-1] may not != sequence_data["time"][-1]
        timesteps = sequence_data["time"]
        timesteps = np.arange(timesteps[0], timesteps[-1], 1)
        hand_bbox = sample_readings(sequence_data, category=key, items=[x for x in sequence_data[key].keys() if "Hand" in x], x_name="time", y_name="bbox", timesteps=timesteps, return_dict=False) # [n, t, 2, 3]
        
        # compute distance from hand to table
        z = []
        for t in range(hand_bbox.shape[1]):
            l = np.mean(hand_bbox[0,t],axis=0)[2]
            r = np.mean(hand_bbox[1,t],axis=0)[2]
            hand_closest_to_table_z = max(l,r)
            z.append(hand_closest_to_table_z - table_z)
        z = np.array(z)
        
        # start and end frame
        start_frame_idx = np.argmax(z>-500)
        end_frame_idx   = len(z) - np.argmax(z[::-1]>-500)
        start_frame = sequence_data["time"][start_frame_idx]
        end_frame   = sequence_data["time"][end_frame_idx]
        #frames      = 
        #timesteps   = np.arange(start_frame, end_frame, self.time_step_size)
        #timesteps_padded = pad(timesteps,self.pose_padded_length)
        #print((end_frame - start_frame)/30, "+ \\")
        #print(timesteps)
        #sys.exit()
        
        # compute the sequence duration
        sequence_duration = int(np.ceil((end_frame - start_frame)/self.time_step_size))
                                
        return_data = {"start_frame_idx":start_frame_idx, "end_frame_idx":end_frame_idx,
                       "start_frame":start_frame, "end_frame":end_frame,
                       
                       "timesteps":timesteps_padded, "timesteps_unpadded_length":timesteps.shape[0],
                       "sequence_duration":sequence_duration}
        #return_data = {hand+"_"+k:v for k,v in return_data.items()}
        return return_data
    
    # # # # # # # # # # # # # # #
    #                           #
    # Pose Processing Functions #
    #                           #
    # # # # # # # # # # # # # # #
    
    def get_pose_data(self, sequence_data, frame_data):
    
        #start_frame = frame_data["start_frame"]
        #end_frame   = frame_data["end_frame"]
        #timesteps   = np.arange(start_frame, end_frame, self.time_step_size)
        frames = frame_data["frames"]
        timesteps = frames
        
        # pose xy data in image space
        pose_xy            = sample_readings(sequence_data, category="person", items=["pose_2d"], x_name="time", y_name="xy",         timesteps=timesteps, return_dict=False) # [t, 15, 2]
        pose_xy_confidence = sample_readings(sequence_data, category="person", items=["pose_2d"], x_name="time", y_name="confidence", timesteps=timesteps, return_dict=False) # [t, 15]
        pose_xy = pose_xy[0]
        pose_xy = pose_xy * self.xyz_scale
        pose_xy_confidence = pose_xy_confidence[0]
        pose_xy_confidence = np.repeat(pose_xy_confidence[:,:,np.newaxis],2,axis=2)
        
        # pad
        #pose_xy_padded            = pad(pose_xy, self.pose_padded_length)
        #pose_xy_confidence_padded = pad(pose_xy_confidence, self.pose_padded_length)
            
        # compute velocities
        #pose_xy_vel_padded = np.zeros(pose_xy_padded.shape)
        #pose_xy_vel_padded[1:] = pose_xy_padded[1:] - pose_xy_padded[:-1]
        #pose_xy_vel_padded[pose_xy.shape[0]] = pose_xy_vel_padded[0] # <--- set the velocity at the final timestep to the zero vector
                
        pose_xy_vel = np.zeros(pose_xy.shape)
        pose_xy_vel[1:] = pose_xy[1:] - pose_xy[:-1]
        pose_xy_vel[pose_xy.shape[0]-1] = pose_xy_vel[0]
                
        return_data = {                       
               "xyz":pose_xy,
               "xyz_vel":pose_xy_vel,
               "xyz_confidence":pose_xy_confidence,
               "xyz_vel_confidence":pose_xy_confidence
              }    
              
        return return_data
    
    def get_hand_data(self, sequence_data, frame_data):
    
        #start_frame = frame_data["start_frame"]
        #end_frame   = frame_data["end_frame"]
        #timesteps   = np.arange(start_frame, end_frame, self.time_step_size)
        frames = frame_data["frames"]
        timesteps = frames
        
        # hand xy data in image space
        lhand_xy            = sample_readings(sequence_data, category="person", items=["lhand_2d"], x_name="time", y_name="xy",         timesteps=timesteps, return_dict=False) # [1, t, 21, 2]
        lhand_xy_confidence = sample_readings(sequence_data, category="person", items=["lhand_2d"], x_name="time", y_name="confidence", timesteps=timesteps, return_dict=False) # [1, t, 21]
        rhand_xy            = sample_readings(sequence_data, category="person", items=["rhand_2d"], x_name="time", y_name="xy",         timesteps=timesteps, return_dict=False) # [1, t, 21, 2]
        rhand_xy_confidence = sample_readings(sequence_data, category="person", items=["rhand_2d"], x_name="time", y_name="confidence", timesteps=timesteps, return_dict=False) # [1, t, 21]
        
        lhand_xy = lhand_xy[0]                                                          # [t, num_joints=21, dim=2]
        lhand_xy_confidence = lhand_xy_confidence[0]                                    # [t, num_joints=21]
        lhand_xy_confidence = np.repeat(lhand_xy_confidence[:,:,np.newaxis],2,axis=2)   # [t, num_joints=21, dim=2]
        rhand_xy = rhand_xy[0]                                                          # [t, num_joints=21]
        rhand_xy_confidence = rhand_xy_confidence[0]                                    # [t, num_joints=21, dim=2]
        rhand_xy_confidence = np.repeat(rhand_xy_confidence[:,:,np.newaxis],2,axis=2)   # [t, num_joints=21, dim=2]
                              
        # pad
        #lhand_xy_padded            = pad(lhand_xy, self.pose_padded_length)
        #lhand_xy_confidence_padded = pad(lhand_xy_confidence, self.pose_padded_length)
        #rhand_xy_padded            = pad(rhand_xy, self.pose_padded_length)
        #rhand_xy_confidence_padded = pad(rhand_xy_confidence, self.pose_padded_length)
        
        # compute velocities
        #lhand_xy_vel_padded = np.zeros(lhand_xy_padded.shape)
        #lhand_xy_vel_padded[1:] = lhand_xy_padded[1:] - lhand_xy_padded[:-1]
        #lhand_xy_vel_padded[lhand_xy.shape[0]] = lhand_xy_vel_padded[0] # <--- set the velocity at the final timestep to the zero vector
        lhand_xy_vel = np.zeros(lhand_xy.shape)
        lhand_xy_vel[1:] = lhand_xy[1:] - lhand_xy[:-1]
        lhand_xy_vel[lhand_xy.shape[0]-1] = lhand_xy_vel[0]
        
        #rhand_xy_vel_padded = np.zeros(rhand_xy_padded.shape)
        #rhand_xy_vel_padded[1:] = rhand_xy_padded[1:] - rhand_xy_padded[:-1]
        #rhand_xy_vel_padded[rhand_xy.shape[0]] = rhand_xy_vel_padded[0] # <--- set the velocity at the final timestep to the zero vector
        rhand_xy_vel = np.zeros(rhand_xy.shape)
        rhand_xy_vel[1:] = rhand_xy[1:] - rhand_xy[:-1]
        rhand_xy_vel[rhand_xy.shape[0]-1] = rhand_xy_vel[0]
        
        # follow kit_mocap format
        # [pose_padded_length, hands=2, joints=21, dim=2] -> [pose_padded_length, 2, 42]
        #finger_padded = np.stack((lhand_xy_padded,rhand_xy_padded),axis=1)                              # [t, 2, num_joints=21, dim=2]
        #finger_padded = np.reshape(finger_padded,[finger_padded.shape[0], finger_padded.shape[1], -1])  # [t, 2, num_joints=21* dim=2]
        #finger_vel_padded = np.stack((lhand_xy_vel_padded,rhand_xy_vel_padded),axis=1)
        #finger_vel_padded = np.reshape(finger_vel_padded,[finger_vel_padded.shape[0], finger_vel_padded.shape[1], -1])
        finger = np.stack([lhand_xy,rhand_xy],axis=1)                                       # [t, 2, 21, 2]
        finger = np.reshape(finger,[finger.shape[0],finger.shape[1],-1])                    # [t, 2, 21* 2]
        finger_vel = np.stack([lhand_xy_vel,rhand_xy_vel],axis=1)                           # [t, 2, 21, 2]
        finger_vel = np.reshape(finger_vel,[finger_vel.shape[0],finger_vel.shape[1],-1])    # [t, 2, 21* 2]
        
        # [pose_padded_length, hands=2, joints=21] -> [pose_padded_length, 2, 42]
        #finger_confidence_padded = np.stack((lhand_xy_confidence_padded,rhand_xy_confidence_padded),axis=1)                                                         # [t, 2, num_joints=21, dim=2]
        #finger_confidence_padded = np.reshape(finger_confidence_padded,[finger_confidence_padded.shape[0], finger_confidence_padded.shape[1], -1])                  # [t, 2, num_joints=21* dim=2]
        #finger_vel_confidence_padded = np.stack((lhand_xy_confidence_padded,rhand_xy_confidence_padded),axis=1)                                                     # [t, 2, num_joints=21, dim=2]
        #finger_vel_confidence_padded = np.reshape(finger_vel_confidence_padded,[finger_vel_confidence_padded.shape[0], finger_vel_confidence_padded.shape[1], -1])  # [t, 2, num_joints=21* dim=2]
        finger_confidence = np.stack([lhand_xy_confidence,rhand_xy_confidence],axis=1)
        finger_confidence = np.reshape(finger_confidence,[finger_confidence.shape[0],finger_confidence.shape[1],-1])
        finger_vel_confidence = np.stack([lhand_xy_confidence,rhand_xy_confidence],axis=1)
        finger_vel_confidence = np.reshape(finger_vel_confidence,[finger_vel_confidence.shape[0], finger_vel_confidence.shape[1], -1])  # [t, 2, num_joints=21* dim=2]
                        
        return_data = {
            "finger":finger,
            "finger_vel":finger_vel,
            "finger_confidence":finger_confidence,
            "finger_vel_confidence":finger_vel_confidence
            }
        
        """
        return_data = {
            "lhand_xy":lhand_xy_padded,                             "lhand_xy_unpadded_length":lhand_xy.shape[0],
            "lhand_xy_vel":lhand_xy_vel_padded,                     "lhand_xy_vel_unpadded_length":lhand_xy.shape[0],
            "lhand_xy_confidence":lhand_xy_confidence_padded,       "lhand_xy_confidence_unpadded_length":lhand_xy_confidence.shape[0],
            "lhand_xy_vel_confidence":lhand_xy_confidence_padded,   "lhand_xy_vel_confidence_unpadded_length":lhand_xy_confidence.shape[0],
            
            "rhand_xy":rhand_xy_padded,                             "rhand_xy_unpadded_length":rhand_xy.shape[0],
            "rhand_xy_vel":rhand_xy_vel_padded,                     "rhand_xy_vel_unpadded_length":rhand_xy.shape[0],
            "rhand_xy_confidence":rhand_xy_confidence_padded,       "rhand_xy_confidence_unpadded_length":rhand_xy_confidence.shape[0],
            "rhand_xy_vel_confidence":rhand_xy_confidence_padded,   "rhand_xy_vel_confidence_unpadded_length":rhand_xy_confidence.shape[0]
            }
        """
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
                
        # object meta data
        object_meta = self.get_object_meta(sequence_data)
                
        #start_frame = frame_data["start_frame"]
        #end_frame   = frame_data["end_frame"]
        #timesteps   = np.arange(start_frame, end_frame, self.time_step_size)
        frames = frame_data["frames"]
        timesteps = frames
        
        # # # # # # # # # # # # # # # #
        # get object and distractors  # 
        # # # # # # # # # # # # # # # #
        
        key = self.object_type # all_2d_objects
        
        # object 3d centroid
        object_names = object_meta["obj_names"] #[x for x in sequence_data[key].keys() if "Hand" not in x] # [x for x in sequence_data["metadata"][key] if "Hand" not in x]
        object_bbox  = sample_readings(sequence_data, category=key, items=object_names, x_name="time", y_name="bbox", timesteps=timesteps, return_dict=False) # [n, t, 2, 3] or [1, t, 2, 3]
        object_bbox  = np.expand_dims(object_bbox,0) if len(object_bbox.shape) == 3 else object_bbox # for wipe action when there is only one object that gets squeezed
        object_pos   = np.mean(object_bbox,axis=2,keepdims=True) # [n, t, 1, 3]
                           
        # initial object position
        initial_object_pos = object_pos[:,0:1]
        
        #print(timesteps)
        #print(object_bbox)
        #print("================")
        #print(np.sum(np.abs(object_bbox)))
        #print(initial_object_pos)
        #sys.exit()
                           
        """if self.num_extra_distractors != -1:
            print(object_meta["obj_names"])
            print(object_meta["obj_ids"])
            print(object_meta["obj_ohs"])
            print(object_bbox.shape)
            print(object_pos.shape)
            sys.exit()"""            
        
        # # # # # # #
        # get hands #
        # # # # # # #
        
        # hand centroids
        hand_names = [x for x in sequence_data[key].keys() if "LeftHand" in x] + [x for x in sequence_data[key].keys() if "RightHand" in x]
        hand_bbox  = sample_readings(sequence_data, category=key, items=hand_names, x_name="time", y_name="bbox", timesteps=timesteps, return_dict=False) # [n, t, 3]
        hand_pos   = np.mean(hand_bbox,axis=2,keepdims=True) # [n, t, 1, 3]

        # # # # # #
        # process #
        # # # # # #
        
        # process hand and object
        # - data will not get centered wrt table if in 2D mode
        table_pos           = sequence_data["table_center"]
        object_bbox         = self.process_obj(object_bbox, table_pos=table_pos, pad_object=True,   prefix="obj_bbox")          # print(object_data["rhand_obj_xyz"].shape) # [t, n, 2, 3]
        object_data         = self.process_obj(object_pos,  table_pos=table_pos, pad_object=True,   prefix="obj")               # print(object_data["rhand_obj_xyz"].shape) # [t, n, 1, 3]
        initial_object_data = self.process_obj(initial_object_pos, table_pos=table_pos, pad_object=True, prefix="initial_obj")  # print(hand_data["rhand_wrist_xyz"].shape) # [1, 2, 1, 3]
        
        hand_data                  = self.process_obj(hand_pos,    table_pos=table_pos, pad_object=False,  prefix="wrist")      # print(hand_data["rhand_wrist_xyz"].shape) # [t, 2, 1, 3]
        hand_data["wrist_xyz"]     = np.squeeze(hand_data["wrist_xyz"])                                                         # print(hand_data["rhand_wrist_xyz"].shape) # [t, 2,    3]
        hand_data["wrist_xyz_vel"] = np.squeeze(hand_data["wrist_xyz_vel"])                                                     # print(hand_data["rhand_wrist_xyz"].shape) # [t, 2,    3]
        
        # object adjacency matrix
        obj_adjacency_matrix = 1 - np.eye(self.object_padded_length)
        for i,obj_id in enumerate(object_meta["obj_ids"]):
            # if obj_id is zero, detach it from all
            if obj_id == 0:
                obj_adjacency_matrix = detach(obj_adjacency_matrix, i)
                
        # object and body adjacency matrix
        if hasattr(self,"adjacency_matrix_type"):
            if self.adjacency_matrix_type == "mtgcn":
                
                # object and human adjacency matrix
                adjacency_matrix = np.zeros([self.object_padded_length + 15, self.object_padded_length + 15])
                
                # object adjacency matrix padded
                obj_adjacency_matrix = np.pad(obj_adjacency_matrix,((0,15),(0,15)))
                
                # human adjacency matrix padded
                human_adjacency_matrix = np.pad(var.mocap_adjacency_matrix,((self.object_padded_length,0),(self.object_padded_length,0)))
                
                # object and human adjacency matrix
                adjacency_matrix = obj_adjacency_matrix + human_adjacency_matrix
        
        else:
            adjacency_matrix = np.zeros([self.object_padded_length + 15, self.object_padded_length + 15])
            
        return_data = {**object_meta, **object_data, **hand_data, **object_bbox, **initial_object_data, "adjacency_matrix":adjacency_matrix}
        return return_data

    """
    def get_extra_distractors(self, task, num_extra_distractors):
        distractor_names = list(set(self.all_objects) - set(self.action_to_objects[task]))
        sampled_distractor_names = [random.choice(distractor_names) for _ in range(num_extra_distractors)]
        return sampled_distractor_names
    """
        
    # get object meta
    # - object names
    # - id
    # - one hot vectors
    def get_object_meta(self, sequence_data):
                
        # len(distractor_names) = 3
        # self.num_extra_distractors = 0
        # -(len(distractor_names) - self.num_extra_distractors) = -3
        
        # len(distractor_names) = 3
        # self.num_extra_distractors = 1
        # -(len(distractor_names) - self.num_extra_distractors) = -2
        
        # # # # # # # # # # # # # # # #
        # get object and distractors  # 
        # # # # # # # # # # # # # # # #
        
        key = self.object_type
        
        # object names and ids
        object_names        = [x for x in sequence_data[key].keys() if "Hand" not in x]
        
        # get distractors
        main_objects = self.metadata.loc[(self.metadata["subject"] == sequence_data["metadata"]["subject"]) & (self.metadata["task"] == sequence_data["metadata"]["main_action"]) & (self.metadata["take"] == sequence_data["metadata"]["take"])]["main_objects"]
        main_objects = main_objects.iloc[0]
        distractor_names = []
        for i,object_name in enumerate(object_names):
            if object_name.split("_")[0] not in main_objects.keys():
                distractor_names.append(object_name)
        
        # object_names without distractors
        if self.add_distractors == 0:
            object_names = [x for x in object_names if x not in distractor_names]
        
        # select distractors
        if self.num_extra_distractors != -1:
            distractor_names = distractor_names[:self.num_extra_distractors]
            object_names = object_names + distractor_names
        
        """# remove distractors
        if self.num_extra_distractors != -1:
            #print("object_names", object_names)
            #print("main_objects", main_objects.keys())
            #print("distractor_names before deletion", distractor_names)
            distractor_names = distractor_names[-(len(distractor_names) - self.num_extra_distractors):]
            #print("distractor_names after deleteion", distractor_names)
            object_names = [x for x in object_names if x not in distractor_names]
            #print("filtered object_names", object_names)
            #print()"""
            
        object_ids        = np.array([self.object_name_to_id[object_name.split("_")[0]] for object_name in object_names])
        object_ids_padded = pad(object_ids,self.object_padded_length).astype(int)
        object_ohs_padded = one_hot(object_ids_padded,self.num_obj_wrist_classes)
                
        # # # # # # #
        # get hands # 
        # # # # # # #
        
        ## hand names and ids
        hand_names = [x for x in sequence_data[key].keys() if "LeftHand" in x] + [x for x in sequence_data[key].keys() if "RightHand" in x]
        hand_ids   = np.array([self.object_name_to_id[hand_name.split("_")[0]] for hand_name in hand_names])
        hand_ohs   = one_hot(hand_ids,self.num_obj_wrist_classes)
                
                       # object data
        return_data = {"obj_names":object_names,
                       "obj_ids":object_ids_padded, "obj_ids_unpadded_length":object_ids.shape[0], 
                       "obj_ohs":object_ohs_padded, "obj_ohs_unpadded_length":object_ids.shape[0],
                       
                       # hand data
                       "hand_names":hand_names, "wrist_ids":hand_ids, "wrist_ohs":hand_ohs}
        
        """
        if self.use_edges_for_finger in ["attention"]:
            # get the objects handled by the left and right hand
            main_action = sequence_data["metadata"]["main_action"]
            #print(sequence_data["metadata"]["filename"]) /home_nfs/haziq/datasets/kit_mocap/data-sorted-simpleGT-v3/Cut/files_motions_3021/Cut1_c_0_05cm_01.xml
            lqk = [1 if x in self.held_object[main_action]["left"] else 0 for x in object_names]
            rqk = [1 if x in self.held_object[main_action]["right"] else 0 for x in object_names]
            
            filename = sequence_data["metadata"]["filename"]
            if "Transfer" in filename and any([x in filename for x in ignore_board_sequence]):
                lqk = [1 if x in self.held_object[main_action]["left"] and x != "cutting_board_small" else 0 for x in object_names]
            if "Transfer" in filename and any([x in filename for x in ignore_bowl_sequence]):
                lqk = [1 if x in self.held_object[main_action]["left"] and x != "mixing_bowl_green" else 0 for x in object_names]

            if np.sum(lqk) != 1 or np.sum(rqk) != 1:
                print(sequence_data["metadata"]["filename"])
                print(object_names)
                print("lqk:",lqk)
                print("rqk:",rqk)
                sys.exit()
            lqk = lqk.index(1)
            rqk = rqk.index(1)
            return_data = {**return_data, "lqk":lqk, "rqk":rqk}
        """
                       
        return return_data
        
    # process all objects
    # - subtract by table_pos
    # - scale (do not scale table)
    # - add scaled noise
    # - pad time and number of objects
    def process_obj(self, pos, table_pos, pad_object, prefix):

        # pos = [n, t, 1, 3]
        # table_pos = [3]
                
        # subtract by unscaled table_pos at every timestep and maybe by the reference object
        pos = pos - table_pos if table_pos is not None and "3d" in self.object_type else pos     # [n, t, 1, 3]     
        
        # scale
        pos = pos * self.xyz_scale
        #print("----")
        #print(pos)
        #print("----")
                                        
        # pad time
        #pos_padded = np.stack([pad(x, self.pose_padded_length) for x in pos])   # [n, padded t, num_markers, 3]
        #table_pos_padded = pad(table_pos, self.pose_padded_length)              #    [padded t, num_markers, 3]
        
        pos_padded = np.copy(pos)
        # pad object
        if pad_object:
            pos_padded = pad(pos_padded, self.object_padded_length) # [padded n, t, 3]
        
        # transpose
        pos_padded = np.transpose(pos_padded, (1,0,2,3)) # [t, padded n, num_markers, 3]
        """if pos_padded.shape[2] == 2:
            print(prefix)
            print("ERROR")
            print(pos_padded.shape)
            sys.exit()"""
        
        #pos_vel_padded = np.zeros(pos_padded.shape)
        #pos_vel_padded[1:] = pos_padded[1:] - pos_padded[:-1]
        #pos_vel_padded[pos.shape[1]] = pos_vel_padded[0] # <--- set the velocity at the final timestep to the zero vector
        pos_vel_padded = np.zeros(pos_padded.shape)
        pos_vel_padded[1:] = pos_padded[1:] - pos_padded[:-1]
        pos_vel_padded[pos_padded.shape[0]-1] = pos_vel_padded[0]
        
        # center
        ctr_padded = np.mean(pos_padded,axis=2,keepdims=True)
        ctr_padded = np.squeeze(ctr_padded)
        
        return_data = {                       
                       prefix+"_pos":ctr_padded,             prefix+"_pos_unpadded_objects":pos.shape[0],
                       prefix+"_xyz":pos_padded,             prefix+"_xyz_unpadded_objects":pos.shape[0],       # obj_pos, obj_pos_unpadded_length
                       prefix+"_xyz_vel":pos_vel_padded,     prefix+"_xyz_vel_unpadded_objects":pos.shape[0],   
                       prefix+"_table_pos":table_pos,                                                           # table pos is not at origin
                      }
        return return_data

# # # # # # # # # # # # #
#                       #
# processing functions  #
#                       #
# # # # # # # # # # # # #

def one_hot(labels, max_label=None):

    one_hot_labels = np.zeros((labels.size, labels.max()+1)) if max_label is None else np.zeros((labels.size, max_label))
    one_hot_labels[np.arange(labels.size),labels] = 1
    
    return one_hot_labels
        
"""
def one_hot(labels, max_label=None, return_mask=False):

    one_hot_labels = np.zeros((labels.size, labels.max()+1)) if max_label is None else np.zeros((labels.size, max_label))
    mask           = np.ones((labels.size)) if max_label is None else np.ones((labels.size))
    
    for i in range(one_hot_labels.shape[0]):
        if labels[i] != -1:
            one_hot_labels[i,labels[i]] = 1
        else:
            one_hot_labels[i,0] = 1
            mask[i] = 0
    
    # cannot handle None or custom labels
    #one_hot_labels[np.arange(labels.size),labels] = 1
    
    if return_mask == True:
        return one_hot_labels, mask
    else:
        return one_hot_labels
"""

# get the id of the reference object
# - will be none for approach action
def get_reference_object_name(object_names, action):
    return None

# pad data
def pad(data, pad_length, return_unpadded_length=0):

    #print(data.shape)
    # data must be [t, ...]
    
    unpadded_length = data.shape[0]
    if pad_length < unpadded_length:
        print("pad_length too short !")
        print("Pad Length = ", pad_length)
        print("Unpadded Sequence Length =", unpadded_length)
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
                        
            if t < np.min(time):
                yt_list.append(values[0])
                
            elif t >= np.max(time):
                yt_list.append(values[-1])
                
            else:
                #print("time",time)
                #print("t",t)
                #print("time[time < t]",time[time < t])
                try:
                    # cannot <= and >= else I get the denominator of 0 when I do the interpolation resulting in infinity
                    i1 = np.where(time == time[time <= t].max())[0][0]
                    i2 = np.where(time == time[time > t].min())[0][0]                    
                except:
                    print(time)
                    print("filename=",data["metadata"]["filename"])
                    print("t=",t)
                    print("time.shape=", time.shape)
                    print("values.shape=", values.shape)
                    print("Error here")
                    print(np.where(time == time[time <= t].max())[0][0])
                    print(np.where(time == time[time > t].min())[0][0])
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