import os
import json
import itertools
import numpy as np

# sequences to skip
skip = [
# todo: find out why i skip this
# - must be due to bad ground truths
["subject_1","task_6_w_hard_drive","take_0"],
# temporary skip while i manually fix get-2d-track-data.py
["subject_6","task_4_k_wiping","take_8"],

# bad track data
["subject_1","task_1_k_cooking","take_6"],
["subject_1","task_1_k_cooking","take_9"],
["subject_1","task_3_k_pouring","take_3"],
["subject_1","task_3_k_pouring","take_5"],
["subject_1","task_3_k_pouring","take_8"],
["subject_1","task_3_k_pouring","take_9"],
["subject_1","task_7_w_free_hard_drive","take_0"],

["subject_2","task_2_cooking_with_bowls","take_0"],
["subject_2","task_3_k_pouring","take_2"],
["subject_2","task_3_k_pouring","take_4"],
["subject_2","task_4_k_wiping","take_1"],
["subject_2","task_4_k_wiping","take_6"],
["subject_2","task_6_w_hard_drive","take_2"],
["subject_2","task_6_w_hard_drive","take_5"],
["subject_2","task_8_w_hammering","take_0"],
["subject_2","task_8_w_hammering","take_1"],

["subject_3","task_1_k_cooking","take_1"],
["subject_3","task_1_k_cooking","take_2"],
["subject_3","task_1_k_cooking","take_3"],
["subject_3","task_1_k_cooking","take_4"],
["subject_3","task_1_k_cooking","take_5"],
["subject_3","task_1_k_cooking","take_8"],                
["subject_3","task_2_k_cooking_with_bowls","take_3"],
["subject_3","task_2_k_cooking_with_bowls","take_4"],
["subject_3","task_2_k_cooking_with_bowls","take_5"],
["subject_3","task_2_k_cooking_with_bowls","take_6"],                
["subject_3","task_3_k_pouring","take_0"],
["subject_3","task_3_k_pouring","take_3"],
["subject_3","task_3_k_pouring","take_6"],
["subject_3","task_3_k_pouring","take_8"],
["subject_3","task_3_k_pouring","take_9"],                
["subject_3","task_6_w_hard_drive","take_9"],                
["subject_3","task_7_free_hard_drive","take_0"],
["subject_3","task_7_free_hard_drive","take_3"],
["subject_3","task_7_free_hard_drive","take_4"],
["subject_3","task_7_free_hard_drive","take_6"],
["subject_3","task_7_free_hard_drive","take_7"],
["subject_3","task_7_free_hard_drive","take_9"],                
["subject_3","task_8_w_hammering","take_5"],
["subject_3","task_8_w_hammering","take_6"],

["subject_4","task_1_k_cooking","take_2"],
["subject_4","task_1_k_cooking","take_7"],
["subject_4","task_1_k_cooking","take_8"],                
["subject_4","task_2_k_cooking_with_bowls","take_3"],                
["subject_4","task_3_k_pouring","take_2"],
["subject_4","task_3_k_pouring","take_5"],
["subject_4","task_3_k_pouring","take_6"],
["subject_4","task_3_k_pouring","take_8"],
["subject_4","task_3_k_pouring","take_9"],                
["subject_4","task_6_w_hard_drive","take_4"],                
["subject_4","task_7_w_free_hard_drive","take_2"],
["subject_4","task_7_w_free_hard_drive","take_5"],
["subject_4","task_7_w_free_hard_drive","take_7"],
["subject_4","task_8_w_hammering","take_3"],

["subject_5","task_3_k_pouring","take_4"],
["subject_5","task_3_k_pouring","take_5"],
["subject_5","task_3_k_pouring","take_7"],
["subject_5","task_3_k_pouring","take_9"],
["subject_5","task_6_w_hard_drive","take_0"],
["subject_5","task_6_w_hard_drive","take_4"],
["subject_5","task_6_w_hard_drive","take_7"],
["subject_5","task_7_w_free_hard_drive","take_1"],
["subject_5","task_7_w_free_hard_drive","take_3"],

["subject_6","task_3_k_pouring","take_3"],
["subject_6","task_3_k_pouring","take_6"],
["subject_6","task_6_w_hard_drive","take_2"],
["subject_6","task_6_w_hard_drive","take_8"],
["subject_6","task_7_w_free_hard_drive","take_8"],                
["subject_6","task_8_w_hammering","take_4"],
["subject_6","task_8_w_hammering","take_5"]                
]

# list of all objects
all_objects = ["bottle","whisk","bowl","cup","knife","banana","screwdriver","sponge","cuttingboard","cereals","woodenwedge","saw","hammer","harddrive","cuttingboard"]

# action -> objects
action_to_objects = {}
action_to_objects["task_1_k_cooking"]              = ["LeftHand","RightHand","bottle","bowl","whisk"] + ["banana","knife","cup","cereals"]
action_to_objects["task_2_k_cooking_with_bowls"]   = ["LeftHand","RightHand","bowl","whisk"]
action_to_objects["task_3_k_pouring"]              = ["LeftHand","RightHand","bottle","cup"] + ["whisk","knife","banana","bowl","cuttingboard","cereals","sponge"]
action_to_objects["task_4_k_wiping"]               = ["LeftHand","RightHand","sponge","bottle","bowl","whisk","cup","banana","knife","cuttingboard","cereals"]
action_to_objects["task_5_k_cereals"]              = []
action_to_objects["task_6_w_hard_drive"]           = ["LeftHand","RightHand","harddrive","screwdriver","hammer","saw","woodenwedge"]
action_to_objects["task_7_w_free_hard_drive"]      = ["LeftHand","RightHand","harddrive","screwdriver"]
action_to_objects["task_8_w_hammering"]            = ["LeftHand","RightHand","hammer","woodenwedge"]
action_to_objects["task_9_w_sawing"]               = ["LeftHand","RightHand","saw","woodenwedge","hammer","screwdriver","harddrive"]

# # # # # # # #
# human data  #
# # # # # # # #  

hand_xyz_dims = [5, 14]

# mocap names
mocap_names = ['LEar', 'LElbow', 'LEye', 'LHip', 'LShoulder', 'LWrist', 'MidHip', 'Neck', 'Nose', 'REar', 'RElbow', 'REye', 'RHip', 'RShoulder', 'RWrist']
l_arm_mocap_idxs = [1,5]
r_arm_mocap_idxs = [10,14]

# mocap group names
mocap_group_names = [['LHip','MidHip'],['MidHip','Neck'],['Neck','LShoulder'],['LShoulder','LElbow'],['LElbow','LWrist'],
                     ['RHip','MidHip'],['MidHip','Neck'],['Neck','RShoulder'],['RShoulder','RElbow'],['RElbow','RWrist'],
                     ['Neck','Nose'],['Nose','LEye'],['LEye','LEar'],
                     ['Neck','Nose'],['Nose','REye'],['REye','REar']]
              
mocap_group_idxs = []
for x in mocap_group_names:
    idxs = [mocap_names.index(y) for y in x]
    #print(x)
    #print(idxs)
    #input()
    mocap_group_idxs.append(idxs)

mocap_adjacency_matrix = np.zeros([len(mocap_names),len(mocap_names)])
for x,y,z in zip(mocap_group_idxs,mocap_group_names,mocap_group_idxs):
    combinations = list(itertools.combinations(x,2))
    #print(y)
    #print(z)
    #print(combinations)
    #input()
    for c in combinations:
        if c[0] == c[1]:
            print("Error in kit_mocap_variables mocap_adjacency_matrix")
            sys.exit()
        mocap_adjacency_matrix[c[0],c[1]] = 1
        mocap_adjacency_matrix[c[1],c[0]] = 1
        
# body
num_body_joints = 15
body_dim        = 2

# object
num_obj_markers = 1
obj_dim         = 3

# hands
num_hands       = 2
hand_dim        = 42