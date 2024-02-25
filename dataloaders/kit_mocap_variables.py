import os
import json
import itertools
import numpy as np

from utils import one_hot

"""

mocap and joint marker names
xyz id = 3x, 3x+1, 3x+2

"""

# mocap names
mocap_names = ["C7", "CLAV", "L3", # final x = 2
               # 3      4       5       6       7       8       9       10      11      12      13      14      15      16      17      18      19      20      21      22      23      24      25      26
               "LAEL", "LANK", "LAOL", "LASI", "LBAK", "LBHD", "LFHD", "LFRA", "LHEE", "LHIP", "LHPS", "LHTS", "LIFD", "LKNE", "LMT1", "LMT5", "LPSI", "LSHO", "LTHI", "LTIP", "LTOE", "LUPA", "LWPS", "LWTS", # final x = 26
               # 27     28      29      30      31      32      33      34      35      36
               "RAEL", "RANK", "RAOL", "RASI", "RBAK", "RBHD", "RFHD", "RFRA", "RHEE", "RHIP", "RHPS", "RHTS", "RIFD", "RKNE", "RMT1", "RMT5", "RPSI", "RSHO", "RTHI", "RTIP", "RTOE", "RUPA", "RWPS", "RWTS", # final x = 50
               "STRN","T10"]       # final x = 52   
               
# arm
l_arm_mocap_names = ["LUPA", "LAEL", "LAOL", "LFRA", "LHPS", "LHTS", "LIFD", "LWPS", "LWTS"]
r_arm_mocap_names = ["RUPA", "RAEL", "RAOL", "RFRA", "RHPS", "RHTS", "RIFD", "RWPS", "RWTS"]
l_arm_mocap_idxs  = [mocap_names.index(l_arm_mocap_name) for l_arm_mocap_name in l_arm_mocap_names]
r_arm_mocap_idxs  = [mocap_names.index(r_arm_mocap_name) for r_arm_mocap_name in r_arm_mocap_names]

# hand
l_hand_mocap_names = ["LHPS", "LHTS", "LIFD", "LWPS", "LWTS"]
r_hand_mocap_names = ["LHPS", "LHTS", "LIFD", "LWPS", "LWTS"]
l_hand_mocap_idxs  = [mocap_names.index(l_hand_mocap_name) for l_hand_mocap_name in l_hand_mocap_names]
r_hand_mocap_idxs  = [mocap_names.index(r_hand_mocap_name) for r_hand_mocap_name in r_hand_mocap_names]

hand_xyz_dims = [13, 14, 15, 25, 26, 37, 38, 39, 49, 50]

l_feet_mocap_names = ["LTOE", "LMT1", "LMT5", "LANK", "LHEE"]
r_feet_mocap_names = ["RTOE", "RMT1", "RMT5", "RANK", "RHEE"]
l_feet_mocap_idxs  = [mocap_names.index(l_feet_mocap_name) for l_feet_mocap_name in l_feet_mocap_names]
r_feet_mocap_idxs  = [mocap_names.index(r_feet_mocap_name) for r_feet_mocap_name in r_feet_mocap_names]

l_leg_mocap_names = ["LTOE", "LMT1", "LMT5", "LANK", "LHEE", "LTIP", "LKNE", "LTHI", "LHIP", "LASI"]
r_leg_mocap_names = ["RTOE", "RMT1", "RMT5", "RANK", "RHEE", "RTIP", "RKNE", "RTHI", "RHIP", "RASI"]
l_leg_mocap_idxs  = [mocap_names.index(l_leg_mocap_name) for l_leg_mocap_name in l_leg_mocap_names]
r_leg_mocap_idxs  = [mocap_names.index(r_leg_mocap_name) for r_leg_mocap_name in r_leg_mocap_names]

                    # front body + shoulders                   # back body
body_mocap_names = ["RSHO","LSHO","CLAV","STRN","RASI","LASI", "C7","LBAK","RBAK","T10","L3","LPSI","RPSI"]
body_mocap_idxs  = [mocap_names.index(x) for x in body_mocap_names]

hip_mocap_names = ["LASI","LPSI","RASI","RPSI"]
hip_mocap_idxs  = [mocap_names.index(x) for x in hip_mocap_names]

"""

mocap adjacency matrix

"""

# mocap group names
mocap_group_names = [
# # # # # # #
# left leg  #
# # # # # # # 

# left foot to left shin
["LTOE","LMT1","LMT5","LHEE","LANK",
"LTIP"],

# left knee to left shin
["LTIP",
"LKNE"],

# left knee to left thigh
["LKNE",
"LTHI"],

# left thigh to left hip
["LTHI",
"LHIP","LASI","LPSI"],

# # # # # # #
# right leg #
# # # # # # # 

# right foot to right shin
["RTOE","RMT1","RMT5","RHEE","RANK",
"RTIP"],

# right knee to right shin
["RTIP",
"RKNE"],

# right knee to right thigh
["RKNE",
"RTHI"],

# right thigh to right hip
["RTHI",
"RHIP","RASI","RPSI"],

# # # # # # # # #
# spinal column #
# # # # # # # # #

# bottom of spine to left and right hip
["L3","T10","STRN",
"RHIP","RASI","RPSI",
"LHIP","LASI","LPSI"],

# spinal column
["CLAV","C7","STRN","T10","L3"],

# head to neck
["RFHD","LFHD","LBHD","RBHD",
"CLAV","C7"],

# # # # # # #
# left arm  #
# # # # # # #

# left wrist to left forearm
["LHPS","LHTS","LWTS","LWPS",
"LFRA"],

# left forearm to left elbow
["LFRA",
"LAEL","LAOL"],

# left elbow to left upper arm
["LAEL","LAOL",
"LUPA"],

# left upper arm to left shoulder
["LUPA",
"LSHO"],

# left shoulder to neck
["LSHO",
"CLAV","C7"],

# # # # # # #
# right arm #
# # # # # # #

# right wrist to right forearm
["RHPS","RHTS","RWTS","RWPS",
"RFRA"],

# right forearm to right elbow
["RFRA",
"RAEL","RAOL"],

# right elbow to right upper arm
["RAEL","RAOL",
"RUPA"],

# right upper arm to right shoulder
["RUPA",
"RSHO"],

# right shoulder to neck
["RSHO",
"CLAV","C7"]]

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

"""

joint data

"""

# joint names
joint_names = ["BLNx_joint", "BLNy_joint", "BLNz_joint", # lower neck        0,1,2
              "BPx_joint", "BPy_joint", "BPz_joint",     # pelvis            3,4,5
              "BTx_joint", "BTy_joint", "BTz_joint",     # thorax            6,7,8
              "BUNx_joint", "BUNy_joint", "BUNz_joint",  # upper neck        9,10,11
               
              "LAx_joint", "LAy_joint", "LAz_joint",     # left ankle        12,13,14
              "LEx_joint", "LEz_joint",                  # left elbow        15,16,
              "LHx_joint", "LHy_joint", "LHz_joint",     # left hip          17,18,19
              "LKx_joint",                               # left knee         20
              "LSx_joint", "LSy_joint", "LSz_joint",     # left shoulder     21,22,23
              "LWx_joint", "LWy_joint",                  # left wrist        24,25
              "LFx_joint",                               # left foot         26
              "LMrot_joint",                             # left metatarsal   27          (between foot and ankle)
              
              "RAx_joint", "RAy_joint", "RAz_joint",     # right ankle       28,29,30
              "REx_joint", "REz_joint",                  # right elbow       31,32
              "RHx_joint", "RHy_joint", "RHz_joint",     # right hip         33,34,35
              "RKx_joint",                               # right knee        36
              "RSx_joint", "RSy_joint", "RSz_joint",     # right shoulder    37,38,39
              "RWx_joint", "RWy_joint",                  # right wrist       40,41,
              "RFx_joint",                               # right foot        42,
              "RMrot_joint"]                             # right metatarsal  43          (between foot and ankle)
joint_idx_to_name = {i:joint_name for i,joint_name in enumerate(joint_names)}

# extended joint names with the root and column segment
root_name   = ["ROOTx_joint", "ROOTy_joint", "ROOTz_joint"]
CS_name     = ["CSx_joint",   "CSy_joint",   "CSz_joint"]
extended_joint_names     = joint_names + root_name + CS_name
extended_joint_idx_to_name = {i:joint_name for i,joint_name in enumerate(extended_joint_names)}

# left and right arm joint names and idxs
l_arm_joint_names = ["LSx_joint", "LSy_joint", "LSz_joint","LEx_joint", "LEz_joint","LWx_joint", "LWy_joint"]
r_arm_joint_names = ["RSx_joint", "RSy_joint", "RSz_joint","REx_joint", "REz_joint","RWx_joint", "RWy_joint"]
l_arm_joint_idxs  = [extended_joint_names.index(l_arm_joint_name) for l_arm_joint_name in l_arm_joint_names]
r_arm_joint_idxs  = [extended_joint_names.index(r_arm_joint_name) for r_arm_joint_name in r_arm_joint_names]

# the axis of rotation for each joint
extended_joint_axis = {}
for joint_name in extended_joint_names:
    # each joint_name must be hardcoded to use the x,y,z for only the axis of rotation
    if joint_name.count("x") > 1 or joint_name.count("y") > 1 or joint_name.count("z") > 1:
        print("Error in joint_name.count()")
        sys.exit()
    # each joint_name must be hardcoded to use the x,y,z for the axis of rotation
    if joint_name.count("x") == 0 and joint_name.count("y") == 0 and joint_name.count("z") == 0:
        if joint_name.count("rot") == 1:
            extended_joint_axis[joint_name] = "y"
        else:
            print("Error in joint_name.count()")
            sys.exit()
    if joint_name.count("x") == 1:
        extended_joint_axis[joint_name] = "x"
    if joint_name.count("y") == 1:
        extended_joint_axis[joint_name] = "y"
    if joint_name.count("z") == 1:
        extended_joint_axis[joint_name] = "z"

# the axis of rotation for each joint idx
extended_joint_idx_axis = {extended_joint_names.index(k):v for k,v in extended_joint_axis.items()}

"""
link order
- link order as described in
  https://ieeexplore.ieee.org/document/7506114 
  Unifying Representations and Large-Scale Whole-Body Motion Databases for Studying Human Motion
- order x,y,z
"""

# complete link orderto generate the skeleton
link_order   = [["ROOTx_joint","ROOTy_joint"],["ROOTy_joint","ROOTz_joint"],
               # spinal column
               # 1) BP
               ["ROOTz_joint","BPx_joint"], ["BPx_joint","BPy_joint"], ["BPy_joint","BPz_joint"],
               # 2) BT
               ["BPz_joint","BTx_joint"],   ["BTx_joint","BTy_joint"], ["BTy_joint","BTz_joint"],
               # 3) CS
               ["BTz_joint","CSx_joint"],   ["CSx_joint","CSy_joint"], ["CSy_joint","CSz_joint"],
               # 3) BLN
               ["CSz_joint","BLNx_joint"],  ["BLNx_joint","BLNy_joint"], ["BLNy_joint","BLNz_joint"],
               # 3) BUN
               ["BLNz_joint","BUNx_joint"],  ["BUNx_joint","BUNy_joint"], ["BUNy_joint","BUNz_joint"],
               
               # top left half
               # 1) LS
               ["CSz_joint","LSx_joint"],   ["LSx_joint","LSy_joint"], ["LSy_joint","LSz_joint"],
               # 2) LE
               ["LSz_joint","LEx_joint"],   ["LEx_joint","LEz_joint"],
               # 3) LW
               ["LEz_joint","LWx_joint"],   ["LWx_joint","LWy_joint"],
                
               # top right half
               # 1) RS
               ["CSz_joint","RSx_joint"],   ["RSx_joint","RSy_joint"], ["RSy_joint","RSz_joint"],
               # 2) RE
               ["RSz_joint","REx_joint"],   ["REx_joint","REz_joint"],
               # 3) RW
               ["REz_joint","RWx_joint"],   ["RWx_joint","RWy_joint"],
 
               # bottom left half
               # 1) LH
               ["ROOTz_joint","LHx_joint"], ["LHx_joint","LHy_joint"], ["LHy_joint","LHz_joint"],
               # 2) LK
               ["LHz_joint","LKx_joint"],
               # 3) LA
               ["LKx_joint","LAx_joint"],   ["LAx_joint","LAy_joint"], ["LAy_joint","LAz_joint"],

               # bottom right half
               # 1) RH
               ["ROOTz_joint","RHx_joint"], ["RHx_joint","RHy_joint"], ["RHy_joint","RHz_joint"],
               # 2) RK
               ["RHz_joint","RKx_joint"],
               # 3) RA
               ["RKx_joint","RAx_joint"],   ["RAx_joint","RAy_joint"], ["RAy_joint","RAz_joint"]]

# link order from root to left wrist
link_order_to_left_wrist  = [["ROOTx_joint","ROOTy_joint"],["ROOTy_joint","ROOTz_joint"],
                            # 1) LS
                            ["CSz_joint","LSx_joint"],   ["LSx_joint","LSy_joint"], ["LSy_joint","LSz_joint"],
                            # 2) LE
                            ["LSz_joint","LEx_joint"],   ["LEx_joint","LEz_joint"],
                            # 3) LW
                            ["LEz_joint","LWx_joint"],   ["LWx_joint","LWy_joint"]]
# link order from root to right wrist
link_order_to_right_wrist = [["ROOTx_joint","ROOTy_joint"],["ROOTy_joint","ROOTz_joint"],
                            # 1) RS
                            ["CSz_joint","RSx_joint"],   ["RSx_joint","RSy_joint"], ["RSy_joint","RSz_joint"],
                            # 2) RE
                            ["RSz_joint","REx_joint"],   ["REx_joint","REz_joint"],
                            # 3) RW
                            ["REz_joint","RWx_joint"],   ["RWx_joint","RWy_joint"]]

# idx version
link_idx_order_to_left_wrist = []
for link in link_order_to_left_wrist:    
    parent_name = link[0]
    child_name  = link[1]    
    parent_idx = extended_joint_names.index(parent_name)
    child_idx  = extended_joint_names.index(child_name)
    link_idx_order_to_left_wrist.append([parent_idx,child_idx])
link_idx_order_to_right_wrist = []
for link in link_order_to_right_wrist:    
    parent_name = link[0]
    child_name  = link[1]    
    parent_idx = extended_joint_names.index(parent_name)
    child_idx  = extended_joint_names.index(child_name)
    link_idx_order_to_right_wrist.append([parent_idx,child_idx])

"""
parent_dict
- key = child
- value = parent
- order x,y,z
"""

               # ROOT
parent_dict = {"ROOTx_joint":"ROOTx_joint", "ROOTy_joint":"ROOTx_joint", "ROOTz_joint":"ROOTy_joint", 

               # spinal column
               # 1) BP
               "BPx_joint":"ROOTz_joint", "BPy_joint":"BPx_joint", "BPz_joint":"BPy_joint",
               # 2) BT
               "BTx_joint":"BPz_joint",   "BTy_joint":"BTx_joint", "BTz_joint":"BTy_joint",
               # 3) CS
               "CSx_joint":"BTz_joint",   "CSy_joint":"CSx_joint", "CSz_joint":"CSy_joint",
               # 3) BLN
               "BLNx_joint":"CSz_joint",   "BLNy_joint":"BLNx_joint", "BLNz_joint":"BLNy_joint",
               # 3) BUN
               "BUNx_joint":"BLNz_joint",   "BUNy_joint":"BUNx_joint", "BUNz_joint":"BUNy_joint",
               
               # top left half
               # 1) LS
               "LSx_joint":"CSz_joint",   "LSy_joint":"LSx_joint", "LSz_joint":"LSy_joint",
               # 2) LE
               "LEx_joint":"LSz_joint",   "LEz_joint":"LEx_joint", 
               # 3) LW
               "LWx_joint":"LEz_joint",   "LWy_joint":"LWx_joint",
               
               # top right half
               # 1) RS
               "RSx_joint":"CSz_joint", "RSy_joint":"RSx_joint", "RSz_joint":"RSy_joint",
               # 2) RE
               "REx_joint":"RSz_joint", "REz_joint":"REx_joint", "RWx_joint":"REz_joint",
               # 3) RW
               "RWy_joint":"RWx_joint", 
               
               # bottom left half
               # 1) LH
               "LHx_joint":"ROOTz_joint", "LHy_joint":"LHx_joint", "LHz_joint":"LHy_joint",
               # 2) LK
               "LKx_joint":"LHz_joint",
               # 3) LA
               "LAx_joint":"LKx_joint", "LAy_joint":"LAx_joint", "LAz_joint":"LAy_joint",
               
               # bottom right half
               # 1) RH
               "RHx_joint":"ROOTz_joint", "RHy_joint":"RHx_joint", "RHz_joint":"RHy_joint",
               # 2) RK
               "RKx_joint":"RHz_joint",
               # 3) RA
               "RAx_joint":"RKx_joint", "RAy_joint":"RAx_joint", "RAz_joint":"RAy_joint"}
extended_parent_idx_dict = {extended_joint_names.index(k):extended_joint_names.index(v) for k,v in parent_dict.items()}

"""
link direction
- link direction as described in
  https://ieeexplore.ieee.org/document/7506114 
  Unifying Representations and Large-Scale Whole-Body Motion Databases for Studying Human Motion
"""

link_direction_dict = {}

# root
link_direction_dict[("ROOTx_joint","ROOTy_joint")] = [0,0,0]
link_direction_dict[("ROOTy_joint","ROOTz_joint")] = [0,0,0]

# spinal column
# 1) ROOT to BP
link_direction_dict[("ROOTz_joint","BPx_joint")] = [0,0,1]
link_direction_dict[("BPx_joint","BPy_joint")]   = [0,0,0]
link_direction_dict[("BPy_joint","BPz_joint")]   = [0,0,0]
# 1) BP to BT
link_direction_dict[("BPz_joint","BTx_joint")]   = [0,0,1]
link_direction_dict[("BTx_joint","BTy_joint")]   = [0,0,0]
link_direction_dict[("BTy_joint","BTz_joint")]   = [0,0,0]
# 1) BT to CS
link_direction_dict[("BTz_joint","CSx_joint")]   = [0,0,1]
link_direction_dict[("CSx_joint","CSy_joint")]   = [0,0,0]
link_direction_dict[("CSy_joint","CSz_joint")]   = [0,0,0]
# 1) CS to BLN
link_direction_dict[("CSz_joint","BLNx_joint")]  = [0,0,1]
link_direction_dict[("BLNx_joint","BLNy_joint")] = [0,0,0]
link_direction_dict[("BLNy_joint","BLNz_joint")] = [0,0,0]
# 1) BLN to BUN
link_direction_dict[("BLNz_joint","BUNx_joint")] = [0,0,1]
link_direction_dict[("BUNx_joint","BUNy_joint")] = [0,0,0]
link_direction_dict[("BUNy_joint","BUNz_joint")] = [0,0,0]

# top left half
# 1) CS to LS
link_direction_dict[("CSz_joint","LSx_joint")]   = [-1,0,0]
link_direction_dict[("LSx_joint","LSy_joint")]   = [0,0,0]
link_direction_dict[("LSy_joint","LSz_joint")]   = [0,0,0]
# 1) LS to LE
link_direction_dict[("LSz_joint","LEx_joint")]   = [0,0,-1]
link_direction_dict[("LEx_joint","LEz_joint")]   = [0,0,0]
# 1) LE to LW
link_direction_dict[("LEz_joint","LWx_joint")]   = [0,0,-1]
link_direction_dict[("LWx_joint","LWy_joint")]   = [0,0,0]

# top right half
# 1) CS to RS
link_direction_dict[("CSz_joint","RSx_joint")]   = [1,0,0]
link_direction_dict[("RSx_joint","RSy_joint")]   = [0,0,0]
link_direction_dict[("RSy_joint","RSz_joint")]   = [0,0,0]
# 1) RS to RE
link_direction_dict[("RSz_joint","REx_joint")]   = [0,0,-1]
link_direction_dict[("REx_joint","REz_joint")]   = [0,0,0]
# 1) RE to RW
link_direction_dict[("REz_joint","RWx_joint")]   = [0,0,-1]
link_direction_dict[("RWx_joint","RWy_joint")]   = [0,0,0]

# bottom left half
# 1) ROOT to LH
link_direction_dict[("ROOTz_joint","LHx_joint")] = [-1,0,0]
link_direction_dict[("LHx_joint","LHy_joint")]   = [0,0,0]
link_direction_dict[("LHy_joint","LHz_joint")]   = [0,0,0]
# 1) LH to LK
link_direction_dict[("LHz_joint","LKx_joint")]   = [0,0,-1]
# 1) LK to LA
link_direction_dict[("LKx_joint","LAx_joint")]   = [0,0,-1]
link_direction_dict[("LAx_joint","LAy_joint")]   = [0,0,0]
link_direction_dict[("LAy_joint","LAz_joint")]   = [0,0,0]

# bottom right half
# 1) ROOT to LH
link_direction_dict[("ROOTz_joint","RHx_joint")] = [1,0,0]
link_direction_dict[("RHx_joint","RHy_joint")]   = [0,0,0]
link_direction_dict[("RHy_joint","RHz_joint")]   = [0,0,0]
# 1) LH to LK
link_direction_dict[("RHz_joint","RKx_joint")]   = [0,0,-1]
# 1) LK to LA
link_direction_dict[("RKx_joint","RAx_joint")]   = [0,0,-1]
link_direction_dict[("RAx_joint","RAy_joint")]   = [0,0,0]
link_direction_dict[("RAy_joint","RAz_joint")]   = [0,0,0]

for k,v in link_direction_dict.items():
    link_direction_dict[k] = np.array(v)
    
# idx version
#link_idx_direction_dict = {(extended_joint_names.index(k[0]),extended_joint_names.index(k[1])):v for k,v in link_direction_dict.items()}
    
"""
link length
- link length as described in
  https://ieeexplore.ieee.org/document/7506114 
  Unifying Representations and Large-Scale Whole-Body Motion Databases for Studying Human Motion
"""

link_length_dict = {}

# root
link_length_dict[("ROOTx_joint","ROOTy_joint")] = 0
link_length_dict[("ROOTy_joint","ROOTz_joint")] = 0

# spinal column
# 1) ROOT to BP
link_length_dict[("ROOTz_joint","BPx_joint")] = 40
link_length_dict[("BPx_joint","BPy_joint")]   = 0
link_length_dict[("BPy_joint","BPz_joint")]   = 0
# 1) BP to BT
link_length_dict[("BPz_joint","BTx_joint")]   = 60
link_length_dict[("BTx_joint","BTy_joint")]   = 0
link_length_dict[("BTy_joint","BTz_joint")]   = 0
# 1) BT to CS
link_length_dict[("BTz_joint","CSx_joint")]   = 188
link_length_dict[("CSx_joint","CSy_joint")]   = 0
link_length_dict[("CSy_joint","CSz_joint")]   = 0
# 1) CS to BLN
link_length_dict[("CSz_joint","BLNx_joint")]  = 22
link_length_dict[("BLNx_joint","BLNy_joint")] = 0
link_length_dict[("BLNy_joint","BLNz_joint")] = 0
# 1) BLN to BUN
link_length_dict[("BLNz_joint","BUNx_joint")] = 30
link_length_dict[("BUNx_joint","BUNy_joint")] = 0
link_length_dict[("BUNy_joint","BUNz_joint")] = 0

# top left half
# 1) CS to LS
link_length_dict[("CSz_joint","LSx_joint")]   = 110
link_length_dict[("LSx_joint","LSy_joint")]   = 0
link_length_dict[("LSy_joint","LSz_joint")]   = 0
# 1) LS to LE
link_length_dict[("LSz_joint","LEx_joint")]   = 188
link_length_dict[("LEx_joint","LEz_joint")]   = 0
# 1) LE to LW
link_length_dict[("LEz_joint","LWx_joint")]   = 145
link_length_dict[("LWx_joint","LWy_joint")]   = 0

# top right half
# 1) CS to RS
link_length_dict[("CSz_joint","RSx_joint")]   = 110
link_length_dict[("RSx_joint","RSy_joint")]   = 0
link_length_dict[("RSy_joint","RSz_joint")]   = 0
# 1) RS to RE
link_length_dict[("RSz_joint","REx_joint")]   = 188
link_length_dict[("REx_joint","REz_joint")]   = 0
# 1) RE to RW
link_length_dict[("REz_joint","RWx_joint")]   = 145
link_length_dict[("RWx_joint","RWy_joint")]   = 0

# bottom left half
# 1) ROOT to LH
link_length_dict[("ROOTz_joint","LHx_joint")] = 52
link_length_dict[("LHx_joint","LHy_joint")]   = 0
link_length_dict[("LHy_joint","LHz_joint")]   = 0
# 1) LH to LK
link_length_dict[("LHz_joint","LKx_joint")]   = 245
# 1) LK to LA
link_length_dict[("LKx_joint","LAx_joint")]   = 246
link_length_dict[("LAx_joint","LAy_joint")]   = 0
link_length_dict[("LAy_joint","LAz_joint")]   = 0

# bottom right half
# 1) ROOT to LH
link_length_dict[("ROOTz_joint","RHx_joint")] = 52
link_length_dict[("RHx_joint","RHy_joint")]   = 0
link_length_dict[("RHy_joint","RHz_joint")]   = 0
# 1) LH to LK
link_length_dict[("RHz_joint","RKx_joint")]   = 245
# 1) LK to LA
link_length_dict[("RKx_joint","RAx_joint")]   = 246
link_length_dict[("RAx_joint","RAy_joint")]   = 0
link_length_dict[("RAy_joint","RAz_joint")]   = 0

# idx version
link_idx_length_dict = {(extended_joint_names.index(k[0]),extended_joint_names.index(k[1])):v for k,v in link_length_dict.items()}

"""

object mocap names
- as defined in the xml files

"""

object_mocap_names = {"apple_juice":["aj_01","aj_02","aj_03","aj_04"],
                      "apple_juice_lid":["aj_lid1","aj_lid2","aj_lid3","aj_lid4"],
                      "broom":["broom1","broom2","broom3","broom4"],
                      "cucumber_attachment":["ca_01","ca_02","ca_03","ca_04"],
                      "cup_large":["cl_01","cl_02","cl_03","cl_04"],
                      "cup_small":["sc_01","sc_02","sc_03","sc_04"],
                      "cutting_board_small":["cbs_01","cbs_02","cbs_03","cbs_04"],
                      "draining_rack":["draining_rack_01","draining_rack_02","draining_rack_03","draining_rack_04"],
                      "egg_whisk":["ew_01","ew_02","ew_03","ew_04"],
                      "knife_black":["knife_black_01","knife_black_02","knife_black_03","knife_black_04"],
                      "ladle":["ladle_01","ladle_02","ladle_03","ladle_04"],
                      "milk_small":["milk_small_01","milk_small_02","milk_small_03","milk_small_04"],
                      "milk_small_lid":["ms_lid1","ms_lid2","ms_lid3","ms_lid4"],
                      "mixing_bowl_green":["mbg_01","mbg_02","mbg_03","mbg_04"],
                      "mixing_bowl_small":["mbs_01","mbs_02","mbs_03","mbs_04"],
                      "peeler":["peeler1","peeler2","peeler3","peeler4"],
                      "plate_dish":["plate_01","plate_02","plate_03","plate_04"],
                      "rolling_pin":["rolling_pin1","rolling_pin2","rolling_pin3","rolling_pin4"],
                      "salad_fork":["salad_fork1","salad_fork2","salad_fork3","salad_fork4"],
                      "salad_spoon":["salad_spoon1","salad_spoon2","salad_spoon3","salad_spoon4"],
                      "sponge_small":["sponge_dry_01","sponge_dry_02","sponge_dry_03","sponge_dry_04"],
                      "tablespoon":["tablespoon_01","tablespoon_02","tablespoon_03","tablespoon_04"],
                      "kitchen_sideboard":["ks_01","ks_02","ks_03","ks_04"],
                      "kitchen_sideboard_long":["ksl_01","ksl_02","ksl_03","ksl_04"],
                      }

"""
object mocap markers
- coordinates or object mocap markers after de-rotating and de-centering wrt its root_position
- load from the same json file no matter what subset since the mocap markers are computed wrt to each sequence
"""

object_mocap_markers = None
object_mocap_marker_path = os.path.join(os.path.expanduser("~"),"datasets","kit_mocap","cached-data","obj-marker-pos.json")
if os.path.isfile(object_mocap_marker_path):
    with open(object_mocap_marker_path,"r") as fp:
        object_mocap_markers = json.load(fp)
    for filename,_ in object_mocap_markers.items():
        for k,v in object_mocap_markers[filename].items():
            object_mocap_markers[filename][k]["xyz"] = np.array(object_mocap_markers[filename][k]["xyz"])
            object_mocap_markers[filename][k]["var"] = np.array(object_mocap_markers[filename][k]["var"])

"""

object names
- names of all relevant objects
- this variable prevents the code from loading the human pose or kinect cameras into the object dictionary

"""

all_objects =  ["apple_juice","apple_juice_lid", #0,1
                 "broom", #15
                 "cucumber","cucumber_attachment","cup_large","cup_small","cutting_board_small", #2,2,3,3,4
                 "draining_rack", #5
                 "egg_whisk", #6
                 "knife_black", #7
                 "ladle", #16
                 "milk_small","milk_small_lid","mixing_bowl_green","mixing_bowl_small", #0,1,8,8
                 "plate_dish","peeler", #9,10
                 "rolling_pin", #11
                 "salad_fork","salad_spoon","sponge_small", #12, 13, 14
                 "tablespoon",
                # extras
                 "kitchen_sideboard",
                 "kitchen_sideboard_long"
                ]
                
"""
action to objects
- the list of objects present in each action
- helps create the one-hot label as I need to know the total objects present in order to create the one-hot mapping
- this can be also be rephrased as folder to objects
- although it is not a good variable to be honest (not really)
"""

# action -> objects
action_to_objects = {}
action_to_objects["Close"]     = ["apple_juice","apple_juice_lid","milk_small","milk_small_lid"]
action_to_objects["Cut"]       = ["cucumber_attachment","cutting_board_small","knife_black"]
action_to_objects["Mix"]       = ["mixing_bowl_green","salad_fork","salad_spoon"]
action_to_objects["Open"]      = ["apple_juice","apple_juice_lid","milk_small","milk_small_lid"]
action_to_objects["Peel"]      = ["cucumber_attachment","cutting_board_small","peeler","mixing_bowl_green"]
action_to_objects["Pour"]      = ["apple_juice","milk_small","cup_small","cup_large"]
action_to_objects["RollOut"]   = ["rolling_pin"]
action_to_objects["Scoop"]     = ["mixing_bowl_green","cup_small","cup_large","ladle","tablespoon","plate_dish","salad_fork","salad_spoon"]
action_to_objects["Stir"]      = ["cup_large","tablespoon","egg_whisk","mixing_bowl_green","mixing_bowl_small"]
action_to_objects["Transfer"]  = ["cutting_board_small","knife_black","mixing_bowl_green"]
action_to_objects["Wipe"]      = ["draining_rack","plate_dish","sponge_small","mixing_bowl_green","cutting_board_small"]
action_to_objects["Shake"]     = []
action_to_objects["Open"]      = ["milk_small_lid"]
action_to_objects["Regrasp"]   = []
action_to_objects["Sweep"]     = ["broom"]

# body
num_body_joints = 53
body_dim        = 3

# object
num_obj_markers = 4
obj_dim         = 3

# hands
num_hands       = 2
hand_dim        = 19

"""
object dimensions
"""

obj_dimensions = {}
obj_dimensions["kitchen_sideboard/kitchen_sideboard"]       = np.array([139.99820000000005, 59.9992, 87.99890000000002])
obj_dimensions["cucumber_attachment/cucumber_cut"]          = np.array([4.499999999999995, 24.99970000000001, 4.4999999999999885])
obj_dimensions["cucumber_attachment/cucumber_peel"]         = np.array([4.499999999999998, 4.499999999999998, 33.99960000000001])
obj_dimensions["mixing_bowl_green/mixing_bowl_green"]       = np.array([28.389, 17.3214, 28.38910000000001])
obj_dimensions["mixing_bowl_small/mixing_bowl_small"]       = np.array([20.0, 20.0, 20.0])
obj_dimensions["cup_small/cup_small"]                       = np.array([7.984599999999999, 9.000400000000015, 7.983400000000004])
obj_dimensions["cup_large/cup_large"]                       = np.array([8.600399999999997, 13.000600000000015, 8.596799999999996])
obj_dimensions["plate_dish/plate_dish"]                     = np.array([27.599600000000006, 1.8000000000000143, 27.599700000000006])
obj_dimensions["cutting_board_small/cutting_board_small"]   = np.array([14.918399999999998, 1.7883000000000042, 30.388199999999998])

"""
reference object given target
- default reference (if it can't be found) will always be the kitchen sideboard!
"""

# fine motion wrt cucumber_attachment

tgt_ref_dict = {}

tgt_ref_dict["Cut"] = {}
tgt_ref_dict["Cut"]["set_1"] = {}
tgt_ref_dict["Cut"]["set_1"]["Idle"] = {}
tgt_ref_dict["Cut"]["set_1"]["Approach"] = {}
tgt_ref_dict["Cut"]["set_1"]["Move"] = {}
tgt_ref_dict["Cut"]["set_1"]["Cut"] = {}
tgt_ref_dict["Cut"]["set_1"]["Hold"] = {}
tgt_ref_dict["Cut"]["set_1"]["Place"] = {}
tgt_ref_dict["Cut"]["set_1"]["Retreat"] = {}
tgt_ref_dict["Cut"]["set_1"]["Cut"]["knife_black"] = "cucumber_attachment"

tgt_ref_dict["Cut_files_motions_3021_Cut1_c_0_05cm_01"] = tgt_ref_dict["Cut"]["set_1"]
tgt_ref_dict["Cut_files_motions_3021_Cut1_c_0_05cm_02"] = tgt_ref_dict["Cut"]["set_1"]
tgt_ref_dict["Cut_files_motions_3021_Cut1_c_0_05cm_03"] = tgt_ref_dict["Cut"]["set_1"]

tgt_ref_dict["Cut_files_motions_3022_Cut2_c_0_2cm_01"] = tgt_ref_dict["Cut"]["set_1"]
tgt_ref_dict["Cut_files_motions_3022_Cut2_c_0_2cm_02"] = tgt_ref_dict["Cut"]["set_1"]
tgt_ref_dict["Cut_files_motions_3022_Cut2_c_0_2cm_03"] = tgt_ref_dict["Cut"]["set_1"]

tgt_ref_dict["Cut_files_motions_3023_Cut3_c_30_05cm_01"] = tgt_ref_dict["Cut"]["set_1"]
tgt_ref_dict["Cut_files_motions_3023_Cut3_c_30_05cm_02"] = tgt_ref_dict["Cut"]["set_1"]
tgt_ref_dict["Cut_files_motions_3023_Cut3_c_30_05cm_03"] = tgt_ref_dict["Cut"]["set_1"]

tgt_ref_dict["Cut_files_motions_3024_Cut4_c_30_2cm_01"] = tgt_ref_dict["Cut"]["set_1"]
tgt_ref_dict["Cut_files_motions_3024_Cut4_c_30_2cm_02"] = tgt_ref_dict["Cut"]["set_1"]

tgt_ref_dict["Cut_files_motions_3025_Cut5_c_0_1_diag_03"] = tgt_ref_dict["Cut"]["set_1"]
tgt_ref_dict["Cut_files_motions_3025_Cut5_c_0_1_diag_04"] = tgt_ref_dict["Cut"]["set_1"]
tgt_ref_dict["Cut_files_motions_3025_Cut5_c_0_1_diag_05"] = tgt_ref_dict["Cut"]["set_1"]

tgt_ref_dict["Cut_files_motions_3028_Cut1_c_0_05cm_02"] = tgt_ref_dict["Cut"]["set_1"]
tgt_ref_dict["Cut_files_motions_3028_Cut1_c_0_05cm_03"] = tgt_ref_dict["Cut"]["set_1"]
tgt_ref_dict["Cut_files_motions_3028_Cut1_c_0_05cm_04"] = tgt_ref_dict["Cut"]["set_1"]

tgt_ref_dict["Cut_files_motions_3029_Cut2_c_0_2cm_01"] = tgt_ref_dict["Cut"]["set_1"]
tgt_ref_dict["Cut_files_motions_3029_Cut2_c_0_2cm_02"] = tgt_ref_dict["Cut"]["set_1"]
tgt_ref_dict["Cut_files_motions_3029_Cut2_c_0_2cm_03"] = tgt_ref_dict["Cut"]["set_1"]

# no fine motion

# list of actions
tgt_ref_dict["RollOut"] = {}
tgt_ref_dict["RollOut"]["set_1"] = {}
tgt_ref_dict["RollOut"]["set_1"]["Idle"] = {}
tgt_ref_dict["RollOut"]["set_1"]["Approach"] = {}
tgt_ref_dict["RollOut"]["set_1"]["Move"] = {}
tgt_ref_dict["RollOut"]["set_1"]["RollOut"] = {}
tgt_ref_dict["RollOut"]["set_1"]["Place"] = {}
tgt_ref_dict["RollOut"]["set_1"]["Retreat"] = {}

tgt_ref_dict["RollOut_files_motions_3113_Roll1_dough_start_05"] = tgt_ref_dict["RollOut"]["set_1"]
tgt_ref_dict["RollOut_files_motions_3113_Roll1_dough_start_06"] = tgt_ref_dict["RollOut"]["set_1"]
tgt_ref_dict["RollOut_files_motions_3113_Roll1_dough_start_07"] = tgt_ref_dict["RollOut"]["set_1"]

tgt_ref_dict["RollOut_files_motions_3114_Roll2_dough_forward_01"] = tgt_ref_dict["RollOut"]["set_1"]
tgt_ref_dict["RollOut_files_motions_3114_Roll2_dough_forward_02"] = tgt_ref_dict["RollOut"]["set_1"]
tgt_ref_dict["RollOut_files_motions_3114_Roll2_dough_forward_03"] = tgt_ref_dict["RollOut"]["set_1"]

tgt_ref_dict["RollOut_files_motions_3117_Roll1_dough_start_04"] = tgt_ref_dict["RollOut"]["set_1"]
tgt_ref_dict["RollOut_files_motions_3117_Roll1_dough_start_05"] = tgt_ref_dict["RollOut"]["set_1"]

tgt_ref_dict["RollOut_files_motions_3118_Roll2_dough_forward_03"] = tgt_ref_dict["RollOut"]["set_1"]
tgt_ref_dict["RollOut_files_motions_3118_Roll2_dough_forward_04"] = tgt_ref_dict["RollOut"]["set_1"]
tgt_ref_dict["RollOut_files_motions_3118_Roll2_dough_forward_07"] = tgt_ref_dict["RollOut"]["set_1"]

# fine motion wrt cucumber_attachment

tgt_ref_dict["Peel"] = {}
tgt_ref_dict["Peel"]["set_1"] = {}
tgt_ref_dict["Peel"]["set_1"]["Idle"] = {}
tgt_ref_dict["Peel"]["set_1"]["Approach"] = {}
tgt_ref_dict["Peel"]["set_1"]["Move"] = {}
tgt_ref_dict["Peel"]["set_1"]["Peel"] = {}
tgt_ref_dict["Peel"]["set_1"]["Hold"] = {}
tgt_ref_dict["Peel"]["set_1"]["Place"] = {}
tgt_ref_dict["Peel"]["set_1"]["Retreat"] = {}
tgt_ref_dict["Peel"]["set_1"]["Peel"]["peeler"] = "cucumber_attachment"

tgt_ref_dict["Peel_files_motions_3109_Peel1_cucumber_cb_04"] = tgt_ref_dict["Peel"]["set_1"]
tgt_ref_dict["Peel_files_motions_3109_Peel1_cucumber_cb_05"] = tgt_ref_dict["Peel"]["set_1"]
tgt_ref_dict["Peel_files_motions_3109_Peel1_cucumber_cb_07"] = tgt_ref_dict["Peel"]["set_1"]

tgt_ref_dict["Peel_files_motions_3110_Peel2_cucumber_gb_01"] = tgt_ref_dict["Peel"]["set_1"]
tgt_ref_dict["Peel_files_motions_3110_Peel2_cucumber_gb_04"] = tgt_ref_dict["Peel"]["set_1"]
tgt_ref_dict["Peel_files_motions_3110_Peel2_cucumber_gb_06"] = tgt_ref_dict["Peel"]["set_1"]

tgt_ref_dict["Peel_files_motions_3111_Peel1_cucumber_cb_01"] = tgt_ref_dict["Peel"]["set_1"]
tgt_ref_dict["Peel_files_motions_3111_Peel1_cucumber_cb_05"] = tgt_ref_dict["Peel"]["set_1"]
tgt_ref_dict["Peel_files_motions_3111_Peel1_cucumber_cb_07"] = tgt_ref_dict["Peel"]["set_1"]

tgt_ref_dict["Peel_files_motions_3112_Peel2_cucumber_gb_03"] = tgt_ref_dict["Peel"]["set_1"]
tgt_ref_dict["Peel_files_motions_3112_Peel2_cucumber_gb_04"] = tgt_ref_dict["Peel"]["set_1"]
tgt_ref_dict["Peel_files_motions_3112_Peel2_cucumber_gb_06"] = tgt_ref_dict["Peel"]["set_1"]

# fine motion wrt plate_dish, mixing_bowl_green, or cutting_board

tgt_ref_dict["Wipe"] = {}
tgt_ref_dict["Wipe"]["set_1"] = {}
tgt_ref_dict["Wipe"]["set_1"]["Idle"] = {}
tgt_ref_dict["Wipe"]["set_1"]["Approach"] = {}
tgt_ref_dict["Wipe"]["set_1"]["Move"] = {}
tgt_ref_dict["Wipe"]["set_1"]["Wipe"] = {}
tgt_ref_dict["Wipe"]["set_1"]["Hold"] = {}
tgt_ref_dict["Wipe"]["set_1"]["Place"] = {}
tgt_ref_dict["Wipe"]["set_1"]["Retreat"] = {}
tgt_ref_dict["Wipe"]["set_1"]["Wipe"]["sponge_small"] = "plate_dish"

tgt_ref_dict["Wipe"]["set_2"] = {}
tgt_ref_dict["Wipe"]["set_2"]["Idle"] = {}
tgt_ref_dict["Wipe"]["set_2"]["Approach"] = {}
tgt_ref_dict["Wipe"]["set_2"]["Move"] = {}
tgt_ref_dict["Wipe"]["set_2"]["Wipe"] = {}
tgt_ref_dict["Wipe"]["set_2"]["Hold"] = {}
tgt_ref_dict["Wipe"]["set_2"]["Place"] = {}
tgt_ref_dict["Wipe"]["set_2"]["Retreat"] = {}
tgt_ref_dict["Wipe"]["set_2"]["Wipe"]["sponge_small"] = "mixing_bowl_green"

tgt_ref_dict["Wipe"]["set_3"] = {}
tgt_ref_dict["Wipe"]["set_3"]["Idle"] = {}
tgt_ref_dict["Wipe"]["set_3"]["Approach"] = {}
tgt_ref_dict["Wipe"]["set_3"]["Move"] = {}
tgt_ref_dict["Wipe"]["set_3"]["Wipe"] = {}
tgt_ref_dict["Wipe"]["set_3"]["Hold"] = {}
tgt_ref_dict["Wipe"]["set_3"]["Place"] = {}
tgt_ref_dict["Wipe"]["set_3"]["Retreat"] = {}
tgt_ref_dict["Wipe"]["set_3"]["Wipe"]["sponge_small"] = "cutting_board_small"

tgt_ref_dict["Wipe_files_motions_3091_Wipe1_pl_sp_front_angle_01"] = tgt_ref_dict["Wipe"]["set_1"]
tgt_ref_dict["Wipe_files_motions_3091_Wipe1_pl_sp_front_angle_02"] = tgt_ref_dict["Wipe"]["set_1"]
tgt_ref_dict["Wipe_files_motions_3091_Wipe1_pl_sp_front_angle_03"] = tgt_ref_dict["Wipe"]["set_1"]

tgt_ref_dict["Wipe_files_motions_3092_Wipe2_pl_sp_back_angle_01"] = tgt_ref_dict["Wipe"]["set_1"]
tgt_ref_dict["Wipe_files_motions_3092_Wipe2_pl_sp_back_angle_03"] = tgt_ref_dict["Wipe"]["set_1"]
tgt_ref_dict["Wipe_files_motions_3092_Wipe2_pl_sp_back_angle_04"] = tgt_ref_dict["Wipe"]["set_1"]

tgt_ref_dict["Wipe_files_motions_3093_Wipe3_pl_sp_front_back_angle_01"] = tgt_ref_dict["Wipe"]["set_1"]
tgt_ref_dict["Wipe_files_motions_3093_Wipe3_pl_sp_front_back_angle_02"] = tgt_ref_dict["Wipe"]["set_1"]
tgt_ref_dict["Wipe_files_motions_3093_Wipe3_pl_sp_front_back_angle_03"] = tgt_ref_dict["Wipe"]["set_1"]

tgt_ref_dict["Wipe_files_motions_3094_Wipe4_pl_sp_front_flat_01"] = tgt_ref_dict["Wipe"]["set_1"]
tgt_ref_dict["Wipe_files_motions_3094_Wipe4_pl_sp_front_flat_03"] = tgt_ref_dict["Wipe"]["set_1"]
tgt_ref_dict["Wipe_files_motions_3094_Wipe4_pl_sp_front_flat_04"] = tgt_ref_dict["Wipe"]["set_1"]

tgt_ref_dict["Wipe_files_motions_3095_Wipe5_gb_sp_outside_spot_03"] = tgt_ref_dict["Wipe"]["set_2"]
tgt_ref_dict["Wipe_files_motions_3095_Wipe5_gb_sp_outside_spot_04"] = tgt_ref_dict["Wipe"]["set_2"]
tgt_ref_dict["Wipe_files_motions_3095_Wipe5_gb_sp_outside_spot_05"] = tgt_ref_dict["Wipe"]["set_2"]

tgt_ref_dict["Wipe_files_motions_3096_Wipe6_gb_sp_outside_02"] = tgt_ref_dict["Wipe"]["set_2"]
tgt_ref_dict["Wipe_files_motions_3096_Wipe6_gb_sp_outside_03"] = tgt_ref_dict["Wipe"]["set_2"]
tgt_ref_dict["Wipe_files_motions_3096_Wipe6_gb_sp_outside_04"] = tgt_ref_dict["Wipe"]["set_2"]

tgt_ref_dict["Wipe_files_motions_3097_Wipe7_gb_sp_outside_bottom_01"] = tgt_ref_dict["Wipe"]["set_2"]
tgt_ref_dict["Wipe_files_motions_3097_Wipe7_gb_sp_outside_bottom_02"] = tgt_ref_dict["Wipe"]["set_2"]
tgt_ref_dict["Wipe_files_motions_3097_Wipe7_gb_sp_outside_bottom_03"] = tgt_ref_dict["Wipe"]["set_2"]

tgt_ref_dict["Wipe_files_motions_3098_Wipe8_cb_sp_top_02"] = tgt_ref_dict["Wipe"]["set_3"]
tgt_ref_dict["Wipe_files_motions_3098_Wipe8_cb_sp_top_03"] = tgt_ref_dict["Wipe"]["set_3"]
tgt_ref_dict["Wipe_files_motions_3098_Wipe8_cb_sp_top_04"] = tgt_ref_dict["Wipe"]["set_3"]

tgt_ref_dict["Wipe_files_motions_3099_Wipe9_cb_sp_top_bottom_01"] = tgt_ref_dict["Wipe"]["set_3"]
tgt_ref_dict["Wipe_files_motions_3099_Wipe9_cb_sp_top_bottom_04"] = tgt_ref_dict["Wipe"]["set_3"]
tgt_ref_dict["Wipe_files_motions_3099_Wipe9_cb_sp_top_bottom_05"] = tgt_ref_dict["Wipe"]["set_3"]

tgt_ref_dict["Wipe_files_motions_3100_Wipe1_pl_sp_front_angle_01"] = tgt_ref_dict["Wipe"]["set_1"]
tgt_ref_dict["Wipe_files_motions_3100_Wipe1_pl_sp_front_angle_02"] = tgt_ref_dict["Wipe"]["set_1"]
tgt_ref_dict["Wipe_files_motions_3100_Wipe1_pl_sp_front_angle_03"] = tgt_ref_dict["Wipe"]["set_1"]

tgt_ref_dict["Wipe_files_motions_3101_Wipe2_pl_sp_back_angle_03"] = tgt_ref_dict["Wipe"]["set_1"]
tgt_ref_dict["Wipe_files_motions_3101_Wipe2_pl_sp_back_angle_05"] = tgt_ref_dict["Wipe"]["set_1"]
tgt_ref_dict["Wipe_files_motions_3101_Wipe2_pl_sp_back_angle_06"] = tgt_ref_dict["Wipe"]["set_1"]

tgt_ref_dict["Wipe_files_motions_3102_Wipe3_pl_sp_front_back_angle_01"] = tgt_ref_dict["Wipe"]["set_1"]
tgt_ref_dict["Wipe_files_motions_3102_Wipe3_pl_sp_front_back_angle_02"] = tgt_ref_dict["Wipe"]["set_1"]
tgt_ref_dict["Wipe_files_motions_3102_Wipe3_pl_sp_front_back_angle_03"] = tgt_ref_dict["Wipe"]["set_1"]

tgt_ref_dict["Wipe_files_motions_3103_Wipe4_pl_sp_front_flat_01"] = tgt_ref_dict["Wipe"]["set_1"]
tgt_ref_dict["Wipe_files_motions_3103_Wipe4_pl_sp_front_flat_02"] = tgt_ref_dict["Wipe"]["set_1"]
tgt_ref_dict["Wipe_files_motions_3103_Wipe4_pl_sp_front_flat_03"] = tgt_ref_dict["Wipe"]["set_1"]

tgt_ref_dict["Wipe_files_motions_3104_Wipe5_gb_sp_outside_spot_03"] = tgt_ref_dict["Wipe"]["set_2"]
tgt_ref_dict["Wipe_files_motions_3104_Wipe5_gb_sp_outside_spot_04"] = tgt_ref_dict["Wipe"]["set_2"]
tgt_ref_dict["Wipe_files_motions_3104_Wipe5_gb_sp_outside_spot_05"] = tgt_ref_dict["Wipe"]["set_2"]

tgt_ref_dict["Wipe_files_motions_3105_Wipe6_gb_sp_outside_02"] = tgt_ref_dict["Wipe"]["set_2"]
tgt_ref_dict["Wipe_files_motions_3105_Wipe6_gb_sp_outside_04"] = tgt_ref_dict["Wipe"]["set_2"]
tgt_ref_dict["Wipe_files_motions_3105_Wipe6_gb_sp_outside_05"] = tgt_ref_dict["Wipe"]["set_2"]

tgt_ref_dict["Wipe_files_motions_3106_Wipe7_gb_sp_outside_bottom_01"] = tgt_ref_dict["Wipe"]["set_2"]
tgt_ref_dict["Wipe_files_motions_3106_Wipe7_gb_sp_outside_bottom_02"] = tgt_ref_dict["Wipe"]["set_2"]
tgt_ref_dict["Wipe_files_motions_3106_Wipe7_gb_sp_outside_bottom_03"] = tgt_ref_dict["Wipe"]["set_2"]

tgt_ref_dict["Wipe_files_motions_3107_Wipe8_cb_sp_top_01"] = tgt_ref_dict["Wipe"]["set_3"]
tgt_ref_dict["Wipe_files_motions_3107_Wipe8_cb_sp_top_02"] = tgt_ref_dict["Wipe"]["set_3"]
tgt_ref_dict["Wipe_files_motions_3107_Wipe8_cb_sp_top_03"] = tgt_ref_dict["Wipe"]["set_3"]

tgt_ref_dict["Wipe_files_motions_3108_Wipe9_cb_sp_top_bottom_01"] = tgt_ref_dict["Wipe"]["set_3"]
tgt_ref_dict["Wipe_files_motions_3108_Wipe9_cb_sp_top_bottom_03"] = tgt_ref_dict["Wipe"]["set_3"]
tgt_ref_dict["Wipe_files_motions_3108_Wipe9_cb_sp_top_bottom_04"] = tgt_ref_dict["Wipe"]["set_3"]

# no fine motion so represent all wrt kitchen_sideboard

tgt_ref_dict["Scoop"] = {}
tgt_ref_dict["Scoop"]["set_1"] = {}
tgt_ref_dict["Scoop"]["set_1"]["Idle"] = {}
tgt_ref_dict["Scoop"]["set_1"]["Approach"] = {}
tgt_ref_dict["Scoop"]["set_1"]["Move"] = {}
tgt_ref_dict["Scoop"]["set_1"]["Scoop"] = {}
tgt_ref_dict["Scoop"]["set_1"]["Hold"] = {}
tgt_ref_dict["Scoop"]["set_1"]["Place"] = {}
tgt_ref_dict["Scoop"]["set_1"]["Retreat"] = {}
#tgt_ref_dict["Scoop"]["set_1"]["Scoop"]["ladle"] = "cup_small"
tgt_ref_dict["Scoop"]["set_2"] = {}
tgt_ref_dict["Scoop"]["set_2"]["Idle"] = {}
tgt_ref_dict["Scoop"]["set_2"]["Approach"] = {}
tgt_ref_dict["Scoop"]["set_2"]["Move"] = {}
tgt_ref_dict["Scoop"]["set_2"]["Scoop"] = {}
tgt_ref_dict["Scoop"]["set_2"]["Hold"] = {}
tgt_ref_dict["Scoop"]["set_2"]["Place"] = {}
tgt_ref_dict["Scoop"]["set_2"]["Retreat"] = {}
#tgt_ref_dict["Scoop"]["set_2"]["Scoop"]["ladle"] = "cup_large"

tgt_ref_dict["Scoop_files_motions_2989_Scoop1_mbg_l_CS_225_30o_02"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_2989_Scoop1_mbg_l_CS_225_30o_03"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_2989_Scoop1_mbg_l_CS_225_30o_04"] = tgt_ref_dict["Scoop"]["set_1"]

tgt_ref_dict["Scoop_files_motions_2990_Scoop2_mbg_l_CS_135_40o_01"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_2990_Scoop2_mbg_l_CS_135_40o_02"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_2990_Scoop2_mbg_l_CS_135_40o_03"] = tgt_ref_dict["Scoop"]["set_1"]

tgt_ref_dict["Scoop_files_motions_2991_Scoop3_mbg_l_CS_m45_20o_01"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_2991_Scoop3_mbg_l_CS_m45_20o_02"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_2991_Scoop3_mbg_l_CS_m45_20o_04"] = tgt_ref_dict["Scoop"]["set_1"]

tgt_ref_dict["Scoop_files_motions_2992_Scoop4_mbg_l_CL_225_30o_01"] = tgt_ref_dict["Scoop"]["set_2"]
tgt_ref_dict["Scoop_files_motions_2992_Scoop4_mbg_l_CL_225_30o_02"] = tgt_ref_dict["Scoop"]["set_2"]
tgt_ref_dict["Scoop_files_motions_2992_Scoop4_mbg_l_CL_225_30o_03"] = tgt_ref_dict["Scoop"]["set_2"]

tgt_ref_dict["Scoop_files_motions_2993_Scoop5_mbg_l_CL_135_40o_01"] = tgt_ref_dict["Scoop"]["set_2"]
tgt_ref_dict["Scoop_files_motions_2993_Scoop5_mbg_l_CL_135_40o_02"] = tgt_ref_dict["Scoop"]["set_2"]
tgt_ref_dict["Scoop_files_motions_2993_Scoop5_mbg_l_CL_135_40o_03"] = tgt_ref_dict["Scoop"]["set_2"]

tgt_ref_dict["Scoop_files_motions_2994_Scoop6_mbg_l_CL_m45_20o_01"] = tgt_ref_dict["Scoop"]["set_2"]
tgt_ref_dict["Scoop_files_motions_2994_Scoop6_mbg_l_CL_m45_20o_04"] = tgt_ref_dict["Scoop"]["set_2"]
tgt_ref_dict["Scoop_files_motions_2994_Scoop6_mbg_l_CL_m45_20o_05"] = tgt_ref_dict["Scoop"]["set_2"]

tgt_ref_dict["Scoop_files_motions_2995_Scoop7_MBG_l_cs_180_30o_02"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_2995_Scoop7_MBG_l_cs_180_30o_03"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_2995_Scoop7_MBG_l_cs_180_30o_04"] = tgt_ref_dict["Scoop"]["set_1"]

tgt_ref_dict["Scoop_files_motions_2996_Scoop8_MBG_l_cs_0_40o_01"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_2996_Scoop8_MBG_l_cs_0_40o_02"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_2996_Scoop8_MBG_l_cs_0_40o_04"] = tgt_ref_dict["Scoop"]["set_1"]

tgt_ref_dict["Scoop_files_motions_2997_Scoop9_MBG_l_cl_180_30o_01"] = tgt_ref_dict["Scoop"]["set_2"]
tgt_ref_dict["Scoop_files_motions_2997_Scoop9_MBG_l_cl_180_30o_02"] = tgt_ref_dict["Scoop"]["set_2"]
tgt_ref_dict["Scoop_files_motions_2997_Scoop9_MBG_l_cl_180_30o_03"] = tgt_ref_dict["Scoop"]["set_2"]

tgt_ref_dict["Scoop_files_motions_2998_Scoop10_MBG_l_cl_0_40o_01"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_2998_Scoop10_MBG_l_cl_0_40o_02"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_2998_Scoop10_MBG_l_cl_0_40o_04"] = tgt_ref_dict["Scoop"]["set_1"]

tgt_ref_dict["Scoop_files_motions_2999_Scoop11_mbg_t_CS_0_40i_01"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_2999_Scoop11_mbg_t_CS_0_40i_02"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_2999_Scoop11_mbg_t_CS_0_40i_03"] = tgt_ref_dict["Scoop"]["set_1"]

tgt_ref_dict["Scoop_files_motions_3000_Scoop12_mbg_t_CS_180_30i_02"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_3000_Scoop12_mbg_t_CS_180_30i_03"] = tgt_ref_dict["Scoop"]["set_1"]

tgt_ref_dict["Scoop_files_motions_3001_Scoop13_mbg_t_CS_270_15i_01"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_3001_Scoop13_mbg_t_CS_270_15i_03"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_3001_Scoop13_mbg_t_CS_270_15i_04"] = tgt_ref_dict["Scoop"]["set_1"]

tgt_ref_dict["Scoop_files_motions_3005_Scoop1_mbg_l_CS_225_30o_02"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_3005_Scoop1_mbg_l_CS_225_30o_03"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_3005_Scoop1_mbg_l_CS_225_30o_04"] = tgt_ref_dict["Scoop"]["set_1"]

tgt_ref_dict["Scoop_files_motions_3006_Scoop2_mbg_l_CS_135_40o_01"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_3006_Scoop2_mbg_l_CS_135_40o_04"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_3006_Scoop2_mbg_l_CS_135_40o_07"] = tgt_ref_dict["Scoop"]["set_1"]

tgt_ref_dict["Scoop_files_motions_3007_Scoop3_mbg_l_CS_m45_20o_01"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_3007_Scoop3_mbg_l_CS_m45_20o_02"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_3007_Scoop3_mbg_l_CS_m45_20o_03"] = tgt_ref_dict["Scoop"]["set_1"]

tgt_ref_dict["Scoop_files_motions_3008_Scoop4_mbg_l_CL_225_30o_01"] = tgt_ref_dict["Scoop"]["set_1"]

tgt_ref_dict["Scoop_files_motions_3009_Scoop5_mbg_l_CL_135_40o_01"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_3009_Scoop5_mbg_l_CL_135_40o_02"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_3009_Scoop5_mbg_l_CL_135_40o_03"] = tgt_ref_dict["Scoop"]["set_1"]

tgt_ref_dict["Scoop_files_motions_3010_Scoop6_mbg_l_CL_m45_20o_01"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_3010_Scoop6_mbg_l_CL_m45_20o_03"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_3010_Scoop6_mbg_l_CL_m45_20o_04"] = tgt_ref_dict["Scoop"]["set_1"]

tgt_ref_dict["Scoop_files_motions_3011_Scoop7_MBG_l_cs_180_30o_02"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_3011_Scoop7_MBG_l_cs_180_30o_04"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_3011_Scoop7_MBG_l_cs_180_30o_05"] = tgt_ref_dict["Scoop"]["set_1"]

tgt_ref_dict["Scoop_files_motions_3012_Scoop8_MBG_l_cs_0_40o_01"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_3012_Scoop8_MBG_l_cs_0_40o_02"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_3012_Scoop8_MBG_l_cs_0_40o_03"] = tgt_ref_dict["Scoop"]["set_1"]

tgt_ref_dict["Scoop_files_motions_3013_Scoop9_MBG_l_cl_180_30o_01"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_3013_Scoop9_MBG_l_cl_180_30o_02"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_3013_Scoop9_MBG_l_cl_180_30o_04"] = tgt_ref_dict["Scoop"]["set_1"]

tgt_ref_dict["Scoop_files_motions_3014_Scoop10_MBG_l_cl_0_40o_01"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_3014_Scoop10_MBG_l_cl_0_40o_03"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_3014_Scoop10_MBG_l_cl_0_40o_04"] = tgt_ref_dict["Scoop"]["set_1"]

tgt_ref_dict["Scoop_files_motions_3015_Scoop11_mbg_t_CS_0_40i_02"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_3015_Scoop11_mbg_t_CS_0_40i_04"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_3015_Scoop11_mbg_t_CS_0_40i_05"] = tgt_ref_dict["Scoop"]["set_1"]

tgt_ref_dict["Scoop_files_motions_3016_Scoop12_mbg_t_CS_180_30i_01"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_3016_Scoop12_mbg_t_CS_180_30i_02"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_3016_Scoop12_mbg_t_CS_180_30i_03"] = tgt_ref_dict["Scoop"]["set_1"]

tgt_ref_dict["Scoop_files_motions_3017_Scoop13_mbg_t_CS_270_15i_01"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_3017_Scoop13_mbg_t_CS_270_15i_02"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_3017_Scoop13_mbg_t_CS_270_15i_03"] = tgt_ref_dict["Scoop"]["set_1"]

tgt_ref_dict["Scoop_files_motions_3057_Scoop17_salad_GB_pl_0_40_04"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_3057_Scoop17_salad_GB_pl_0_40_06"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_3057_Scoop17_salad_GB_pl_0_40_09"] = tgt_ref_dict["Scoop"]["set_1"]

tgt_ref_dict["Scoop_files_motions_3058_Scoop18_salad_PL_gb_180_40_03"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_3058_Scoop18_salad_PL_gb_180_40_04"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_3058_Scoop18_salad_PL_gb_180_40_05"] = tgt_ref_dict["Scoop"]["set_1"]

tgt_ref_dict["Scoop_files_motions_3059_Scoop18_salad_PL_gb_90_30_01"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_3059_Scoop18_salad_PL_gb_90_30_03"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_3059_Scoop18_salad_PL_gb_90_30_05"] = tgt_ref_dict["Scoop"]["set_1"]

tgt_ref_dict["Scoop_files_motions_3060_Scoop17_salad_GB_pl_0_40_01"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_3060_Scoop17_salad_GB_pl_0_40_02"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_3060_Scoop17_salad_GB_pl_0_40_03"] = tgt_ref_dict["Scoop"]["set_1"]

tgt_ref_dict["Scoop_files_motions_3061_Scoop18_salad_PL_gb_180_40_01"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_3061_Scoop18_salad_PL_gb_180_40_02"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_3061_Scoop18_salad_PL_gb_180_40_03"] = tgt_ref_dict["Scoop"]["set_1"]

tgt_ref_dict["Scoop_files_motions_3062_Scoop19_salad_PL_gb_90_30_02"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_3062_Scoop19_salad_PL_gb_90_30_03"] = tgt_ref_dict["Scoop"]["set_1"]
tgt_ref_dict["Scoop_files_motions_3062_Scoop19_salad_PL_gb_90_30_04"] = tgt_ref_dict["Scoop"]["set_1"]

# fine motion wrt cup_small, mixing_bowl_small, or mixing_bowl_green

tgt_ref_dict["Stir"] = {}
tgt_ref_dict["Stir"]["set_1"] = {}
tgt_ref_dict["Stir"]["set_1"]["Idle"] = {}
tgt_ref_dict["Stir"]["set_1"]["Approach"] = {}
tgt_ref_dict["Stir"]["set_1"]["Move"] = {}
tgt_ref_dict["Stir"]["set_1"]["Stir"] = {}
tgt_ref_dict["Stir"]["set_1"]["Hold"] = {}
tgt_ref_dict["Stir"]["set_1"]["Place"] = {}
tgt_ref_dict["Stir"]["set_1"]["Retreat"] = {}
tgt_ref_dict["Stir"]["set_1"]["Stir"]["tablespoon"] = "cup_large"

tgt_ref_dict["Stir"]["set_2"] = {}
tgt_ref_dict["Stir"]["set_2"]["Idle"] = {}
tgt_ref_dict["Stir"]["set_2"]["Approach"] = {}
tgt_ref_dict["Stir"]["set_2"]["Move"] = {}
tgt_ref_dict["Stir"]["set_2"]["Stir"] = {}
tgt_ref_dict["Stir"]["set_2"]["Hold"] = {}
tgt_ref_dict["Stir"]["set_2"]["Place"] = {}
tgt_ref_dict["Stir"]["set_2"]["Retreat"] = {}
tgt_ref_dict["Stir"]["set_2"]["Stir"]["tablespoon"] = "mixing_bowl_small"

tgt_ref_dict["Stir"]["set_3"] = {}
tgt_ref_dict["Stir"]["set_3"]["Idle"] = {}
tgt_ref_dict["Stir"]["set_3"]["Approach"] = {}
tgt_ref_dict["Stir"]["set_3"]["Move"] = {}
tgt_ref_dict["Stir"]["set_3"]["Stir"] = {}
tgt_ref_dict["Stir"]["set_3"]["Hold"] = {}
tgt_ref_dict["Stir"]["set_3"]["Place"] = {}
tgt_ref_dict["Stir"]["set_3"]["Retreat"] = {}
tgt_ref_dict["Stir"]["set_3"]["Stir"]["egg_whisk"] = "mixing_bowl_green"

tgt_ref_dict["Stir_files_motions_2941_Stir1_bc_s_right_01"] = tgt_ref_dict["Stir"]["set_1"]
tgt_ref_dict["Stir_files_motions_2941_Stir1_bc_s_right_02"] = tgt_ref_dict["Stir"]["set_1"]
tgt_ref_dict["Stir_files_motions_2941_Stir1_bc_s_right_03"] = tgt_ref_dict["Stir"]["set_1"]

tgt_ref_dict["Stir_files_motions_2942_Stir2_bc_s_45cm_00"] = tgt_ref_dict["Stir"]["set_1"]
tgt_ref_dict["Stir_files_motions_2942_Stir2_bc_s_45cm_01"] = tgt_ref_dict["Stir"]["set_1"]
tgt_ref_dict["Stir_files_motions_2942_Stir2_bc_s_45cm_02"] = tgt_ref_dict["Stir"]["set_1"]

tgt_ref_dict["Stir_files_motions_2943_Stir3_rb_s_right_00"] = tgt_ref_dict["Stir"]["set_2"]
tgt_ref_dict["Stir_files_motions_2943_Stir3_rb_s_right_01"] = tgt_ref_dict["Stir"]["set_2"]
tgt_ref_dict["Stir_files_motions_2943_Stir3_rb_s_right_02"] = tgt_ref_dict["Stir"]["set_2"]

tgt_ref_dict["Stir_files_motions_2944_Stir4_rb_s_45cm_01"] = tgt_ref_dict["Stir"]["set_2"]
tgt_ref_dict["Stir_files_motions_2944_Stir4_rb_s_45cm_02"] = tgt_ref_dict["Stir"]["set_2"]

tgt_ref_dict["Stir_files_motions_2945_Stir5_rb_s_inside_01"] = tgt_ref_dict["Stir"]["set_2"]
tgt_ref_dict["Stir_files_motions_2945_Stir5_rb_s_inside_02"] = tgt_ref_dict["Stir"]["set_2"]
tgt_ref_dict["Stir_files_motions_2945_Stir5_rb_s_inside_03"] = tgt_ref_dict["Stir"]["set_2"]

tgt_ref_dict["Stir_files_motions_2946_Stir6_gb_w_tilted_00"] = tgt_ref_dict["Stir"]["set_3"]
tgt_ref_dict["Stir_files_motions_2946_Stir6_gb_w_tilted_01"] = tgt_ref_dict["Stir"]["set_3"]
tgt_ref_dict["Stir_files_motions_2946_Stir6_gb_w_tilted_02"] = tgt_ref_dict["Stir"]["set_3"]

tgt_ref_dict["Stir_files_motions_2947_Stir7_gb_w_stab_00"] = tgt_ref_dict["Stir"]["set_3"]
tgt_ref_dict["Stir_files_motions_2947_Stir7_gb_w_stab_02"] = tgt_ref_dict["Stir"]["set_3"]
tgt_ref_dict["Stir_files_motions_2947_Stir7_gb_w_stab_03"] = tgt_ref_dict["Stir"]["set_3"]

tgt_ref_dict["Stir_files_motions_2948_Stir8_gb_w_grasp_03"] = tgt_ref_dict["Stir"]["set_3"]
tgt_ref_dict["Stir_files_motions_2948_Stir8_gb_w_grasp_04"] = tgt_ref_dict["Stir"]["set_3"]
tgt_ref_dict["Stir_files_motions_2948_Stir8_gb_w_grasp_06"] = tgt_ref_dict["Stir"]["set_3"]

tgt_ref_dict["Stir_files_motions_2949_Stir9_gb_w_next_00"] = tgt_ref_dict["Stir"]["set_3"]
tgt_ref_dict["Stir_files_motions_2949_Stir9_gb_w_next_01"] = tgt_ref_dict["Stir"]["set_3"]
tgt_ref_dict["Stir_files_motions_2949_Stir9_gb_w_next_02"] = tgt_ref_dict["Stir"]["set_3"]

tgt_ref_dict["Stir_files_motions_2950_Stirring1_fast_01"] = tgt_ref_dict["Stir"]["set_3"]
tgt_ref_dict["Stir_files_motions_2950_Stirring1_fast_02"] = tgt_ref_dict["Stir"]["set_3"]

tgt_ref_dict["Stir_files_motions_2954_Stir1_cl_t_right_01"] = tgt_ref_dict["Stir"]["set_1"]
tgt_ref_dict["Stir_files_motions_2954_Stir1_cl_t_right_03"] = tgt_ref_dict["Stir"]["set_1"]
tgt_ref_dict["Stir_files_motions_2954_Stir1_cl_t_right_04"] = tgt_ref_dict["Stir"]["set_1"]

tgt_ref_dict["Stir_files_motions_2955_Stir2_cl_t_45deg_01"] = tgt_ref_dict["Stir"]["set_1"]
tgt_ref_dict["Stir_files_motions_2955_Stir2_cl_t_45deg_02"] = tgt_ref_dict["Stir"]["set_1"]
tgt_ref_dict["Stir_files_motions_2955_Stir2_cl_t_45deg_03"] = tgt_ref_dict["Stir"]["set_1"]

tgt_ref_dict["Stir_files_motions_2956_Stir3_mbs_t_right_01"] = tgt_ref_dict["Stir"]["set_2"]
tgt_ref_dict["Stir_files_motions_2956_Stir3_mbs_t_right_02"] = tgt_ref_dict["Stir"]["set_2"]
tgt_ref_dict["Stir_files_motions_2956_Stir3_mbs_t_right_03"] = tgt_ref_dict["Stir"]["set_2"]

tgt_ref_dict["Stir_files_motions_2957_Stir4_mbs_t_45deg_01"] = tgt_ref_dict["Stir"]["set_2"]
tgt_ref_dict["Stir_files_motions_2957_Stir4_mbs_t_45deg_02"] = tgt_ref_dict["Stir"]["set_2"]
tgt_ref_dict["Stir_files_motions_2957_Stir4_mbs_t_45deg_03"] = tgt_ref_dict["Stir"]["set_2"]

tgt_ref_dict["Stir_files_motions_2958_Stir5_mbg_ew_tilted_03"] = tgt_ref_dict["Stir"]["set_3"]
tgt_ref_dict["Stir_files_motions_2958_Stir5_mbg_ew_tilted_04"] = tgt_ref_dict["Stir"]["set_3"]
tgt_ref_dict["Stir_files_motions_2958_Stir5_mbg_ew_tilted_06"] = tgt_ref_dict["Stir"]["set_3"]

tgt_ref_dict["Stir_files_motions_2959_Stir6_mbg_ew_lifted_01"] = tgt_ref_dict["Stir"]["set_3"]
tgt_ref_dict["Stir_files_motions_2959_Stir6_mbg_ew_lifted_03"] = tgt_ref_dict["Stir"]["set_3"]

tgt_ref_dict["Stir_files_motions_2960_Stir7_mbg_ew_stab_01"] = tgt_ref_dict["Stir"]["set_3"]
tgt_ref_dict["Stir_files_motions_2960_Stir7_mbg_ew_stab_04"] = tgt_ref_dict["Stir"]["set_3"]
tgt_ref_dict["Stir_files_motions_2960_Stir7_mbg_ew_stab_05"] = tgt_ref_dict["Stir"]["set_3"]

tgt_ref_dict["Stir_files_motions_2961_Stir8_mbg_ew_grasp_tilted_06"] = tgt_ref_dict["Stir"]["set_3"]
tgt_ref_dict["Stir_files_motions_2961_Stir8_mbg_ew_grasp_tilted_07"] = tgt_ref_dict["Stir"]["set_3"]
tgt_ref_dict["Stir_files_motions_2961_Stir8_mbg_ew_grasp_tilted_08"] = tgt_ref_dict["Stir"]["set_3"]

tgt_ref_dict["Stir_files_motions_2962_Stir9_mbg_ew_grasp_lifted_01"] = tgt_ref_dict["Stir"]["set_3"]
tgt_ref_dict["Stir_files_motions_2962_Stir9_mbg_ew_grasp_lifted_02"] = tgt_ref_dict["Stir"]["set_3"]
tgt_ref_dict["Stir_files_motions_2962_Stir9_mbg_ew_grasp_lifted_03"] = tgt_ref_dict["Stir"]["set_3"]

tgt_ref_dict["Stir_files_motions_2963_Stir10_mbg_ew_fast_02"] = tgt_ref_dict["Stir"]["set_3"]
tgt_ref_dict["Stir_files_motions_2963_Stir10_mbg_ew_fast_03"] = tgt_ref_dict["Stir"]["set_3"]
tgt_ref_dict["Stir_files_motions_2963_Stir10_mbg_ew_fast_05"] = tgt_ref_dict["Stir"]["set_3"]

# fine motion wrt mixing_bowl_green

tgt_ref_dict["Mix"] = {}
tgt_ref_dict["Mix"]["set_1"] = {}
tgt_ref_dict["Mix"]["set_1"]["Idle"] = {}
tgt_ref_dict["Mix"]["set_1"]["Approach"] = {}
tgt_ref_dict["Mix"]["set_1"]["Move"] = {}
tgt_ref_dict["Mix"]["set_1"]["Mix"] = {}
tgt_ref_dict["Mix"]["set_1"]["Hold"] = {}
tgt_ref_dict["Mix"]["set_1"]["Place"] = {}
tgt_ref_dict["Mix"]["set_1"]["Retreat"] = {}
tgt_ref_dict["Mix"]["set_1"]["Mix"]["salad_fork"]  = "mixing_bowl_green"
tgt_ref_dict["Mix"]["set_1"]["Mix"]["salad_spoon"] = "mixing_bowl_green"

tgt_ref_dict["Mix_files_motions_3121_Mix1_salad_gb_parallel_06"] = tgt_ref_dict["Mix"]["set_1"]
tgt_ref_dict["Mix_files_motions_3121_Mix1_salad_gb_parallel_07"] = tgt_ref_dict["Mix"]["set_1"]
tgt_ref_dict["Mix_files_motions_3121_Mix1_salad_gb_parallel_08"] = tgt_ref_dict["Mix"]["set_1"]

tgt_ref_dict["Mix_files_motions_3122_Mix2_salad_gb_circle_04"] = tgt_ref_dict["Mix"]["set_1"]
tgt_ref_dict["Mix_files_motions_3122_Mix2_salad_gb_circle_05"] = tgt_ref_dict["Mix"]["set_1"]
tgt_ref_dict["Mix_files_motions_3122_Mix2_salad_gb_circle_06"] = tgt_ref_dict["Mix"]["set_1"]

tgt_ref_dict["Mix_files_motions_3123_Mix3_salad_gb_diagonal_05"] = tgt_ref_dict["Mix"]["set_1"]
tgt_ref_dict["Mix_files_motions_3123_Mix3_salad_gb_diagonal_06"] = tgt_ref_dict["Mix"]["set_1"]
tgt_ref_dict["Mix_files_motions_3123_Mix3_salad_gb_diagonal_07"] = tgt_ref_dict["Mix"]["set_1"]

tgt_ref_dict["Mix_files_motions_3124_Mix1_salad_gb_parallel_05"] = tgt_ref_dict["Mix"]["set_1"]
tgt_ref_dict["Mix_files_motions_3124_Mix1_salad_gb_parallel_08"] = tgt_ref_dict["Mix"]["set_1"]
tgt_ref_dict["Mix_files_motions_3124_Mix1_salad_gb_parallel_09"] = tgt_ref_dict["Mix"]["set_1"]

# no fine motion so everything can be expressed wrt to kitchen_sideboard

tgt_ref_dict["Transfer"] = {}
tgt_ref_dict["Transfer"]["set_1"] = {}
tgt_ref_dict["Transfer"]["set_1"]["Idle"] = {}
tgt_ref_dict["Transfer"]["set_1"]["Approach"] = {}
tgt_ref_dict["Transfer"]["set_1"]["Move"] = {}
tgt_ref_dict["Transfer"]["set_1"]["Transfer"] = {}
tgt_ref_dict["Transfer"]["set_1"]["Hold"] = {}
tgt_ref_dict["Transfer"]["set_1"]["Place"] = {}
tgt_ref_dict["Transfer"]["set_1"]["Retreat"] = {}

tgt_ref_dict["Transfer_files_motions_3035_Transfer1_mbg_01"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3035_Transfer1_mbg_02"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3035_Transfer1_mbg_03"] = tgt_ref_dict["Transfer"]["set_1"]

tgt_ref_dict["Transfer_files_motions_3036_Transfer2_mbg_01"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3036_Transfer2_mbg_02"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3036_Transfer2_mbg_03"] = tgt_ref_dict["Transfer"]["set_1"]

tgt_ref_dict["Transfer_files_motions_3037_Transfer3_mbg_01"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3037_Transfer3_mbg_02"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3037_Transfer3_mbg_03"] = tgt_ref_dict["Transfer"]["set_1"]

tgt_ref_dict["Transfer_files_motions_3038_Transfer4_mbg_02"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3038_Transfer4_mbg_03"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3038_Transfer4_mbg_04"] = tgt_ref_dict["Transfer"]["set_1"]

tgt_ref_dict["Transfer_files_motions_3039_Transfer5_cbs_01"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3039_Transfer5_cbs_02"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3039_Transfer5_cbs_03"] = tgt_ref_dict["Transfer"]["set_1"]

tgt_ref_dict["Transfer_files_motions_3040_Transfer6_cbs_01"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3040_Transfer6_cbs_02"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3040_Transfer6_cbs_04"] = tgt_ref_dict["Transfer"]["set_1"]

tgt_ref_dict["Transfer_files_motions_3041_Transfer7_cbs_01"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3041_Transfer7_cbs_03"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3041_Transfer7_cbs_05"] = tgt_ref_dict["Transfer"]["set_1"]

tgt_ref_dict["Transfer_files_motions_3042_Transfer8_cbs_01"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3042_Transfer8_cbs_02"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3042_Transfer8_cbs_03"] = tgt_ref_dict["Transfer"]["set_1"]

tgt_ref_dict["Transfer_files_motions_3043_Transfer9_cbs_01"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3043_Transfer9_cbs_03"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3043_Transfer9_cbs_04"] = tgt_ref_dict["Transfer"]["set_1"]

tgt_ref_dict["Transfer_files_motions_3044_Transfer10_cbs_01"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3044_Transfer10_cbs_02"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3044_Transfer10_cbs_03"] = tgt_ref_dict["Transfer"]["set_1"]

tgt_ref_dict["Transfer_files_motions_3045_Transfer11_cbs_01"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3045_Transfer11_cbs_02"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3045_Transfer11_cbs_03"] = tgt_ref_dict["Transfer"]["set_1"]

tgt_ref_dict["Transfer_files_motions_3046_Transfer12_cbs_01"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3046_Transfer12_cbs_02"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3046_Transfer12_cbs_03"] = tgt_ref_dict["Transfer"]["set_1"]

tgt_ref_dict["Transfer_files_motions_3047_Transfer1_mbg_01"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3047_Transfer1_mbg_02"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3047_Transfer1_mbg_03"] = tgt_ref_dict["Transfer"]["set_1"]

tgt_ref_dict["Transfer_files_motions_3048_Transfer2_mbg_01"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3048_Transfer2_mbg_02"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3048_Transfer2_mbg_03"] = tgt_ref_dict["Transfer"]["set_1"]

tgt_ref_dict["Transfer_files_motions_3049_Transfer3_mbg_01"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3049_Transfer3_mbg_02"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3049_Transfer3_mbg_03"] = tgt_ref_dict["Transfer"]["set_1"]

tgt_ref_dict["Transfer_files_motions_3050_Transfer4_mbg_03"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3050_Transfer4_mbg_05"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3050_Transfer4_mbg_06"] = tgt_ref_dict["Transfer"]["set_1"]

tgt_ref_dict["Transfer_files_motions_3063_Transfer5_cbs_05"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3063_Transfer5_cbs_06"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3063_Transfer5_cbs_07"] = tgt_ref_dict["Transfer"]["set_1"]

tgt_ref_dict["Transfer_files_motions_3064_Transfer6_cbs_02"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3064_Transfer6_cbs_03"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3064_Transfer6_cbs_05"] = tgt_ref_dict["Transfer"]["set_1"]

tgt_ref_dict["Transfer_files_motions_3065_Transfer7_cbs_01"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3065_Transfer7_cbs_02"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3065_Transfer7_cbs_03"] = tgt_ref_dict["Transfer"]["set_1"]

tgt_ref_dict["Transfer_files_motions_3066_Transfer8_cbs_01"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3066_Transfer8_cbs_02"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3066_Transfer8_cbs_03"] = tgt_ref_dict["Transfer"]["set_1"]

tgt_ref_dict["Transfer_files_motions_3067_Transfer9_cbs_01"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3067_Transfer9_cbs_02"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3067_Transfer9_cbs_03"] = tgt_ref_dict["Transfer"]["set_1"]

tgt_ref_dict["Transfer_files_motions_3068_Transfer10_cbs_01"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3068_Transfer10_cbs_04"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3068_Transfer10_cbs_05"] = tgt_ref_dict["Transfer"]["set_1"]

tgt_ref_dict["Transfer_files_motions_3069_Transfer11_cbs_01"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3069_Transfer11_cbs_04"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3069_Transfer11_cbs_05"] = tgt_ref_dict["Transfer"]["set_1"]

tgt_ref_dict["Transfer_files_motions_3070_Transfer12_cbs_02"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3070_Transfer12_cbs_03"] = tgt_ref_dict["Transfer"]["set_1"]
tgt_ref_dict["Transfer_files_motions_3070_Transfer12_cbs_04"] = tgt_ref_dict["Transfer"]["set_1"]

# fine motion wrt to either cup_small or cup_large, or simply the kitchen_sideboard

tgt_ref_dict["Pour"] = {}
tgt_ref_dict["Pour"]["set_1"] = {}
tgt_ref_dict["Pour"]["set_1"]["Idle"] = {}
tgt_ref_dict["Pour"]["set_1"]["Approach"] = {}
tgt_ref_dict["Pour"]["set_1"]["Move"] = {}
tgt_ref_dict["Pour"]["set_1"]["Pour"] = {}
tgt_ref_dict["Pour"]["set_1"]["Hold"] = {}
tgt_ref_dict["Pour"]["set_1"]["Place"] = {}
tgt_ref_dict["Pour"]["set_1"]["Retreat"] = {}

tgt_ref_dict["Pour_files_motions_2967_Pour1_cs_ms_0_20l_02"] = tgt_ref_dict["Pour"]["set_1"]
tgt_ref_dict["Pour_files_motions_2967_Pour1_cs_ms_0_20l_03"] = tgt_ref_dict["Pour"]["set_1"]
tgt_ref_dict["Pour_files_motions_2967_Pour1_cs_ms_0_20l_04"] = tgt_ref_dict["Pour"]["set_1"]

tgt_ref_dict["Pour_files_motions_2968_Pour2_cl_ms_0_20l_01"] = tgt_ref_dict["Pour"]["set_1"]
tgt_ref_dict["Pour_files_motions_2968_Pour2_cl_ms_0_20l_02"] = tgt_ref_dict["Pour"]["set_1"]
tgt_ref_dict["Pour_files_motions_2968_Pour2_cl_ms_0_20l_03"] = tgt_ref_dict["Pour"]["set_1"]

tgt_ref_dict["Pour_files_motions_2969_Pour3_cl_ms_45_40l_02"] = tgt_ref_dict["Pour"]["set_1"]
tgt_ref_dict["Pour_files_motions_2969_Pour3_cl_ms_45_40l_03"] = tgt_ref_dict["Pour"]["set_1"]
tgt_ref_dict["Pour_files_motions_2969_Pour3_cl_ms_45_40l_04"] = tgt_ref_dict["Pour"]["set_1"]

tgt_ref_dict["Pour_files_motions_2970_Pour4_cl_ms_180_30l_01"] = tgt_ref_dict["Pour"]["set_1"]
tgt_ref_dict["Pour_files_motions_2970_Pour4_cl_ms_180_30l_02"] = tgt_ref_dict["Pour"]["set_1"]
tgt_ref_dict["Pour_files_motions_2970_Pour4_cl_ms_180_30l_04"] = tgt_ref_dict["Pour"]["set_1"]

tgt_ref_dict["Pour_files_motions_2971_Pour5_cl_ms_0_20h_01"] = tgt_ref_dict["Pour"]["set_1"]
tgt_ref_dict["Pour_files_motions_2971_Pour5_cl_ms_0_20h_02"] = tgt_ref_dict["Pour"]["set_1"]
tgt_ref_dict["Pour_files_motions_2971_Pour5_cl_ms_0_20h_03"] = tgt_ref_dict["Pour"]["set_1"]

tgt_ref_dict["Pour_files_motions_2972_Pour6_cl_ms_45_40h_01"] = tgt_ref_dict["Pour"]["set_1"]
tgt_ref_dict["Pour_files_motions_2972_Pour6_cl_ms_45_40h_02"] = tgt_ref_dict["Pour"]["set_1"]
tgt_ref_dict["Pour_files_motions_2972_Pour6_cl_ms_45_40h_04"] = tgt_ref_dict["Pour"]["set_1"]

tgt_ref_dict["Pour_files_motions_2973_Pour7_cs_aj_0_20l_01"] = tgt_ref_dict["Pour"]["set_1"]
tgt_ref_dict["Pour_files_motions_2973_Pour7_cs_aj_0_20l_03"] = tgt_ref_dict["Pour"]["set_1"]
tgt_ref_dict["Pour_files_motions_2973_Pour7_cs_aj_0_20l_04"] = tgt_ref_dict["Pour"]["set_1"]

tgt_ref_dict["Pour_files_motions_2974_Pour8_cl_aj_0_20l_01"] = tgt_ref_dict["Pour"]["set_1"]
tgt_ref_dict["Pour_files_motions_2974_Pour8_cl_aj_0_20l_02"] = tgt_ref_dict["Pour"]["set_1"]
tgt_ref_dict["Pour_files_motions_2974_Pour8_cl_aj_0_20l_04"] = tgt_ref_dict["Pour"]["set_1"]

tgt_ref_dict["Pour_files_motions_2975_Pour9_cl_aj_45_40l_02"] = tgt_ref_dict["Pour"]["set_1"]
tgt_ref_dict["Pour_files_motions_2975_Pour9_cl_aj_45_40l_03"] = tgt_ref_dict["Pour"]["set_1"]
tgt_ref_dict["Pour_files_motions_2975_Pour9_cl_aj_45_40l_04"] = tgt_ref_dict["Pour"]["set_1"]

tgt_ref_dict["Pour_files_motions_2976_Pour10_cl_aj_180_30l_03"] = tgt_ref_dict["Pour"]["set_1"]
tgt_ref_dict["Pour_files_motions_2976_Pour10_cl_aj_180_30l_04"] = tgt_ref_dict["Pour"]["set_1"]
tgt_ref_dict["Pour_files_motions_2976_Pour10_cl_aj_180_30l_06"] = tgt_ref_dict["Pour"]["set_1"]

tgt_ref_dict["Pour_files_motions_2978_Pour1_rc_sb_01"] = tgt_ref_dict["Pour"]["set_1"]
tgt_ref_dict["Pour_files_motions_2978_Pour1_rc_sb_02"] = tgt_ref_dict["Pour"]["set_1"]

tgt_ref_dict["Pour_files_motions_2979_Pour2_bc_sb_00"] = tgt_ref_dict["Pour"]["set_1"]
tgt_ref_dict["Pour_files_motions_2979_Pour2_bc_sb_01"] = tgt_ref_dict["Pour"]["set_1"]
tgt_ref_dict["Pour_files_motions_2979_Pour2_bc_sb_02"] = tgt_ref_dict["Pour"]["set_1"]

tgt_ref_dict["Pour_files_motions_2980_Pour3_bc_sb_00"] = tgt_ref_dict["Pour"]["set_1"]
tgt_ref_dict["Pour_files_motions_2980_Pour3_bc_sb_01"] = tgt_ref_dict["Pour"]["set_1"]
tgt_ref_dict["Pour_files_motions_2980_Pour3_bc_sb_02"] = tgt_ref_dict["Pour"]["set_1"]

tgt_ref_dict["Pour_files_motions_2981_Pour4_bc_sb_00"] = tgt_ref_dict["Pour"]["set_1"]
tgt_ref_dict["Pour_files_motions_2981_Pour4_bc_sb_01"] = tgt_ref_dict["Pour"]["set_1"]
tgt_ref_dict["Pour_files_motions_2981_Pour4_bc_sb_02"] = tgt_ref_dict["Pour"]["set_1"]

tgt_ref_dict["Pour_files_motions_2982_Pour5_bc_sb_00"] = tgt_ref_dict["Pour"]["set_1"]
tgt_ref_dict["Pour_files_motions_2982_Pour5_bc_sb_01"] = tgt_ref_dict["Pour"]["set_1"]
tgt_ref_dict["Pour_files_motions_2982_Pour5_bc_sb_02"] = tgt_ref_dict["Pour"]["set_1"]

tgt_ref_dict["Pour_files_motions_2983_Pour6_bc_sb_00"] = tgt_ref_dict["Pour"]["set_1"]
tgt_ref_dict["Pour_files_motions_2983_Pour6_bc_sb_01"] = tgt_ref_dict["Pour"]["set_1"]
tgt_ref_dict["Pour_files_motions_2983_Pour6_bc_sb_02"] = tgt_ref_dict["Pour"]["set_1"]

tgt_ref_dict["Pour_files_motions_2984_Pour8_rc_bb_00"] = tgt_ref_dict["Pour"]["set_1"]
tgt_ref_dict["Pour_files_motions_2984_Pour8_rc_bb_01"] = tgt_ref_dict["Pour"]["set_1"]
tgt_ref_dict["Pour_files_motions_2984_Pour8_rc_bb_02"] = tgt_ref_dict["Pour"]["set_1"]

tgt_ref_dict["Pour_files_motions_2985_Pour9_bc_bb_00"] = tgt_ref_dict["Pour"]["set_1"]
tgt_ref_dict["Pour_files_motions_2985_Pour9_bc_bb_01"] = tgt_ref_dict["Pour"]["set_1"]
tgt_ref_dict["Pour_files_motions_2985_Pour9_bc_bb_02"] = tgt_ref_dict["Pour"]["set_1"]

tgt_ref_dict["Pour_files_motions_2986_PourX_bc_bb_00"] = tgt_ref_dict["Pour"]["set_1"]
tgt_ref_dict["Pour_files_motions_2986_PourX_bc_bb_01"] = tgt_ref_dict["Pour"]["set_1"]
tgt_ref_dict["Pour_files_motions_2986_PourX_bc_bb_02"] = tgt_ref_dict["Pour"]["set_1"]

tgt_ref_dict["Pour_files_motions_2987_PourXI_bc_bb_00"] = tgt_ref_dict["Pour"]["set_1"]
tgt_ref_dict["Pour_files_motions_2987_PourXI_bc_bb_01"] = tgt_ref_dict["Pour"]["set_1"]
tgt_ref_dict["Pour_files_motions_2987_PourXI_bc_bb_02"] = tgt_ref_dict["Pour"]["set_1"]

multiple_interactions = ["short/1480/Pour11_cl_aj_135_40l_02"]

# for the extended dataset

action_keystate_to_oh_vector = {}
action_keystate_to_oh_vector[('Close',)] = 0
action_keystate_to_oh_vector[('Cut',)] = 1
action_keystate_to_oh_vector[('Left', 'Pick', 'egg_whisk')] = 2
action_keystate_to_oh_vector[('Left', 'Place', 'ladle')] = 3
action_keystate_to_oh_vector[('Left', 'Pick', 'cucumber_attachment')] = 4
action_keystate_to_oh_vector[('Left', 'Pick', 'rolling_pin')] = 5
action_keystate_to_oh_vector[('Left', 'Pick', 'cutting_board_small')] = 6
action_keystate_to_oh_vector[('Left', 'Pick', 'knife_black')] = 7
action_keystate_to_oh_vector[('Left', 'Place', 'sponge_small')] = 8
action_keystate_to_oh_vector[('Left', 'Pick', 'apple_juice_lid')] = 9
action_keystate_to_oh_vector[('Left', 'Place', 'cucumber_attachment')] = 10
action_keystate_to_oh_vector[('Left', 'Place', 'rolling_pin')] = 11
action_keystate_to_oh_vector[('Left', 'Place', 'broom')] = 12
action_keystate_to_oh_vector[('Left', 'Place', 'cutting_board_small')] = 13
action_keystate_to_oh_vector[('Left', 'Place', 'knife_black')] = 14
action_keystate_to_oh_vector[('Left', 'Place', 'mixing_bowl_green')] = 15
action_keystate_to_oh_vector[('Left', 'Pick', 'broom')] = 16
action_keystate_to_oh_vector[('Left', 'Place', 'apple_juice_lid')] = 17
action_keystate_to_oh_vector[('Left', 'Pick', 'mixing_bowl_green')] = 18
action_keystate_to_oh_vector[('Left', 'Pick', 'plate_dish')] = 19
action_keystate_to_oh_vector[('Left', 'Pick', 'cup_small')] = 20
action_keystate_to_oh_vector[('Left', 'Pick', 'tablespoon')] = 21
action_keystate_to_oh_vector[('Left', 'Place', 'milk_small_lid')] = 22
action_keystate_to_oh_vector[('Left', 'Pick', 'milk_small')] = 23
action_keystate_to_oh_vector[('Left', 'Place', 'peeler')] = 24
action_keystate_to_oh_vector[('Left', 'Pick', 'salad_fork')] = 25
action_keystate_to_oh_vector[('Left', 'Pick', 'mixing_bowl_green', 'Right', 'Pick', 'mixing_bowl_green')] = 26
action_keystate_to_oh_vector[('Left', 'Pick', 'salad_spoon')] = 27
action_keystate_to_oh_vector[('Left', 'Pick', 'milk_small_lid')] = 28
action_keystate_to_oh_vector[('Left', 'Pick', 'peeler')] = 29
action_keystate_to_oh_vector[('Left', 'Place', 'plate_dish')] = 30
action_keystate_to_oh_vector[('Left', 'Place', 'cup_small')] = 31
action_keystate_to_oh_vector[('Left', 'Place', 'tablespoon')] = 32
action_keystate_to_oh_vector[('Left', 'Place', 'milk_small')] = 33
action_keystate_to_oh_vector[('Left', 'Pick', 'salad_fork', 'Right', 'Pick', 'salad_spoon')] = 34
action_keystate_to_oh_vector[('Left', 'Pick', 'cup_large')] = 35
action_keystate_to_oh_vector[('Left', 'Place', 'salad_fork')] = 36
action_keystate_to_oh_vector[('Left', 'Place', 'salad_spoon')] = 37
action_keystate_to_oh_vector[('Left', 'Pick', 'apple_juice')] = 38
action_keystate_to_oh_vector[('Left', 'Pick', 'ladle')] = 39
action_keystate_to_oh_vector[('Left', 'Place', 'cup_large')] = 40
action_keystate_to_oh_vector[('Left', 'Pick', 'draining_rack')] = 41
action_keystate_to_oh_vector[('Left', 'Place', 'draining_rack')] = 42
action_keystate_to_oh_vector[('Left', 'Place', 'apple_juice')] = 43
action_keystate_to_oh_vector[('Left', 'Place', 'egg_whisk')] = 44
action_keystate_to_oh_vector[('Left', 'Pick', 'sponge_small')] = 45
action_keystate_to_oh_vector[('Mix',)] = 46
action_keystate_to_oh_vector[('Open',)] = 47
action_keystate_to_oh_vector[('Peel',)] = 48
action_keystate_to_oh_vector[('Pick', 'rolling_pin', 'rolling_pin')] = 49
action_keystate_to_oh_vector[('Pick', 'apple_juice', 'apple_juice')] = 50
action_keystate_to_oh_vector[('Pick', 'milk_small', 'milk_small')] = 51
action_keystate_to_oh_vector[('Pick', 'mixing_bowl_green', 'sponge_small')] = 52
action_keystate_to_oh_vector[('Pick', 'cutting_board_small', 'cutting_board_small')] = 53
action_keystate_to_oh_vector[('Pick', 'cup_small', 'cup_small')] = 54
action_keystate_to_oh_vector[('Pick', 'cup_small', 'tablespoon')] = 55
action_keystate_to_oh_vector[('Pick', 'salad_fork', 'salad_spoon')] = 56
action_keystate_to_oh_vector[('Pick', 'mixing_bowl_green', 'mixing_bowl_green')] = 57
action_keystate_to_oh_vector[('Place', 'cup_small', 'cup_small')] = 58
action_keystate_to_oh_vector[('Place', 'rolling_pin', 'rolling_pin')] = 59
action_keystate_to_oh_vector[('Place', 'plate_dish', 'plate_dish')] = 60
action_keystate_to_oh_vector[('Place', 'salad_spoon', 'salad_fork')] = 61
action_keystate_to_oh_vector[('Place', 'cutting_board_small', 'knife_black')] = 62
action_keystate_to_oh_vector[('Place', 'mixing_bowl_green', 'mixing_bowl_green')] = 63
action_keystate_to_oh_vector[('Place', 'cup_large', 'cup_large')] = 64
action_keystate_to_oh_vector[('Place', 'salad_fork', 'salad_spoon')] = 65
action_keystate_to_oh_vector[('Place', 'apple_juice', 'apple_juice_lid')] = 66
action_keystate_to_oh_vector[('Pour',)] = 67
action_keystate_to_oh_vector[('Right', 'Place', 'milk_small_lid')] = 68
action_keystate_to_oh_vector[('Right', 'Place', 'peeler')] = 69
action_keystate_to_oh_vector[('Right', 'Place', 'cup_large', 'Left', 'Pick', 'apple_juice')] = 70
action_keystate_to_oh_vector[('Right', 'Pick', 'ladle')] = 71
action_keystate_to_oh_vector[('Right', 'Place', 'cup_large')] = 72
action_keystate_to_oh_vector[('Right', 'Place', 'apple_juice')] = 73
action_keystate_to_oh_vector[('Right', 'Place', 'egg_whisk')] = 74
action_keystate_to_oh_vector[('Right', 'Pick', 'sponge_small')] = 75
action_keystate_to_oh_vector[('Right', 'Pick', 'draining_rack')] = 76
action_keystate_to_oh_vector[('Right', 'Pick', 'apple_juice')] = 77
action_keystate_to_oh_vector[('Right', 'Pick', 'egg_whisk')] = 78
action_keystate_to_oh_vector[('Right', 'Place', 'ladle')] = 79
action_keystate_to_oh_vector[('Right', 'Pick', 'cucumber_attachment')] = 80
action_keystate_to_oh_vector[('Right', 'Pick', 'cutting_board_small')] = 81
action_keystate_to_oh_vector[('Right', 'Pick', 'knife_black')] = 82
action_keystate_to_oh_vector[('Right', 'Place', 'sponge_small')] = 83
action_keystate_to_oh_vector[('Right', 'Place', 'draining_rack')] = 84
action_keystate_to_oh_vector[('Right', 'Pick', 'apple_juice_lid')] = 85
action_keystate_to_oh_vector[('Right', 'Place', 'cucumber_attachment')] = 86
action_keystate_to_oh_vector[('Right', 'Place', 'rolling_pin')] = 87
action_keystate_to_oh_vector[('Right', 'Place', 'broom')] = 88
action_keystate_to_oh_vector[('Right', 'Place', 'cutting_board_small')] = 89
action_keystate_to_oh_vector[('Right', 'Place', 'knife_black')] = 90
action_keystate_to_oh_vector[('Right', 'Pick', 'rolling_pin')] = 91
action_keystate_to_oh_vector[('Right', 'Pick', 'broom')] = 92
action_keystate_to_oh_vector[('Right', 'Place', 'apple_juice_lid')] = 93
action_keystate_to_oh_vector[('Right', 'Pick', 'mixing_bowl_green')] = 94
action_keystate_to_oh_vector[('Right', 'Pick', 'plate_dish')] = 95
action_keystate_to_oh_vector[('Right', 'Pick', 'cup_small')] = 96
action_keystate_to_oh_vector[('Right', 'Pick', 'tablespoon')] = 97
action_keystate_to_oh_vector[('Right', 'Pick', 'milk_small')] = 98
action_keystate_to_oh_vector[('Right', 'Pick', 'salad_fork')] = 99
action_keystate_to_oh_vector[('Right', 'Pick', 'salad_spoon')] = 100
action_keystate_to_oh_vector[('Right', 'Pick', 'milk_small_lid')] = 101
action_keystate_to_oh_vector[('Right', 'Pick', 'peeler')] = 102
action_keystate_to_oh_vector[('Right', 'Place', 'mixing_bowl_green')] = 103
action_keystate_to_oh_vector[('Right', 'Place', 'plate_dish')] = 104
action_keystate_to_oh_vector[('Right', 'Place', 'cup_small')] = 105
action_keystate_to_oh_vector[('Right', 'Place', 'tablespoon')] = 106
action_keystate_to_oh_vector[('Right', 'Place', 'milk_small')] = 107
action_keystate_to_oh_vector[('Right', 'Pick', 'cup_large')] = 108
action_keystate_to_oh_vector[('Right', 'Place', 'salad_fork')] = 109
action_keystate_to_oh_vector[('Right', 'Place', 'salad_spoon')] = 110
action_keystate_to_oh_vector[('RollOut',)] = 111
action_keystate_to_oh_vector[('Scoop',)] = 112
action_keystate_to_oh_vector[('Shake',)] = 113
action_keystate_to_oh_vector[('Stir',)] = 114
action_keystate_to_oh_vector[('Sweep',)] = 115
action_keystate_to_oh_vector[('Transfer',)] = 116
action_keystate_to_oh_vector[('Wipe',)] = 117

for k,v in action_keystate_to_oh_vector.items():
    action_keystate_to_oh_vector[k] = np.squeeze(one_hot(np.array(v),len(action_keystate_to_oh_vector)))
    
oh_vector_to_action_keystate = {}
oh_vector_to_action_keystate[0] = ('Close',)
oh_vector_to_action_keystate[1] = ('Cut',)
oh_vector_to_action_keystate[2] = ('Left', 'Pick', 'egg_whisk')
oh_vector_to_action_keystate[3] = ('Left', 'Place', 'ladle')
oh_vector_to_action_keystate[4] = ('Left', 'Pick', 'cucumber_attachment')
oh_vector_to_action_keystate[5] = ('Left', 'Pick', 'rolling_pin')
oh_vector_to_action_keystate[6] = ('Left', 'Pick', 'cutting_board_small')
oh_vector_to_action_keystate[7] = ('Left', 'Pick', 'knife_black')
oh_vector_to_action_keystate[8] = ('Left', 'Place', 'sponge_small')
oh_vector_to_action_keystate[9] = ('Left', 'Pick', 'apple_juice_lid')
oh_vector_to_action_keystate[10] = ('Left', 'Place', 'cucumber_attachment')
oh_vector_to_action_keystate[11] = ('Left', 'Place', 'rolling_pin')
oh_vector_to_action_keystate[12] = ('Left', 'Place', 'broom')
oh_vector_to_action_keystate[13] = ('Left', 'Place', 'cutting_board_small')
oh_vector_to_action_keystate[14] = ('Left', 'Place', 'knife_black')
oh_vector_to_action_keystate[15] = ('Left', 'Place', 'mixing_bowl_green')
oh_vector_to_action_keystate[16] = ('Left', 'Pick', 'broom')
oh_vector_to_action_keystate[17] = ('Left', 'Place', 'apple_juice_lid')
oh_vector_to_action_keystate[18] = ('Left', 'Pick', 'mixing_bowl_green')
oh_vector_to_action_keystate[19] = ('Left', 'Pick', 'plate_dish')
oh_vector_to_action_keystate[20] = ('Left', 'Pick', 'cup_small')
oh_vector_to_action_keystate[21] = ('Left', 'Pick', 'tablespoon')
oh_vector_to_action_keystate[22] = ('Left', 'Place', 'milk_small_lid')
oh_vector_to_action_keystate[23] = ('Left', 'Pick', 'milk_small')
oh_vector_to_action_keystate[24] = ('Left', 'Place', 'peeler')
oh_vector_to_action_keystate[25] = ('Left', 'Pick', 'salad_fork')
oh_vector_to_action_keystate[26] = ('Left', 'Pick', 'mixing_bowl_green', 'Right', 'Pick', 'mixing_bowl_green')
oh_vector_to_action_keystate[27] = ('Left', 'Pick', 'salad_spoon')
oh_vector_to_action_keystate[28] = ('Left', 'Pick', 'milk_small_lid')
oh_vector_to_action_keystate[29] = ('Left', 'Pick', 'peeler')
oh_vector_to_action_keystate[30] = ('Left', 'Place', 'plate_dish')
oh_vector_to_action_keystate[31] = ('Left', 'Place', 'cup_small')
oh_vector_to_action_keystate[32] = ('Left', 'Place', 'tablespoon')
oh_vector_to_action_keystate[33] = ('Left', 'Place', 'milk_small')
oh_vector_to_action_keystate[34] = ('Left', 'Pick', 'salad_fork', 'Right', 'Pick', 'salad_spoon')
oh_vector_to_action_keystate[35] = ('Left', 'Pick', 'cup_large')
oh_vector_to_action_keystate[36] = ('Left', 'Place', 'salad_fork')
oh_vector_to_action_keystate[37] = ('Left', 'Place', 'salad_spoon')
oh_vector_to_action_keystate[38] = ('Left', 'Pick', 'apple_juice')
oh_vector_to_action_keystate[39] = ('Left', 'Pick', 'ladle')
oh_vector_to_action_keystate[40] = ('Left', 'Place', 'cup_large')
oh_vector_to_action_keystate[41] = ('Left', 'Pick', 'draining_rack')
oh_vector_to_action_keystate[42] = ('Left', 'Place', 'draining_rack')
oh_vector_to_action_keystate[43] = ('Left', 'Place', 'apple_juice')
oh_vector_to_action_keystate[44] = ('Left', 'Place', 'egg_whisk')
oh_vector_to_action_keystate[45] = ('Left', 'Pick', 'sponge_small')
oh_vector_to_action_keystate[46] = ('Mix',)
oh_vector_to_action_keystate[47] = ('Open',)
oh_vector_to_action_keystate[48] = ('Peel',)
oh_vector_to_action_keystate[49] = ('Pick', 'rolling_pin', 'rolling_pin')
oh_vector_to_action_keystate[50] = ('Pick', 'apple_juice', 'apple_juice')
oh_vector_to_action_keystate[51] = ('Pick', 'milk_small', 'milk_small')
oh_vector_to_action_keystate[52] = ('Pick', 'mixing_bowl_green', 'sponge_small')
oh_vector_to_action_keystate[53] = ('Pick', 'cutting_board_small', 'cutting_board_small')
oh_vector_to_action_keystate[54] = ('Pick', 'cup_small', 'cup_small')
oh_vector_to_action_keystate[55] = ('Pick', 'cup_small', 'tablespoon')
oh_vector_to_action_keystate[56] = ('Pick', 'salad_fork', 'salad_spoon')
oh_vector_to_action_keystate[57] = ('Pick', 'mixing_bowl_green', 'mixing_bowl_green')
oh_vector_to_action_keystate[58] = ('Place', 'cup_small', 'cup_small')
oh_vector_to_action_keystate[59] = ('Place', 'rolling_pin', 'rolling_pin')
oh_vector_to_action_keystate[60] = ('Place', 'plate_dish', 'plate_dish')
oh_vector_to_action_keystate[61] = ('Place', 'salad_spoon', 'salad_fork')
oh_vector_to_action_keystate[62] = ('Place', 'cutting_board_small', 'knife_black')
oh_vector_to_action_keystate[63] = ('Place', 'mixing_bowl_green', 'mixing_bowl_green')
oh_vector_to_action_keystate[64] = ('Place', 'cup_large', 'cup_large')
oh_vector_to_action_keystate[65] = ('Place', 'salad_fork', 'salad_spoon')
oh_vector_to_action_keystate[66] = ('Place', 'apple_juice', 'apple_juice_lid')
oh_vector_to_action_keystate[67] = ('Pour',)
oh_vector_to_action_keystate[68] = ('Right', 'Place', 'milk_small_lid')
oh_vector_to_action_keystate[69] = ('Right', 'Place', 'peeler')
oh_vector_to_action_keystate[70] = ('Right', 'Place', 'cup_large', 'Left', 'Pick', 'apple_juice')
oh_vector_to_action_keystate[71] = ('Right', 'Pick', 'ladle')
oh_vector_to_action_keystate[72] = ('Right', 'Place', 'cup_large')
oh_vector_to_action_keystate[73] = ('Right', 'Place', 'apple_juice')
oh_vector_to_action_keystate[74] = ('Right', 'Place', 'egg_whisk')
oh_vector_to_action_keystate[75] = ('Right', 'Pick', 'sponge_small')
oh_vector_to_action_keystate[76] = ('Right', 'Pick', 'draining_rack')
oh_vector_to_action_keystate[77] = ('Right', 'Pick', 'apple_juice')
oh_vector_to_action_keystate[78] = ('Right', 'Pick', 'egg_whisk')
oh_vector_to_action_keystate[79] = ('Right', 'Place', 'ladle')
oh_vector_to_action_keystate[80] = ('Right', 'Pick', 'cucumber_attachment')
oh_vector_to_action_keystate[81] = ('Right', 'Pick', 'cutting_board_small')
oh_vector_to_action_keystate[82] = ('Right', 'Pick', 'knife_black')
oh_vector_to_action_keystate[83] = ('Right', 'Place', 'sponge_small')
oh_vector_to_action_keystate[84] = ('Right', 'Place', 'draining_rack')
oh_vector_to_action_keystate[85] = ('Right', 'Pick', 'apple_juice_lid')
oh_vector_to_action_keystate[86] = ('Right', 'Place', 'cucumber_attachment')
oh_vector_to_action_keystate[87] = ('Right', 'Place', 'rolling_pin')
oh_vector_to_action_keystate[88] = ('Right', 'Place', 'broom')
oh_vector_to_action_keystate[89] = ('Right', 'Place', 'cutting_board_small')
oh_vector_to_action_keystate[90] = ('Right', 'Place', 'knife_black')
oh_vector_to_action_keystate[91] = ('Right', 'Pick', 'rolling_pin')
oh_vector_to_action_keystate[92] = ('Right', 'Pick', 'broom')
oh_vector_to_action_keystate[93] = ('Right', 'Place', 'apple_juice_lid')
oh_vector_to_action_keystate[94] = ('Right', 'Pick', 'mixing_bowl_green')
oh_vector_to_action_keystate[95] = ('Right', 'Pick', 'plate_dish')
oh_vector_to_action_keystate[96] = ('Right', 'Pick', 'cup_small')
oh_vector_to_action_keystate[97] = ('Right', 'Pick', 'tablespoon')
oh_vector_to_action_keystate[98] = ('Right', 'Pick', 'milk_small')
oh_vector_to_action_keystate[99] = ('Right', 'Pick', 'salad_fork')
oh_vector_to_action_keystate[100] = ('Right', 'Pick', 'salad_spoon')
oh_vector_to_action_keystate[101] = ('Right', 'Pick', 'milk_small_lid')
oh_vector_to_action_keystate[102] = ('Right', 'Pick', 'peeler')
oh_vector_to_action_keystate[103] = ('Right', 'Place', 'mixing_bowl_green')
oh_vector_to_action_keystate[104] = ('Right', 'Place', 'plate_dish')
oh_vector_to_action_keystate[105] = ('Right', 'Place', 'cup_small')
oh_vector_to_action_keystate[106] = ('Right', 'Place', 'tablespoon')
oh_vector_to_action_keystate[107] = ('Right', 'Place', 'milk_small')
oh_vector_to_action_keystate[108] = ('Right', 'Pick', 'cup_large')
oh_vector_to_action_keystate[109] = ('Right', 'Place', 'salad_fork')
oh_vector_to_action_keystate[110] = ('Right', 'Place', 'salad_spoon')
oh_vector_to_action_keystate[111] = ('RollOut',)
oh_vector_to_action_keystate[112] = ('Scoop',)
oh_vector_to_action_keystate[113] = ('Shake',)
oh_vector_to_action_keystate[114] = ('Stir',)
oh_vector_to_action_keystate[115] = ('Sweep',)
oh_vector_to_action_keystate[116] = ('Transfer',)
oh_vector_to_action_keystate[117] = ('Wipe',)