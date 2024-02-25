import numpy as np

skip = ["Pour/files_motions_2977", "Pour/files_motions_2988"]

"""
joint_names
- joint names in the order   
"""

# joint_names stored in order in the xml file
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

root_name   = ["ROOTx_joint", "ROOTy_joint", "ROOTz_joint"]
CS_name     = ["CSx_joint",   "CSy_joint",   "CSz_joint"]
extended_joint_names     = joint_names + root_name + CS_name

extended_joint_axis = {}
for joint_name in extended_joint_names:
    if joint_name.count("x") > 1 or joint_name.count("y") > 1 or joint_name.count("z") > 1:
        print("Error in joint_name.count()")
        sys.exit()
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
 
"""
link order
- link order as described in
  https://ieeexplore.ieee.org/document/7506114 
  Unifying Representations and Large-Scale Whole-Body Motion Databases for Studying Human Motion
"""

               # root
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
               # 1) LS
               ["CSz_joint","RSx_joint"],   ["RSx_joint","RSy_joint"], ["RSy_joint","RSz_joint"],
               # 2) LE
               ["RSz_joint","REx_joint"],   ["REx_joint","REz_joint"],
               # 3) LW
               ["REz_joint","RWx_joint"],   ["RWx_joint","RWy_joint"],
 
               # bottom left half
               # 1) LH
               ["ROOTz_joint","LHx_joint"], ["LHx_joint","LHy_joint"], ["LHy_joint","LHz_joint"],
               # 2) LK
               ["LHz_joint","LKx_joint"],
               # 3) LA
               ["LKx_joint","LAx_joint"],   ["LAx_joint","LAy_joint"], ["LAy_joint","LAz_joint"],

               # bottom right half
               # 1) LH
               ["ROOTz_joint","RHx_joint"], ["RHx_joint","RHy_joint"], ["RHy_joint","RHz_joint"],
               # 2) LK
               ["RHz_joint","RKx_joint"],
               # 3) LA
               ["RKx_joint","RAx_joint"],   ["RAx_joint","RAy_joint"], ["RAy_joint","RAz_joint"]]
 
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


"""
link direction
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

"""
link_length_dict
"""

"""
link direction
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

"""
converterConfigFile

"""

# TTD
# - cucumber_cut
# - cucumber_peel

# model file
outputModelFile = {"1480":"/home/haziq/MMMTools/data/Model/Winter/mmm.xml",
                   "1717":"/home/haziq/MMMTools/data/Model/Winter/mmm.xml",
                   "1723":"/home/haziq/MMMTools/data/Model/Winter/mmm.xml",
                   "1747":"/home/haziq/MMMTools/data/Model/Winter/mmm.xml",
                   "3322":"/home/haziq/MMMTools/data/Model/Winter/mmm.xml",
                   "3323":"/home/haziq/MMMTools/data/Model/Winter/mmm.xml"}
objectModelFile = {"egg_whisk":"/home/haziq/MMMTools/data/Model/Objects/egg_whisk/egg_whisk.xml",
                   "apple_juice":"/home/haziq/MMMTools/data/Model/Objects/apple_juice/apple_juice.xml",
                   "apple_juice_lid":"/home/haziq/MMMTools/data/Model/Objects/apple_juice_lid/apple_juice_lid.xml",
                   "milk_small":"/home/haziq/MMMTools/data/Model/Objects/milk_small/milk_small.xml",
                   "milk_small_lid":"/home/haziq/MMMTools/data/Model/Objects/milk_small_lid/milk_small_lid.xml",
                   "ladle":"/home/haziq/MMMTools/data/Model/Objects/ladle/ladle.xml",
                   "mixing_bowl_small":"/home/haziq/MMMTools/data/Model/Objects/mixing_bowl_small/mixing_bowl_small.xml",
                   "mixing_bowl_green":"/home/haziq/MMMTools/data/Model/Objects/mixing_bowl_green/mixing_bowl_green.xml",
                   "salad_fork":"/home/haziq/MMMTools/data/Model/Objects/salad_fork/salad_fork.xml",
                   "rolling_pin":"/home/haziq/MMMTools/data/Model/Objects/rolling_pin/rolling_pin.xml",
                   "kitchen_sideboard":"/home/haziq/MMMTools/data/Model/Objects/kitchen_sideboard/kitchen_sideboard.xml",
                   "draining_rack":"/home/haziq/MMMTools/data/Model/Objects/draining_rack/draining_rack.xml",
                   "plate_dish":"/home/haziq/MMMTools/data/Model/Objects/plate_dish/plate_dish.xml",
                   "sponge_small":"/home/haziq/MMMTools/data/Model/Objects/sponge_small/sponge_small.xml",
                   "salad_spoon":"/home/haziq/MMMTools/data/Model/Objects/salad_spoon/salad_spoon.xml",
                   "cucumber_attachment":"/home/haziq/MMMTools/data/Model/Objects/cucumber_attachment/cucumber_cut.xml",
                   "cutting_board_small":"/home/haziq/MMMTools/data/Model/Objects/cutting_board_small/cutting_board_small.xml",
                   "knife_black":"/home/haziq/MMMTools/data/Model/Objects/knife_black/knife_black.xml",
                   "peeler":"/home/haziq/MMMTools/data/Model/Objects/peeler/peeler.xml",
                   "cup_large":"/home/haziq/MMMTools/data/Model/Objects/cup_large/cup_large.xml",
                   "cup_small":"/home/haziq/MMMTools/data/Model/Objects/cup_small/cup_small.xml",
                   "tablespoon":"/home/haziq/MMMTools/data/Model/Objects/tablespoon/tablespoon.xml"}

# physical markers defined in motion.xml -> physical markers defind in mmm.xml, mixing_bowl_green.xml, ...
converterConfigFile = {"1480":"/home/haziq/MMMTools/data/Model/Winter/NloptConverterVicon2MMM_WinterConfig.xml",
                       "1717":"/home/haziq/MMMTools/data/Model/Winter/NloptConverterVicon2MMM_WinterConfig.xml",
                       "1723":"/home/haziq/MMMTools/data/Model/Winter/NloptConverterVicon2MMM_WinterConfig.xml",
                       "1747":"/home/haziq/MMMTools/data/Model/Winter/NloptConverterVicon2MMM_WinterConfig.xml",
                       "3322":"/home/haziq/MMMTools/data/Model/Winter/NloptConverterVicon2MMM_WinterConfig.xml",
                       "3323":"/home/haziq/MMMTools/data/Model/Winter/NloptConverterVicon2MMM_WinterConfig.xml",
                       "egg_whisk":"/home/haziq/MMMTools/data/Model/Objects/egg_whisk/NloptConverterVicon2MMM_EggWhiskConfig.xml",
                       "apple_juice":"/home/haziq/MMMTools/data/Model/Objects/apple_juice/NloptConverterVicon2MMM_AppleJuiceConfig.xml",
                       "apple_juice_lid":"/home/haziq/MMMTools/data/Model/Objects/apple_juice_lid/NloptConverterVicon2MMM_AppleJuiceLidConfig.xml",
                       "milk_small":"/home/haziq/MMMTools/data/Model/Objects/milk_small/NloptConverterVicon2MMM_MilkSmallConfig.xml",
                       "milk_small_lid":"/home/haziq/MMMTools/data/Model/Objects/milk_small_lid/NloptConverterVicon2MMM_MilkSmallLidConfig.xml",
                       "ladle":"/home/haziq/MMMTools/data/Model/Objects/ladle/NloptConverterVicon2MMM_LadleConfig.xml",
                       "mixing_bowl_small":"/home/haziq/MMMTools/data/Model/Objects/mixing_bowl_small/NloptConverterVicon2MMM_MixingBowlSmallConfig.xml",
                       "mixing_bowl_green":"/home/haziq/MMMTools/data/Model/Objects/mixing_bowl_green/NloptConverterVicon2MMM_MixingBowlGreenConfig.xml",
                       "salad_fork":"/home/haziq/MMMTools/data/Model/Objects/salad_fork/NloptConverterVicon2MMM_SaladForkConfig.xml",
                       "rolling_pin":"/home/haziq/MMMTools/data/Model/Objects/rolling_pin/NloptConverterVicon2MMM_RollingPinConfig.xml",
                       "kitchen_sideboard":"/home/haziq/MMMTools/data/Model/Objects/kitchen_sideboard/NloptConverterVicon2MMM_KitchenSideboardConfig.xml",
                       "draining_rack":"/home/haziq/MMMTools/data/Model/Objects/draining_rack/NloptConverterVicon2MMM_DrainingRackConfig.xml",
                       "plate_dish":"/home/haziq/MMMTools/data/Model/Objects/plate_dish/NloptConverterVicon2MMM_PlateDishConfig.xml",
                       "sponge_small":"/home/haziq/MMMTools/data/Model/Objects/sponge_small/NloptConverterVicon2MMM_SpongeSmallConfig.xml",
                       "salad_spoon":"/home/haziq/MMMTools/data/Model/Objects/salad_spoon/NloptConverterVicon2MMM_SaladSpoonConfig.xml",
                       "cucumber_attachment":"/home/haziq/MMMTools/data/Model/Objects/cucumber_attachment/NloptConverterVicon2MMM_CucumberCutConfig.xml",
                       "cutting_board_small":"/home/haziq/MMMTools/data/Model/Objects/cutting_board_small/NloptConverterVicon2MMM_CuttingBoardSmallConfig.xml",
                       "knife_black":"/home/haziq/MMMTools/data/Model/Objects/knife_black/NloptConverterVicon2MMM_KnifeBlackConfig.xml",
                       "peeler":"/home/haziq/MMMTools/data/Model/Objects/peeler/NloptConverterVicon2MMM_PeelerConfig.xml",
                       "cup_large":"/home/haziq/MMMTools/data/Model/Objects/cup_large/NloptConverterVicon2MMM_CupLargeConfig.xml",
                       "cup_small":"/home/haziq/MMMTools/data/Model/Objects/cup_small/NloptConverterVicon2MMM_CupSmallConfig.xml",
                       "tablespoon":"/home/haziq/MMMTools/data/Model/Objects/tablespoon/NloptConverterVicon2MMM_TablespoonConfig.xml",
                       "broom":"/home/haziq/MMMTools/data/Model/Objects/broom/NloptConverterVicon2MMM_BroomConfig.xml"}

# subject height, mass, and hand_length
outputModelProcessorConfigFile = {"1480":"/home/haziq/MMMTools/data/Model/Winter/config/1480.xml",
                                  "1717":"/home/haziq/MMMTools/data/Model/Winter/config/1717.xml",
                                  "1723":"/home/haziq/MMMTools/data/Model/Winter/config/1723.xml",
                                  "1747":"/home/haziq/MMMTools/data/Model/Winter/config/1747.xml",
                                  "3322":"/home/haziq/MMMTools/data/Model/Winter/config/3322.xml",
                                  "3323":"/home/haziq/MMMTools/data/Model/Winter/config/3323.xml"}

# for true and val
outputModelFile_dict = {**{"true_"+k:v for k,v in outputModelFile.items()},**{"pred_"+k:v for k,v in outputModelFile.items()}}
converterConfigFile_dict = {**{"true_"+k:v for k,v in converterConfigFile.items()},**{"pred_"+k:v for k,v in converterConfigFile.items()}}
outputModelProcessorConfigFile_dict = {**{"true_"+k:v for k,v in outputModelProcessorConfigFile.items()},**{"pred_"+k:v for k,v in outputModelProcessorConfigFile.items()}}

# list of all relevant objects
# - this is useful when running 
#   object_nodes = root.find("Motion[@name='"+x+"']") for x in self.all_objects                  
#   as it helps exclude the human motion, or the azure kinect cameras, etc
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
                 "kitchen_sideboard"
                ]
                
# action_action_to_color [R,G,B]
object_to_color = {None:(127,111,101),     
                   "cucumber_attachment":(0,128,0),
                   "peeler":(162,187,214), 
                   "mixing_bowl_green":(255,175,0),       
                   "knife_black":(193,0,31),
                   "egg_whisk":(250,105,0),      
                   "cup_small":(208,159,98),   
                   "salad_fork":(0,82,140),        
                   "salad_spoon":(0,125,52),       
                   "rolling_pin":(52,69,1),         
                   "milk_small":(0,255,249),       
                   "cup_large":(53,68,1),     
                   "apple_juice":(3,0,254),         
                   "ladle":(249,128,188),      
                   "tablespoon":(95,0,41),        
                   "cutting_board_small":(0,1,68),           
                   "plate_dish":(158,148,1),     
                   "mixing_bowl_small":(255,0,254),         
                   "sponge_small":(153,153,153)} 