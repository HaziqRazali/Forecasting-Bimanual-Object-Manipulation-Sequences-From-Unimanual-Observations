import os
os.environ['MPLCONFIGDIR'] = os.path.join(os.path.expanduser("~"),"tmp")
import re
import cv2
import sys
import time
import json
import torch
import socket
import logging
import argparse
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

#from torch.nn.functional import cross_entropy, mse_loss, l1_loss

from glob import glob
from pathlib import Path
from collections import OrderedDict, Counter
from tensorboardX import SummaryWriter
from misc.misc import *
from misc.losses import *
from dataloaders.utils import dict_arr_to_list

torch.manual_seed(1337)

# Import for args
##################################################### 
parser = argparse.ArgumentParser()
parser.add_argument('--args', required=True, type=str)
parser.add_argument('--config_file', required=True, type=str)
args, unknown = parser.parse_known_args()
args_import = "from {} import *".format(args.args)
exec(args_import)
args = argparser(args)

# Imports for Architecture, Data Loader 
##################################################### 
architecture_import = "from {} import *".format(args.architecture)
exec(architecture_import)
data_loader_import = "from {} import *".format(args.data_loader)
exec(data_loader_import)
                        
# Prepare Data Loaders
##################################################### 
bool_dtype, long_dtype, float_dtype, _ = get_dtypes()
# load data
va_data = dataloader(args, args.test_dtype)
# data loader
va_loader = torch.utils.data.DataLoader(va_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True, pin_memory=torch.cuda.is_available())
  
# Prepare Dump File
#####################################################
dump = [None]*int((len(va_data) - len(va_data)%args.batch_size))

# Prepare Network and Optimizers
##################################################### 
net = model(args)
print("Total # of parameters: ", count_parameters(net))
print("===========================================================")
print()
net.type(float_dtype)

# Load weights and initialize checkpoints
#####################################################
if args.custom_loader == 0:
    print("Attempting to load from: " + os.path.join(args.weight_root,args.experiment_name))
    print("===========================================================")
    print()
    if os.path.join(args.weight_root,args.experiment_name) is not None and os.path.isdir(os.path.join(args.weight_root,args.experiment_name)):
        
        # to make sure all named modules are loaded
        #for name, layer in net.named_modules():
        #    print(name)
        #sys.exit()
        
        # load the best epoch for each task
        for epoch_name,layer_names,task_name in zip(args.checkpoint_epoch_names,args.checkpoint_layer_names,args.checkpoint_task_names):
            
            print("===========================================================")
            print("task_name = {} epoch_name = {}".format(task_name,epoch_name))
            print("layer_names = {}".format(layer_names))
            print("-----------------------------------------------------------")
            
            # load best
            if epoch_name == -1:
                #pt_files = glob(os.path.join(args.weight_root,args.experiment_name,"*"))
                pt_files = os.listdir(os.path.join(args.weight_root,args.experiment_name))
                pt_files = sorted([x for x in pt_files if task_name == re.sub("_epoch_\d\d\d\d_best_\d\d\d\d.pt","",x)])
                
                #for x in pt_files:
                #    print(x)
                #    x = re.sub("_epoch_\d\d\d\d_best_\d\d\d\d.pt","",x)
                #    print(x)
                #    print()
                #sys.exit()
                
                # i gotta remove _epoch_XXXX_best_XXXX.pt
                epoch_name = os.path.basename(pt_files[-1])
            
            # load closest
            if type(epoch_name) == type(1337):
                #pt_files = glob(os.path.join(args.weight_root,args.experiment_name,"*"))
                pt_files = os.listdir(os.path.join(args.weight_root,args.experiment_name))
                pt_files = sorted([x for x in pt_files if task_name in x])
                int_pt_files = [int(re.search('best_(.*?).pt', pt_file).group(1)) for pt_file in pt_files]
                epoch_name  = pt_files[int_pt_files.index(min(int_pt_files, key=lambda x:abs(x-epoch_name)))]
                epoch_name  = os.path.basename(epoch_name)
            
            print("Attempting to load from: " + os.path.join(args.weight_root,args.experiment_name,epoch_name), "into", layer_names)
            print("-----------------------------------------------------------")
        
            # load checkpoint dictionary
            checkpoint = torch.load(os.path.join(args.weight_root,args.experiment_name,epoch_name))
                    
            # load weights
            model_state = checkpoint["model_state"]
            print("model_state.items()")
            print("These are the weights in .pt file")
            for k,v in model_state.items():
                print(k)
            print("-----------------------------------------------------------")
            print()

            # not good because "encoder" will also grab "hand_encoder" or "object_encoder" etc
            #layer_dict = {k:v for k,v in model_state.items() for layer_name in layer_names if layer_name in k}       

            # not good if the layer name is e.g. main.hand_encoder
            #layer_dict = {k:v for k,v in model_state.items() for layer_name in layer_names if layer_name == k.split(".")[0]}  
            
            # better
            layer_dict = {}
            for k,v in model_state.items():
                for layer_name in layer_names:
                    k_split  = k.split(".")         # [forecasting_module, decoder_esm, grab_net_finger_decoder, decoder, 1, 2, bias]
                    k_merged = [""]*len(k_split)
                    k_merged[0] = k_split[0]
                    for i in range(1,len(k_split)):
                        k_merged[i] = k_merged[i-1] + "." + k_split[i]
                    if any([layer_name == k for k in k_merged]):
                        layer_dict[k] = v
                        break
            
            """layer_dict = {}
            for k,v in model_state.items():
                for layer_name in layer_names:
                    k = k.split(".")[0]
                    if layer_name == k:
                        layer_dict[k] = v
                        break
                    print(k)
                    sys.exit()"""        
            print("-----------------------------------------------------------")
            print("layer_dict.keys()")
            print("These are the weights successfully transferred from my .pt file")
            for key in layer_dict.keys():
                print(key)
            print("-----------------------------------------------------------")
            print("===========================================================")
            print()
            net.load_state_dict(layer_dict,strict=args.strict)
        
        print("Model Loaded")
        
    else:
        print(os.path.join(args.weight_root,args.experiment_name) + " not found")
        sys.exit() 
        
# Main Loop
####################################################  
def loop(net, inp_data, optimizer, counter, args, mode):        
    # {'human_joints_t0':human_joints_t0, 'human_joints_t1':human_joints_t1, 'object_data':object_data, "key_object":key_object, "frame":frame, "key_frame":key_frame}
    
    assert mode == "val"

    # move to gpu
    for k,v in inp_data.items():
        inp_data[k] = inp_data[k].cuda() if torch.cuda.device_count() > 0 and type(v) != type([]) else inp_data[k]
        
    # Forward pass
    t1 = time.time()
    out_data = net(inp_data, mode=mode)
    t2 = time.time()
    print("Foward Pass Time: ", 1/(t2-t1), flush=True) 

    # move all to cpu numpy
    #losses = iterdict(losses)
    inp_data = iterdict(inp_data)
    out_data = iterdict(out_data)
    
    return {"out_data":out_data}
    
# save results 
####################################################  
def save(out, inp, args):
        
    # handle conflicting names
    keys = set(inp.keys()) | set(out.keys())
    for key in keys:
        if key in inp and key in out:
            inp["_".join(["true",key])] = inp[key]
            out["_".join(["pred",key])] = out[key]
            del inp[key]
            del out[key]
    
    # merge dict
    data = {**inp, **out}
        
    # remove items i do not want
    data = {k:v for k,v in data.items() if all([x not in k for x in args.remove])}
        
    # json can only save list
    for k,v in data.items():
        data[k] = data[k].tolist() if isinstance(v, type(np.array([0]))) else data[k]
    #    print(type(v), type(data[k])) # either numpy array or list
    #sys.exit()
    
    # save each frame
    for i in range(len(data["sequence"])):
        
        # create folder
        foldername = os.path.join(args.result_root,args.result_name,data["sequence"][i]) # "/tmp/haziq/datasets/mogaze/humoro/results/" 
        path = Path(foldername)
        path.mkdir(parents=True,exist_ok=True)
                               
        # save filename
        #print(args.result_root, model, data["sequence"][i], str(int(data["inp_frame"][i])).zfill(10)+".json")
        filename = os.path.join(args.result_root,args.result_name,data["sequence"][i],str(int(data["inp_frame"][i])).zfill(10)+".json") # "/tmp/haziq/datasets/mogaze/humoro/results/"
                                
        # create json for each frame 
        data_i = {k:v[i] if type(v) == type([]) else v for k,v in data.items()}
        
        # also store args
        args_dict = vars(args)
        args_dict = dict_arr_to_list(args_dict)
        #for key, value in args.__dict__.items():
        #    setattr(args_dict, key, value)
        #for k,v in args_dict.items():
        #    print(k, v)
        #sys.exit()
        for k,v in args_dict.items():
            if k not in data_i.keys():
                data_i[k] = v
        
        #data_i = {k:v[i] if type(v) == type(np.array([0])) else v for k,v in data.items()}
        #for k,v in data_i.items():
        #    print(k, type(v))
        #print(data_i["sequence"])
        #sys.exit()
        
        #print(np.array(data_i["true_obj_xyz"]).shape)
        #print(np.array(data_i["pred_obj_xyz"]).shape)
        
        # get all keys with unpadded_length
        variables_to_unpad = [k.replace("_unpadded_length","") for k in data_i.keys() if "unpadded_length" in k]
        for k in variables_to_unpad:
            
            # unpad true
            if any(["true_"+k == data_i_key for data_i_key in data_i.keys()]):
                data_i["true_"+k] = data_i["true_"+k][:data_i[k+"_unpadded_length"]]

            # unpad pred
            if any(["pred_"+k == data_i_key for data_i_key in data_i.keys()]):
                data_i["pred_"+k] = data_i["pred_"+k][:data_i[k+"_unpadded_length"]]
            
            # unpad
            if any([k == data_i_key for data_i_key in data_i.keys()]):
                #print(k)
                #print(data_i[k+"_unpadded_length"])
                #print(k, data_i[k+"_unpadded_length"])
                data_i[k] = data_i[k][:data_i[k+"_unpadded_length"]]   
                
        # get all keys with unpadded_objects
        variables_to_unpad = [k.replace("_unpadded_objects","") for k in data_i.keys() if "unpadded_objects" in k]
        for k in variables_to_unpad:
            
            # unpad true
            if any(["true_"+k == data_i_key for data_i_key in data_i.keys()]):
                data_i["true_"+k] = np.array(data_i["true_"+k])[:,:data_i[k+"_unpadded_objects"]].tolist()

            # unpad pred
            if any(["pred_"+k == data_i_key for data_i_key in data_i.keys()]):
                data_i["pred_"+k] = np.array(data_i["pred_"+k])[:,:data_i[k+"_unpadded_objects"]].tolist()
            
            # unpad
            if any([k == data_i_key for data_i_key in data_i.keys()]):
                data_i[k] = np.array(data_i[k])[:,:data_i[k+"_unpadded_objects"]].tolist()
                        
        #for k in data_i.keys():
        #    print(k,type(data_i[k]))
        #sys.exit()
        
        #sys.exit()
        # write to json
        with open(filename, 'w') as f:
            json.dump(data_i, f)
 
# validation
####################################################  
with torch.no_grad():        
    net.eval()
    va_losses = {}
    for batch_idx, va_data in enumerate(va_loader):
    
        print("Validation batch ", batch_idx, " of ", len(va_loader))
        
        # forward pass
        va_output = loop(net=net,inp_data=va_data,optimizer=None,counter=None,args=args,mode="val")
                
        # save results
        save(va_output["out_data"], va_data, args)
