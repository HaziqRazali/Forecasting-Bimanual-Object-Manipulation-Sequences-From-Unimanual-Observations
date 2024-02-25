import os
os.environ['MPLCONFIGDIR'] = os.path.join(os.path.expanduser("~"),"tmp")
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

#from torch.nn.functional import cross_entropy, mse_loss, binary_cross_entropy

from pathlib import Path
from collections import OrderedDict, Counter
from tensorboardX import SummaryWriter
from misc.misc import *
from misc.losses import *

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

# Prepare Logging
##################################################### 
# loss plots
datetime = time.strftime("%c")  
writer = SummaryWriter(os.path.join(args.log_root,args.log_name,datetime))
# full logger
Path(os.path.join(args.log_root,args.log_name)).mkdir(parents=True, exist_ok=True)
targets = logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(args.log_root,args.log_name)+".log", 'w')
logging.basicConfig(format='%(message)s', level=logging.INFO, handlers=targets)

# initial logs
logging.info("MACHINE {}".format(socket.gethostname()))
logging.info("PID {}".format(os.getpid()))
logging.info(args_import)
for k,v in vars(args).items():
    if k == "loss_weights":
        logging.info("{} {}".format(k, [Counter(vi) for vi in v]))
    elif k == "freeze_layer_names":
        pass
    else:
        logging.info("{} {}".format(k,v))

# Imports for Architecture, Data Loader 
##################################################### 
architecture_import = "from {} import *".format(args.architecture)
exec(architecture_import)
data_loader_import = "from {} import *".format(args.data_loader)
exec(data_loader_import)
    
# default checkpoints 
checkpoint = {
    'model_summary': None,  
    'model_state': None,
    'optim_state': None,
    'epoch': 0,
    'tr_counter': 0,
    'va_counter': 0}
# additional checkpoints
for task_name in args.task_names:
    checkpoint["_".join([task_name,"loss"])] = np.inf
    checkpoint["_".join([task_name,"epoch"])] = np.inf
    
tr_counter = 0 
va_counter = 0
epoch = 0
path = Path(os.path.join(args.weight_root,args.experiment_name))
path.mkdir(parents=True, exist_ok=True)
  
# Prepare Data Loaders
##################################################### 
bool_dtype, long_dtype, float_dtype, _ = get_dtypes()
# load data
tr_data = dataloader(args, "train")
va_data = dataloader(args, "val")
# data loader
tr_loader = torch.utils.data.DataLoader(tr_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=torch.cuda.is_available())
va_loader = torch.utils.data.DataLoader(va_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=torch.cuda.is_available())

# Prepare Network and Optimizers
##################################################### 
net = model(args)
logging.info(net)
logging.info("Total # of parameters: {}".format(count_parameters(net)))
# must set model type before initializing the optimizer 
# https://discuss.pytorch.org/t/code-that-loads-sgd-fails-to-load-adam-state-to-gpu/61783/2 
net.type(float_dtype)
optimizer = optim.Adam(net.parameters(), lr=args.lr)
compute_loss = compute_loss(args)
 
# Maybe load weights and initialize checkpoints
##################################################### 
if args.load_pretrained_weight == 1:
    
    args.pretrained_weight_root = os.path.expanduser(args.pretrained_weight_root)
    
    # if i am loading elsewhere
    if os.path.join(args.pretrained_weight_root,args.pretrained_experiment_name) is not None and os.path.isdir(os.path.join(args.pretrained_weight_root,args.pretrained_experiment_name)):
        pretrained_weight_root        = args.pretrained_weight_root
        pretrained_experiment_name    = args.pretrained_experiment_name
    elif os.path.join(args.weight_root,args.experiment_name) is not None and os.path.isdir(os.path.join(args.weight_root,args.experiment_name)):
        pretrained_weight_root        = args.weight_root
        pretrained_experiment_name    = args.experiment_name        
    else:
        sys.exit("Model load weights not found")
        
    # load the best epoch for each task
    for pretrained_epoch_name,pretrained_layer_names,pretrained_task_name in zip(args.pretrained_epoch_names,args.pretrained_layer_names,args.pretrained_task_names):
        
        print("pretrained_task_name = {} pretrained_epoch_name = {}".format(pretrained_task_name,pretrained_epoch_name))
        print("pretrained_layer_names = {}".format(pretrained_layer_names))
        print()
        
        # load best
        if pretrained_epoch_name == -1:
            #pt_files = glob(os.path.join(args.weight_root,args.experiment_name,"*"))
            pt_files = os.listdir(os.path.join(pretrained_weight_root,pretrained_experiment_name))
            pt_files = sorted([x for x in pt_files if pretrained_task_name+"_epoch" in x])
            pretrained_epoch_name = os.path.basename(pt_files[-1])
        
        # load closest
        if type(pretrained_epoch_name) == type(1337):
            #pt_files = glob(os.path.join(args.weight_root,args.experiment_name,"*"))
            pt_files = os.listdir(os.path.join(pretrained_weight_root,pretrained_experiment_name))
            pt_files = sorted([x for x in pt_files if pretrained_task_name+"_epoch" in x])
            int_pt_files = [int(re.search('best_(.*?).pt', pt_file).group(1)) for pt_file in pt_files]
            pretrained_epoch_name  = pt_files[int_pt_files.index(min(int_pt_files, key=lambda x:abs(x-pretrained_epoch_name)))]
            pretrained_epoch_name  = os.path.basename(pretrained_epoch_name)
        
        print("Attempting to load from: " + os.path.join(args.pretrained_weight_root,args.pretrained_experiment_name,pretrained_epoch_name), "into", pretrained_layer_names)
        print()
    
        # load checkpoint dictionary
        checkpoint = torch.load(os.path.join(pretrained_weight_root,pretrained_experiment_name,pretrained_epoch_name))
                
        # load weights
        model_state = checkpoint["model_state"]
        print("These are the weights in .pt file")
        for k,v in model_state.items():
            print(k)
        print()

        # not good because "encoder" will also grab "hand_encoder" or "object_encoder" etc
        #layer_dict = {k:v for k,v in model_state.items() for layer_name in layer_names if layer_name in k}       

        # not good if the layer name is e.g. main.hand_encoder
        #layer_dict = {k:v for k,v in model_state.items() for layer_name in layer_names if layer_name == k.split(".")[0]}  
        
        # better
        layer_dict = {}
        for k,v in model_state.items():
            for layer_name in pretrained_layer_names:
                k_split = k.split(".")
                if any([layer_name == k for k in k_split]):
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
        print("=======================")
        print("pretrained_epoch_name", pretrained_epoch_name)
        print("These are the weights successfully transferred from my .pt file")
        for key in layer_dict.keys():
            print(key)
        print()
        net.load_state_dict(layer_dict,strict=args.pretrained_strict)
    
    print("Model Loaded")
    
    """# load the best epoch for each task
    # - what if I want to load the latest version even if its not the best one ?
    #   - i should be able to do it simply by loading the same .pt file but with different component names
    for epoch_name,layer_names in zip(args.epoch_names,args.layer_names):
    
        # load checkpoint dictionary
        checkpoint = torch.load(os.path.join(pretrained_weight_root,pretrained_experiment_name,epoch_name))
                
        # load weights
        model_state = checkpoint["model_state"]
        layer_dict = {k:v for k,v in model_state.items() for layer_name in layer_names if layer_name in k}
        logging.info("epoch_name {}".format(epoch_name))
        logging.info("layer_dict.keys()")
        logging.info(layer_dict.keys())
        logging.info()
        net.load_state_dict(layer_dict,strict=True)
    
    # what is the right way to load if I am saving the best weights for each component?
    # - the right way would be to load the component with the latest checkpoint last i.e. right most of the args"""
    
    # incase i am freezing the base network and training the head network
    if args.pretrained_load_optimizer == 1:
        optimizer.load_state_dict(checkpoint["optim_state"])

# Load weights and initialize checkpoints
##################################################### 
if args.restore_from_checkpoint == 1:
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
                pt_files = sorted([x for x in pt_files if task_name in x])
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
                    k_split = k.split(".")
                    if any([layer_name == k for k in k_split]):
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

# the loss checkpoints will have gotten refreshed if loading from a pretrained network so we reinsert them here
for task_name in args.task_names:
    if task_name+"_loss" not in checkpoint:
        checkpoint[task_name+"_loss"] = np.inf
    if task_name+"_epoch" not in checkpoint:
        checkpoint[task_name+"_epoch"] = np.inf
        
# Main Loop
####################################################  
def loop(net, inp_data, optimizer, counter, epoch, args, mode):        
    
    assert mode == "train" or mode == "val"
    
    # move to gpu
    for k,v in inp_data.items():
        inp_data[k] = inp_data[k].cuda() if torch.cuda.device_count() > 0 and not isinstance(v, list) else inp_data[k]
        
    # maybe freeze some layers
    if args.freeze is not None:
        net = eval(args.freeze)(net, args.freeze_layer_names[epoch], os.path.join(args.log_root,args.log_name))
    
    with torch.autograd.set_detect_anomaly(True):
    #with torch.cuda.amp.autocast():
        # Forward pass
        out_data = net(inp_data, mode=mode)
            
        # compute unscaled losses        
        losses = {}
        for loss_name, loss_function in zip(args.loss_names, args.loss_functions):
            #print()
            #print(loss_name)
            #print(inp_data["obj_xyz_unpadded_length"])
            #print(inp_data["obj_xyz_unpadded_objects"])
            losses[loss_name] = compute_loss(inp_data, out_data, loss_name, loss_function)
            
        # write unscaled losses to log file
        for k,v in losses.items():
            if v != 0:
                writer.add_scalar(os.path.join(k,mode), v.item(), counter)
        
        # scale the losses since save_checkpoint checks the dictionary losses
        # also backprop the scaled losses
        total_loss = 0
        for loss_name, loss_weight in zip(args.loss_names, args.loss_weights):
            #print(loss_weight)
            #print(loss_name)
            #print(losses[loss_name])
            #print(loss_weight[epoch])
            losses[loss_name] *= loss_weight[epoch]
            total_loss += losses[loss_name]
            
        if mode == "train" and optimizer is not None:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
    # move all to cpu numpy
    losses = iterdict(losses)
    inp_data = iterdict(inp_data)
    out_data = iterdict(out_data)
        
    return {"out_data":out_data, "losses":losses}

tr_counter = checkpoint['tr_counter']
va_counter = checkpoint['va_counter']
epoch = checkpoint["epoch"]
logging.info("{} tr_counter: {} va_counter: {}".format(epoch, tr_counter, va_counter))

# Train
####################################################
for i in range(epoch, 100000): 
    
    # training ---------------------
    net.train()
    start = time.time()
    t1 = time.time()
        
    for batch_idx, tr_data in enumerate(tr_loader):
        
        t2 = time.time()
        
        if batch_idx%100 == 0:
            #print(batch_idx)
            logging.info("Epoch {} training batch {} of {} Time: {}".format(str(i).zfill(2), batch_idx, len(tr_loader), t2-t1))
            
        tr_output = loop(net=net,inp_data=tr_data,optimizer=optimizer,counter=tr_counter,epoch=i,args=args,mode="train")
        tr_counter= tr_counter+1
        if batch_idx!=0 and batch_idx%args.tr_step == 0:
            break
        
        t1 = time.time()
                                       
        #break
    end = time.time()
    tr_time = end - start
    # training ---------------------
    
    # validation ---------------------
    start = time.time()
    t1 = time.time()
    with torch.no_grad():        
        net.eval()
        va_losses = {}
        for batch_idx, va_data in enumerate(va_loader):
            t2 = time.time()
            
            if batch_idx%50 == 0:
                logging.info("Epoch {} validation batch {} of {} Time: {}".format(str(i).zfill(2), batch_idx, len(va_loader), t2-t1))
            
            va_output = loop(net=net,inp_data=va_data,optimizer=None,counter=va_counter,epoch=i,args=args,mode="val")
            va_counter= va_counter+1 
                       
            # accumulate loss                    
            for key in args.loss_names:
                va_losses = collect(va_losses, key=key, value=va_output["losses"][key]) if key in va_output["losses"] else va_losses
            
            if batch_idx!=0 and batch_idx%args.va_step == 0:
                break
                
            t1 = time.time()
            
    end = time.time()
    va_time = end - start     
    
    # average loss
    for k,v in va_losses.items():
        va_losses[k] = np.mean(va_losses[k])
        
    # checkpoint
    for task_name,task_component in zip(args.task_names,args.task_components):
        
        # compute task loss for current epoch
        task_loss = np.sum([va_losses[x] for x in task_component])
        
        # maybe reset best loss
        if args.reset_loss is not None and any([i == x for x in args.reset_loss]):
            logging.info("Resetting loss at epoch {}".format(i))
            checkpoint["_".join([task_name,"loss"])] = np.inf
        
        # maybe save model
        checkpoint = save_model(checkpoint, net, args, optimizer, tr_counter, va_counter, task_name, current_epoch=i, current_loss=task_loss)
    
    # print
    logging.info("{:20} {}".format("Curr Loss: ", va_losses))
    logging.info("{:20} {}".format("Best Loss: ", {k:v for k,v in checkpoint.items() if "loss" in k}))
    logging.info("{:20} {}".format("Best Epoch: ", {k:v for k,v in checkpoint.items() if "epoch" in k}))
    logging.info("{:20} {}".format("Training time: ", tr_time))
    logging.info("{:20} {}".format("Validation time: ", va_time))
