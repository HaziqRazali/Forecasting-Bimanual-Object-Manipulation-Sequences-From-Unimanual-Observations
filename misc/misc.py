import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.functional import cross_entropy

# Save function
####################################################  
def save_model(checkpoint, net, args, optimizer, tr_counter, va_counter, task_name, current_epoch, current_loss):
    
    # do not save if current loss is greater (lousier) than the checkpoint 
    if current_loss > checkpoint["_".join([task_name,"loss"])]:
        return checkpoint
        
    checkpoint['model_summary'] = str(net)
    checkpoint['args']          = str(args)
    checkpoint["_".join([task_name,"loss"])]  = current_loss
    checkpoint["_".join([task_name,"epoch"])] = current_epoch
    checkpoint['model_state']   = net.state_dict()
    checkpoint['optim_state']   = optimizer.state_dict() 
    checkpoint['epoch']         = current_epoch
    checkpoint['tr_counter']    = tr_counter
    checkpoint['va_counter']    = va_counter   
    
    #if  args.debug==False:
    checkpoint_path = os.path.join(args.weight_root, args.experiment_name, "_".join([task_name, "epoch", str(current_epoch).zfill(4), "best", str(current_epoch).zfill(4)+".pt"]))
    torch.save(checkpoint, checkpoint_path)
    return checkpoint

# Convert dictionary data type
####################################################  
def iterdict(d, operation=".cpu().detach().numpy()"):
    for k, v in d.items():
        if isinstance(v, dict):
            iterdict(v)
        else:
            #print(k, type(v), type(v) == type(torch.Tensor(1)))
            if type(v) == type(torch.Tensor(1)):
                v = eval("v"+operation)
            d.update({k: v})
    return d

# Flatten dictionary
####################################################  
def flattendict(d, parent_key ='', sep ='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
  
        if isinstance(v, MutableMapping):
            items.extend(flattendict(v, new_key, sep = sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# Set data type
####################################################
def get_dtypes():
    bool_dtype  = torch.BoolTensor
    long_dtype  = torch.LongTensor
    float_dtype = torch.FloatTensor
    device      = "cpu"
    if torch.cuda.device_count() > 0:
        bool_dtype  = torch.cuda.BoolTensor
        long_dtype  = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
        device      = torch.cuda.current_device()
    return bool_dtype, long_dtype, float_dtype, device
    
# Accumulate list in dictionary
####################################################          
def collect(dictionary, key, value):
    if key not in dictionary:
        dictionary[key] = [value]
    else:
        dictionary[key].append(value)
    return dictionary
    
# count the total number of parameters
####################################################    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
# for train_infogan.py
####################################################   
def write_stats_from_var(log_dict, torch_var, name, idx=None):
    if idx is None:
        # log_dict['%s_mean' % name] = torch_var.data.mean()
        # log_dict['%s_std' % name] = torch_var.data.std()
        # log_dict['%s_max' % name] = torch_var.data.max()
        # log_dict['%s_min' % name] = torch_var.data.min()
        np_var = torch_var.data.cpu().numpy()
        for i in [0, 25, 50, 75, 100]:
            log_dict['%s_%d' % (name, i)] = np.percentile(np_var, i)
    else:
        assert type(idx) == int
        assert len(torch_var.size()) == 2
        write_stats_from_var(log_dict, torch_var[:, idx], '%d_%s' % (idx, name))