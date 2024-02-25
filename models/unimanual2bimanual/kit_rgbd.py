import torch.distributions as tdist
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch_geometric.nn as tgnn
import numpy as np
import time
import sys
import logging
sys.path.append("..")

from torch_geometric.nn import MessagePassing
from models.components import *
from models.utils import *
from models.unimanual2bimanual.utils import *
from models.unimanual2bimanual.components import *
from models.unimanual2bimanual.ensemble_kit_rgbd_reconstruction_module import *
from models.unimanual2bimanual.ensemble_kit_rgbd_forecasting_module import *

print_var = 0

# freeze all except layer_names
def unfreeze(net, layer_names, log_path):
        
    # logger
    logger = logging.getLogger(log_path)
        
    # de-activate gradients for all
    for name, param in net.named_parameters():
        param.requires_grad = False
    
    global print_var
    if print_var == 0:
        logger.info("Unfreezing the following layers")
        logger.info("===============================")
    
    # quite dangerous to use in but i got no choice...
    
    # re-activate selected ones
    for i,layer_name in enumerate(layer_names):
        for name, param in net.named_parameters():
            if layer_name in name:
                param.requires_grad = True
                if print_var == 0:
                    logger.info("{}".format(name))
            
    if print_var == 0:
        print_var = 1        
        
    """for name, param in net.named_parameters():
        name = name.split(".") # transformer_decoder.layers.0.self_attn.in_proj_weight -> transformer_decoder
        if any([name == layer_name for layer_name in layer_names]):
            param.requires_grad = False
        else:
            param.requires_grad = True"""
    
    """for name,param in net.named_parameters():
        print(name, param.requires_grad)
    sys.exit()"""
    
    return net
    
class model(nn.Module):
    def __init__(self, args):
        super(model, self).__init__()
                 
        for key, value in args.__dict__.items():
            setattr(self, key, value)
                        
        self.reconstruction_module  = ensemble_reconstruction_module(args)
        self.forecasting_module     = ensemble_forecasting_module(args)
        
    def forward(self, data, mode):
          
        # forward pass
        reconstruction_module_out   = self.reconstruction_module(data, mode=mode)
        forecasting_module_out      = self.forecasting_module(data, mode=mode)
        
        # return data
        return_data = {**reconstruction_module_out, **forecasting_module_out}
        return return_data
        