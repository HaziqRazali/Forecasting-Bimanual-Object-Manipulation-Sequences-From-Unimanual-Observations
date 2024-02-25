import os
import sys

sys.path.append("..")
from args.general import *

def argparser(args):
                
    # parse the args that are common across all projects
    args = general_argparser(args)
    
    """
    assert the args specific to the project
    """
        
    assert len(args.main_actions) == len(args.sample_ratio)
    
    return args
    