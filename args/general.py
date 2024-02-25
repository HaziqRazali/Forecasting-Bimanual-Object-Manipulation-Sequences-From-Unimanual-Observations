import os
import ast
import sys
from configobj import ConfigObj

def general_argparser(args):

    # config parser
    if os.path.isfile(args.config_file) == False:
        print("{} does not exist!".format(args.config_file))
    config = ConfigObj(args.config_file,unrepr=True)
    
    """
    process args
    """
    
    # process paths    
    for key in ["log_root", "weight_root", "data_root", "result_root"]:
        config[key] = os.path.expanduser(config[key])
    
    # process loss schedule
    config["loss_weights"] = [eval(x) for x in config["loss_weights"]]
        
    # pass config to args
    for key, value in config.items():
        setattr(args, key, value)
    
    """
    overwrite args
    """
        
    # parse commands without the double dash
    for arg in sys.argv[1:]:
        if '--' not in arg:
            sep = arg.find('=')
            try:
                key, value = arg[:sep], ast.literal_eval(arg[sep + 1:])
            except:
                key, value = arg[:sep], arg[sep + 1:] 
            
            # need to re-eval
            if key == "loss_weights":
                value = [eval(x) for x in value]
                
            # make sure key is in the config file
            #getattr(args, key)
            setattr(args, key, value)
        
    """
    assert args
    """
        
    caller = sys.argv[0]
    caller = os.path.basename(caller)
        
    # sanity check make sure the config file has the same name as the experiment name
    if os.path.basename(args.config_file).replace(".ini","") != os.path.basename(args.experiment_name):
        print("args.config_file={} not in args.experiment_name={}".format(args.config_file.replace(".ini",""),os.path.basename(args.experiment_name)))
        sys.exit()
    
    # update log name
    args.log_name = args.experiment_name if args.log_name is None else args.log_name
    
    if "train" in caller:
    
        # the lists containing the loss schedules must have the same length
        if args.loss_weights is not None:
            assert len(set(map(len,args.loss_weights))) == 1
        
        # the loss names, functions and weights must have the same length    
        if args.loss_names is not None or args.loss_functions is not None or args.loss_weights is not None:
            assert len(args.loss_names) == len(args.loss_functions)
            assert len(args.loss_functions) == len(args.loss_weights)
            
        # len of task_names must be of same len of the task_component list of lists
        assert len(args.task_names) == len(args.task_components)
        
        # list of epoch_names must be of the same length as the list of lists
        assert len(args.checkpoint_epoch_names) == len(args.checkpoint_layer_names)
        
        if args.freeze is not None:
            assert len(args.freeze_layer_names) == len(args.freeze_epochs)
            args.freeze_layer_names = [[freeze_layer_names] * freeze_epochs for freeze_layer_names, freeze_epochs in zip(args.freeze_layer_names, args.freeze_epochs)][0]
    
    return args