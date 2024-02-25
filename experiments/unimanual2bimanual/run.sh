
CUDA_VISIBLE_DEVICES=0 python "$HOME/Forecasting-Bimanual-Object-Manipulation-Sequences-From-Unimanual-Observations/train.py" --args="args.unimanual2bimanual" --config_file="kit_mocap/joint.ini"
CUDA_VISIBLE_DEVICES=0 python "$HOME/Forecasting-Bimanual-Object-Manipulation-Sequences-From-Unimanual-Observations/test.py" --args="args.unimanual2bimanual" --config_file="kit_mocap/joint.ini"