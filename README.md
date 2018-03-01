# multi-resolution-tensor-training

## main.py
Multi-threaded routine with flags
- start train and eval and vis multi-threads
- create shared model 


## train.py
Training routine with PyTorch
- shared model: parameter servers 
- data_loader: mini-batch feeder
- load_state_dict: current weights of the (shared) model
- put: visualize the statistcs of the model (via visdom)
- cuda: not applicable for multi-threaded GPU (not sure yet)

## datareader.py
Data loader from the sequences
- currently generate data from get_item func
- can also be changed to load data file

## evaluate.py
Visualization from the model
- use visdom to load the state_dict
- plot the statistics on the web browser





