#/bin/bash

project_name=$1
exp_name=$2
dataset=$3
model_type=$4
hparam=$5
s3=$6
attach=$7
loc="$8"

tmux kill-session -t "multires"
tmux kill-session -t "jupyter"
tmux kill-session -t "visdom"
if [ "$4" == "s3" ]; then
tmux kill-session -t "s3sync"
fi

sleep 1

tmux new-session -d -s "multires"
tmux new-session -d -s "jupyter"
tmux new-session -d -s "visdom"
if [ "$s3" == "s3" ]; then
tmux new-session -d -s "s3sync"
fi

sleep 1

tmux send-keys -t "multires" "sac explore-ma; ./sh/run_exps.sh.$loc $project_name $exp_name $dataset $model_type '$hparam'" C-m
tmux send-keys -t "jupyter" "sac explore-ma; jupyter notebook" C-m
tmux send-keys -t "visdom" "sac explore-ma; python -m visdom.server -port=11111" C-m
if [ "$s3" == "s3" ]; then
tmux send-keys -t "s3sync" "sac explore-ma; ./sh/sync_with_s3.sh" C-m
fi

if [ "$attach" == "attach" ]; then
tmux attach -t "multires"
fi
