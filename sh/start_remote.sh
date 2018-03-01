#/bin/bash

CMD=$1
project_name=$2
dataset=$3
model_type=$4

echo "./sh/start_remote.sh [cmd] [dataset] [model_type]"
echo "./sh/start_remote.sh $CMD $2 $3"

if [ "$CMD" == "cms-start" ]; then

exp_name=multires
ssh -t stzheng@ml-login2.cms.caltech.edu "cd ~/projects/$project_name; git pull; ./sh/create_tmux.sh $project_name $exp_name $dataset $model_type \"1e-3;1e-4\" nos3 attach local; echo \"ml2 started!\""

fi

if [ "$CMD" == "cms-archive" ]; then

for i in "" 2 3 4 5; do

src=$project_name

ssh -t stzheng@ml-login$i.cms.caltech.edu \
"mkdir -p /tmp/stephan/logs/archive; rsync -avP /tmp/stephan/logs/$src/ /tmp/stephan/logs/archive; rm -rf /tmp/stephan/logs/$src/"

done

fi

if [ "$CMD" == "cms-check" ]; then

cmds="top"

for i in "" 2 3 4 5; do
ssh \
-t stzheng@ml-login$i.cms.caltech.edu \
$cmds
done

fi

if [ "$CMD" == "cms-getdata" ]; then

cmds="mkdir -p /tmp/stephan/data/$project_name; cp -r /cs/ml/datasets/$project_name /tmp/stephan/data/$project_name"

for i in "" 2 3 4 5; do
ssh \
-t stzheng@ml-login$i.cms.caltech.edu \
$cmds
done

fi

if [ "$CMD" == "cms-clean" ]; then

cmds="tmux kill-server; rm -rf /tmp/stephan/logs/$project_name/*"

for i in 2 3 4 5; do
ssh \
-t stzheng@ml-login$i.cms.caltech.edu \
$cmds
done

fi




# Update when needed




OLDIFS=$IFS

ip=(\
ec2-34-215-130-161.us-west-2.compute.amazonaws.com \
ec2-34-214-162-30.us-west-2.compute.amazonaws.com \
ec2-34-214-56-142.us-west-2.compute.amazonaws.com \
ec2-35-162-195-40.us-west-2.compute.amazonaws.com \
ec2-35-165-184-237.us-west-2.compute.amazonaws.com \
ec2-35-167-214-49.us-west-2.compute.amazonaws.com \
ec2-35-161-0-150.us-west-2.compute.amazonaws.com \
ec2-52-25-176-135.us-west-2.compute.amazonaws.com
)

if [ "$CMD" == "aws-getdata" ]; then

# Data is stored in tmp -- if instance is stopped, tmp is flushed. This fetches it again.

# cmds="cd ~; mkdir -p /tmp/stephan/data/basketball/parsed; scp -r stzheng@ml-login2.cms.caltech.edu:/tmp/stephan/data/basketball/parsed/new_split_200000_sw0.50_tt0.90 /tmp/stephan/data/basketball/parsed; echo \"$i fetched data!\""
cmds="cd ~; mkdir -p /tmp/stephan/data/fvf; scp -r stzheng@ml-login2.cms.caltech.edu:/cs/ml/datasets/flyvsfly/new_split_200000_tt0.90 /tmp/stephan/data/fvf; echo \"$i fetched data!\""

for i in "${ip[@]}"; do
ssh \
-i ~/projects/stz.pem \
-t ubuntu@$i \
$cmds
done

fi

if [ "$CMD" == "aws-check" ]; then


cmds="top"

for i in "${ip[@]}"; do
ssh \
-i ~/projects/stz.pem \
-t ubuntu@$i \
$cmds
done

fi

if [ "$CMD" == "aws-clean" ]; then

cmds="tmux kill-server"
cmds="$cmds;rm -rf /tmp/stephan/logs/*"

for i in "${ip[@]}"; do
ssh \
-i ~/projects/stz.pem \
-t ubuntu@$i \
$cmds
done

fi

if [ "$CMD" == "aws-start" ]; then

exp=(\
loss_conv,"1e-8;1e-7;1e-6;1e-5;1e-4","" \
sd_div,"1e-8;1e-7;1e-6;1e-5","" \
sd_div,"1e-4;1e-3;1e-2;1","" \
mu_sd,"1e-8;1e-7;1e-6;1e-5;1e-4","1e-2" \
mu_sd,"1e-8;1e-7;1e-6","1e-4;1e-3" \
mu_sd,"1e-5;1e-4;1e-3","5e-4;5e-3;5e-2" \
ent_div,"1e-6;1e-5;1e-4;1e-3;1e-2","" \
ent_div,"5e-6;5e-5;5e-4;5e-3;5e-2",""
)

NUM_MACHINES=7 # 0-indexed!

for i in $(seq 0 $NUM_MACHINES); do

IFS=','
read ip variant hparam1 hparam2 <<< "${ip[i]},${exp[i]}"

echo "start_remote.sh $ip $variant with $hparam1, $hparam2"

ssh \
-i ~/projects/stz.pem \
-t ubuntu@$ip \
"cd ~/projects/multi-resolution-tensor-training; git pull; ./sh/create_tmux.sh $variant $dataset $model_type '${hparam1}' '${hparam2}' s3 noattach aws; echo \"$ip started [$variant]!\""
done

fi

IFS=$OLDIFS
