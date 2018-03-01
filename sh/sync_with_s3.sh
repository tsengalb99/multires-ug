#/bin/bash

set -x

BUCKET="caltech-ml-multires"
LOCAL_DIR="/tmp/stephan/logs/multires"
REMOTE_DIR="s3://$BUCKET/group3"

while true; do

aws s3 sync $LOCAL_DIR $REMOTE_DIR

sleep 60

done
