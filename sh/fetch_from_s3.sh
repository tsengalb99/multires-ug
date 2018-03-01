#/bin/bash

S3_DIR=s3://caltech-ml-multires/group3
LOCAL_DIR=/tmp/stephan/logs/multires/group3

mkdir -p $LOCAL_DIR

aws s3 sync $S3_DIR $LOCAL_DIR \
--exclude="*.pth.tar" \
--exclude="*F-*" \
--exclude="*_F*" \
--exclude="*B-*" \
--exclude="*_B-*"\
--exclude="*10-12-17*"
# --dryrun

echo "syncing with s3"
echo $LOCAL_DIR
echo $S3_D
#
