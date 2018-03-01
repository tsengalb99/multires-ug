#/bin/bash
set -x
# for ds in multires_fvf_200000_tt0.90


for ds in multires_bb_200000_sw0.50_tt0.90; do
  REMOTE_DIR=/tmp/stephan/logs/multires/$ds
  LOCAL_DIR=/tmp/stephan/logs/multires/group3

  mkdir -p $LOCAL_DIR

  for i in 2 3 4; do

  rsync -azP stzheng@ml-login$i.cms.caltech.edu:$REMOTE_DIR $LOCAL_DIR \
  --exclude="*.pth.tar" \
  --exclude="*F-*" \
  --exclude="*_F*" \
  --exclude="*B-*" \
  --exclude="*_B-*"

  done
done
