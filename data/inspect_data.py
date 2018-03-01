import os, struct

fn = "/tmp/new_split_dev/train/gt_shot_made.train.bin"

with open(fn, "rb") as f:
	while True:
		bs = f.read(12)
		if not bs:
			break
		(a, t, l) = struct.unpack("3i", bs)

		print(a,t,l)

