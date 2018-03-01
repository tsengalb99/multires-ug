import os, sys, random

total  = int(98216596 / 21 / 4) # 1169246

TMP_INDIR="/tmp/stephan/data/fvf"
INDIR="/cs/ml/datasets/flyvsfly"
OUTDIR="/cs/ml/datasets/flyvsfly/new_split"

FN_F_POSE = "features_pose.bin"
FN_F_SPAT = "features_spat.bin"
FN_GT = "groundtruth.bin"

# Split train / test
train_prob = 0.9
train_idx = []
test_idx = []

for i in range(total):
    rand01 = random.random()
    if rand01 < train_prob:
        train_idx += [i]
    else:
        test_idx += [i]


# Only keep parts of it
keep_size = 200000 # 2000000 # 15000000
keep_weak_train = int(keep_size *  train_prob)
keep_weak_test = int(keep_size * (1-train_prob))

train_idx = random.sample(train_idx, min(len(train_idx), keep_weak_train))
test_idx = random.sample(test_idx, min(len(test_idx), keep_weak_test))

train_idx.sort()
test_idx.sort()

OUTDIR = "{}_{}_tt{:.2f}".format(OUTDIR, keep_size, train_prob)
os.makedirs(OUTDIR, exist_ok=True)

fi = os.path.join(TMP_INDIR, "groundtruth", FN_GT)
fo_tr = os.path.join(OUTDIR, "train")
fo_tt = os.path.join(OUTDIR, "test")

os.makedirs(fo_tr, exist_ok=True)
os.makedirs(fo_tt, exist_ok=True)


def combine_weak_strong():
    _fs = pj(fp, "groundtruth/groundtruth_strong.bin")
    _fw = pj(fp, "groundtruth/groundtruth_weak.bin")
    _of = pj("/tmp/stephan/data/fvf/groundtruth/groundtruth.bin")

    num = 3
    e = 0
    with open(_fs, "rb") as fs, open(_fw, "rb") as fw, open(_of, "wb") as of:

      fw.seek(0)
      fs.seek(0)

      N = 2 * 10
      labels = [-1] * N
      curr_tw = 0
      curr_ts = 0
      nobsw = False
      nobss = False

      yesw = False
      yess = False

      while True:

        if curr_tw <= curr_ts:
          bsw = fw.read(num * BYTES_PER_INT)

          if not bsw:
            nobsw = True

          a,t,l = struct.unpack("{}i".format(num), bsw)
          labels[a] = 0

          if t > curr_tw:
            curr_tw = t
            yesw = True

        if curr_ts <= curr_tw:
          bss = fs.read(num * BYTES_PER_INT)

          if not bss:
            nobss = True

          a,t,l = struct.unpack("{}i".format(num), bss)
          labels[a] = l

          if t > curr_tw:
            curr_ts = t
            yess = True

        if yesw and yess:
          _bs = struct.pack("{}i".format(1+N), t, *labels)
          of.write(_bs)

          labels = [-1] * N
          yesw = False
          yess = False

        if vals[1] % 10000 == 0:
          print(e, vals)
        e += 1

        if nobsw and nobss: break



with open(fi, "rb") as infile, \
     open(fo_tr + "/gt.train.bin", "wb+") as outfile_tr, \
     open(fo_tt + "/gt.test.bin", "wb+") as outfile_tt:

    N = 2 * 10

    NUM_BYTES_PER_UNIT = (1 + N) * 4
    ff = infile
    of = outfile_tr
    for idx in train_idx:
        ff.seek(NUM_BYTES_PER_UNIT*idx, 0)
        bytes = ff.read(NUM_BYTES_PER_UNIT)
        of.write(bytes)

    of = outfile_tt
    for idx in test_idx:
        ff.seek(NUM_BYTES_PER_UNIT*idx, 0)
        bytes = ff.read(NUM_BYTES_PER_UNIT)
        of.write(bytes)


f = FN_F_POSE

fi = os.path.join(INDIR, "features", f)
fo_tr = os.path.join(OUTDIR, "train", f).replace(".bin", "") + ".train.bin"
fo_tt = os.path.join(OUTDIR, "test", f).replace(".bin", "") + ".test.bin"

with open(fi, "rb") as infile, \
     open(fo_tr, "wb+") as outfile_tr, \
     open(fo_tt, "wb+") as outfile_tt:
    NUM_BYTES_PER_UNIT = 2 * 16 * 4
    ff = infile
    of = outfile_tr
    for idx in train_idx:
        ff.seek(NUM_BYTES_PER_UNIT*idx, 0)
        bytes = ff.read(NUM_BYTES_PER_UNIT)
        of.write(bytes)

    of = outfile_tt
    for idx in test_idx:
        ff.seek(NUM_BYTES_PER_UNIT*idx, 0)
        bytes = ff.read(NUM_BYTES_PER_UNIT)
        of.write(bytes)

f = FN_F_SPAT

fi = os.path.join(INDIR, "features", f)
fo_tr = os.path.join(OUTDIR, "train", f).replace(".bin", "") + ".train.bin"
fo_tt = os.path.join(OUTDIR, "test", f).replace(".bin", "") + ".test.bin"

with open(fi, "rb") as infile, open(fo_tr, "wb+") as outfile_tr, open(fo_tt, "wb+") as outfile_tt:
    NUM_BYTES_PER_UNIT = 2 * 8 * 4
    ff = infile
    of = outfile_tr
    for idx in train_idx:
        ff.seek(NUM_BYTES_PER_UNIT*idx, 0)
        bytes = ff.read(NUM_BYTES_PER_UNIT)
        of.write(bytes)

    of = outfile_tt
    for idx in test_idx:
        ff.seek(NUM_BYTES_PER_UNIT*idx, 0)
        bytes = ff.read(NUM_BYTES_PER_UNIT)
        of.write(bytes)
