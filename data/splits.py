import os, sys, random


weak   = 14451032
strong =  1289692
total  = 15740724

INDIR="/cs/ml/datasets/bball/v1/bball_tracking"
OUTDIR="/cs/ml/datasets/bball/v1/new_split"

FN_GT_WEAK = "gtruth_ballhandler_weak.bin"
FN_GT_STRONG = "gtruth_ballhandler_strong.bin"


# Split train / test
train_prob = 0.9
strong_train_idx = []
strong_test_idx = []
weak_train_idx = []
weak_test_idx = []

for i in range(weak):
    rand01 = random.random()
    if rand01 < train_prob:
        weak_train_idx += [i]
    else:
        weak_test_idx += [i]

for i in range(weak, weak + strong):
    rand01 = random.random()
    if rand01 < train_prob:
        strong_train_idx += [i]
    else:
        strong_test_idx += [i]

# Only keep parts of it
keep_size = 200000 # 2000000 # 15000000
keep_strong = 0.5 # strong / total # new strong-weak ratio
keep_strong_train = int(keep_size * train_prob * keep_strong)
keep_strong_test = int(keep_size * (1-train_prob) * keep_strong)
keep_weak_train = int(keep_size *  train_prob * (1-keep_strong))
keep_weak_test = int(keep_size * (1-train_prob) * (1-keep_strong))

print(keep_strong_train, keep_strong_test, keep_weak_train, keep_weak_test)
print(len(strong_train_idx), len(strong_test_idx), len(weak_train_idx), len(weak_test_idx))

strong_train_idx = random.sample(strong_train_idx, min(len(strong_train_idx), keep_strong_train))
strong_test_idx = random.sample(strong_test_idx, min(len(strong_test_idx), keep_strong_test))
weak_train_idx = random.sample(weak_train_idx, min(len(weak_train_idx), keep_weak_train))
weak_test_idx = random.sample(weak_test_idx, min(len(weak_test_idx), keep_weak_test))

strong_train_idx.sort()
strong_test_idx.sort()
weak_train_idx.sort()
weak_test_idx.sort()

OUTDIR = "{}_{}_sw{:.2f}_tt{:.2f}".format(OUTDIR, keep_size, keep_strong, train_prob)
os.makedirs(OUTDIR, exist_ok=True)

fi_neg = os.path.join(INDIR, "groundtruth", FN_GT_WEAK)
fi_pos = os.path.join(INDIR, "groundtruth", FN_GT_STRONG)
fo_tr = os.path.join(OUTDIR, "train")
fo_tt = os.path.join(OUTDIR, "test")

os.makedirs(fo_tr, exist_ok=True)
os.makedirs(fo_tt, exist_ok=True)

with open(fi_neg, "rb") as infile_neg, open(fi_pos, "rb") as infile_pos, \
     open(fo_tr + "/gt.train.bin", "wb+") as outfile_tr, \
     open(fo_tt + "/gt.test.bin", "wb+") as outfile_tt:
    NUM_BYTES_PER_UNIT = 3*4
    ff = infile_neg
    of = outfile_tr
    for idx in weak_train_idx:
        ff.seek(NUM_BYTES_PER_UNIT*idx, 0)
        bytes = ff.read(NUM_BYTES_PER_UNIT)
        of.write(bytes)

    ff = infile_pos
    for idx in strong_train_idx:
        ff.seek(NUM_BYTES_PER_UNIT*(idx - weak), 0)
        bytes = ff.read(NUM_BYTES_PER_UNIT)
        of.write(bytes)

    ff = infile_neg
    of = outfile_tt
    for idx in weak_test_idx:
        ff.seek(NUM_BYTES_PER_UNIT*idx, 0)
        bytes = ff.read(NUM_BYTES_PER_UNIT)
        of.write(bytes)

    ff = infile_pos
    for idx in strong_test_idx:
        ff.seek(NUM_BYTES_PER_UNIT*(idx - weak), 0)
        bytes = ff.read(NUM_BYTES_PER_UNIT)
        of.write(bytes)


for f in ["ballh_c1.bin", "ballh_c2.bin", "ballh_c5.bin", "ballh_c10.bin"]:
    fi = os.path.join(INDIR, "features", f)
    fo_tr = os.path.join(OUTDIR, "train", f).replace("_c", "_lvl").replace(".bin", "").replace("ballh", "feat_bh")
    fo_tt = os.path.join(OUTDIR, "test", f).replace("_c", "_lvl").replace(".bin", "").replace("ballh", "feat_bh")
    with open(fi, "rb") as infile, \
         open(fo_tr + ".train.bin", "wb+") as outfile_tr, \
         open(fo_tt + ".test.bin", "wb+") as outfile_tt:
        NUM_BYTES_PER_UNIT = 4
        ff = infile
        of = outfile_tr
        for idx in weak_train_idx:
            ff.seek(NUM_BYTES_PER_UNIT*idx, 0)
            bytes = ff.read(NUM_BYTES_PER_UNIT)
            of.write(bytes)

        for idx in strong_train_idx:
            ff.seek(NUM_BYTES_PER_UNIT*idx, 0)
            bytes = ff.read(NUM_BYTES_PER_UNIT)
            of.write(bytes)

        of = outfile_tt
        for idx in weak_test_idx:
            ff.seek(NUM_BYTES_PER_UNIT*idx, 0)
            bytes = ff.read(NUM_BYTES_PER_UNIT)
            of.write(bytes)

        for idx in strong_test_idx:
            ff.seek(NUM_BYTES_PER_UNIT*idx, 0)
            bytes = ff.read(NUM_BYTES_PER_UNIT)
            of.write(bytes)

for f in ["feat_defender_occupancy_coarsegrid_lvl2.bin", "feat_defender_occupancy_raw_lvl1.bin"]:
    fi = os.path.join(INDIR, "features", f)
    fo_tr = os.path.join(OUTDIR, "train", f).replace("occupancy_", "").replace("coarsegrid_", "").replace("raw_", "").replace(".bin", "").replace("defender", "def")
    fo_tt = os.path.join(OUTDIR, "test", f).replace("occupancy_", "").replace("coarsegrid_", "").replace("raw_", "").replace(".bin", "").replace("defender", "def")
    with open(fi, "rb") as infile, open(fo_tr + ".train.bin", "wb+") as outfile_tr, open(fo_tt + ".test.bin", "wb+") as outfile_tt:
        NUM_BYTES_PER_UNIT = 5*4
        ff = infile
        of = outfile_tr
        for idx in weak_train_idx:
            ff.seek(NUM_BYTES_PER_UNIT*idx, 0)
            bytes = ff.read(NUM_BYTES_PER_UNIT)
            of.write(bytes)

        for idx in strong_train_idx:
            ff.seek(NUM_BYTES_PER_UNIT*idx, 0)
            bytes = ff.read(NUM_BYTES_PER_UNIT)
            of.write(bytes)

        of = outfile_tt
        for idx in weak_test_idx:
            ff.seek(NUM_BYTES_PER_UNIT*idx, 0)
            bytes = ff.read(NUM_BYTES_PER_UNIT)
            of.write(bytes)

        for idx in strong_test_idx:
            ff.seek(NUM_BYTES_PER_UNIT*idx, 0)
            bytes = ff.read(NUM_BYTES_PER_UNIT)
            of.write(bytes)

