import random, struct, sys, os
import numpy as np
from torch.utils.data import Dataset

BYTES_PER_INT = 4

def load_ints(fn, num, pos=0):
    with open(fn, "rb") as f:
        f.seek(pos * BYTES_PER_INT, 0)
        bytes = f.read(num * BYTES_PER_INT)
        tup = struct.unpack("{}i".format(num), bytes)
    return np.array(tup)

class LorenzDataset(Dataset):
    """
    """
    def __init__(self, args, data_dir, res=1, train=True,
        valid=False, train_valid_split=0.1, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.args = args
        self.data_dir = data_dir
        self.res = res
        self.train = train
        self.valid = valid
        self.train_valid_split = train_valid_split

        #load data file
        self.dat = np.load(self.data_dir)

        self.N = self.dat.shape[0]
        print('data set shape', self.dat.shape)
        print("Loaded dataset with", self.N, "examples")

        #self.s0 = np.random.rand(self.N, 3)


    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        i = random.randrange(0, self.N)

        total_time = self.args.input_len + self.args.output_len

        # states = self.gen_lorenz_series(self.s0[i], total_time, 1)
        states = self.dat[i,]
        data = states[:self.args.input_len]
        label = states[self.args.input_len:]
        return data.astype(float), label.astype(float)

    def lorenz(self, x, y, z, s=10, r=28, b=2.667):
        x_dot = s*(y - x)
        y_dot = r*x - y - x*z
        z_dot = x*y - b*z
        return np.array([x_dot, y_dot, z_dot])

    def gen_lorenz_series(self, s0, num_steps, num_freq):
        dt = 0.01
        s = np.empty((num_steps,3))
        s[0] = s0
        ss = np.empty((num_steps//num_freq,3))
        j = 0
        for i in range(num_steps-1):
            # Derivatives of the X, Y, Z state,
            if i%num_freq ==0:
                ss[j] = s[i]
                j += 1
            sdot = self.lorenz(s[i,0], s[i,1], s[i,2])
            s[i + 1] = s[i] + sdot  * dt

        return ss

    def from_file(self):
        # print("Seeking to", idx, "/", self.num_test, self.__len__())
        f_bh = load_ints(self.feat_bh, 1, pos=idx)
        if self.res_bh > 1:
            for i in range(f_bh.shape[0]):
                f_bh[i] = self.downsample(f_bh[i], scale=self.res_bh, nr=50, nc=40)

        f_def = load_ints(self.feat_def, 5, pos=5*idx)
        if self.res_def > 1:
            for i in range(f_def.shape[0]):
                if f_def[i] > -1:
                    f_def[i] = self.downsample(f_def[i], scale=self.res_def, nr=12, nc=12)

        assert(idx < self.N)

        pos = 3 * idx
        y = load_ints(self.gt, 3, pos=pos)

        sample = {'f_bh': f_bh, 'f_def': f_def, "y": y}

        if self.transform:
            sample = self.transform(sample)

    def downsample(self, i, scale=1, nr=50, nc=40):
        r = int(i % nr)
        c = int(i / nr)

        _nr = int(nr / scale)
        # _nc = int(nc / scale)
        _r = int(r / scale)
        _c = int(c / scale)

        _i = _c * _nr + _r

        # print("DEBUG:", i, r, c, nr, "->", _i, _r, _c, _nr)

        return _i
