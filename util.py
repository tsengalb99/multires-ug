# from graphviz import Digraph
import math
import os
import dill as pickle
import sys
import shutil
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import defaultdict
import time
import matplotlib
from functools import reduce
import matplotlib.pylab as plt
from multiprocessing import Lock, Value
import boto3

PRED_TEAM = [0]

def rescale_01(t):
  d = np.max(t) - np.min(t)
  return (t - np.min(t)) / d


def upscale(t, in_r, in_c, out_r, out_c):
  assert(out_c % in_c == 0)
  assert(out_r % in_r == 0)
  t = np.kron(t, np.ones((int(out_c / in_c), int(out_r / in_r))))
  return t


def vis_spatial_im_row_major(vis, t, opts, in_r, in_c, out_r, out_c, args, step):
  from skimage import transform as tf
  # spatial tensor to image

  title = opts["title"]
  cap = opts["caption"]

  for k in range(t.shape[1]):
    _t = t[:,k].reshape((in_c, in_r))
    _t = rescale_01(_t)
    _t = upscale(_t, in_r, in_c, out_r, out_c)

    t_im = np.zeros((3, out_c, out_r))
    t_im[0] = _t

    t_im = np.transpose(t_im, (0, 2, 1))

    opts["title"] = title + " step {}".format(step)
    opts["caption"] = cap + " dim-{}".format(k)
    vis.image(t_im, opts=opts)


def update_visdom_scalar(vis, x, win, y, xlabel="Episode", ylabel="Loss",
  title="", legend=[], name=""):
    if win is None:
      # (N, M) = Y_np.shape # N = number of points, M = number of time series
      # assert(N == 1)
      Y = np.array([[y], [y]])
      win = vis.line(X=np.array([x, x]), Y=Y,
                     opts=dict(ylabel=ylabel, xlabel=xlabel, title=title, legend=legend))
    else:
      vis.updateTrace(X=np.array([x]), Y=np.array([y]), win=win, name=name)
      # vis.line(X=x * np.ones_like(Y_np), Y=Y_np, update='append', win=win)

    return win

def update_visdom(vis, x, win, Y_np, xlabel="Episode", ylabel="Loss",
  title="", legend=[], name=""):
    if win is None:
      (N, M) = Y_np.shape # N = number of points, M = number of time series
      assert(N == 1)
      Y = np.row_stack((Y_np, Y_np))
      win = vis.line(X=np.array([x, x]), Y=Y,
                     opts=dict(ylabel=ylabel, xlabel=xlabel, title=title, legend=legend))
    else:
      # vis.updateTrace(X=x * np.ones_like(Y_np), Y=Y_np, win=win, name=name)
      vis.line(X=x * np.ones_like(Y_np), Y=Y_np, update='append', win=win)

    return win


def update_visdom_scatter(vis, i_episode, win, Y_np, xlabel="Episode", ylabel="Loss", title="", legend=[]):
    Y = np.expand_dims(Y_np, 0)
    if win is None:
        win = vis.scatter(X=np.array([i_episode]),
                       Y=Y, opts=dict(ylabel=ylabel, xlabel=xlabel, title=title, legend=legend))
    else:
        vis.scatter(X=np.array([np.array(i_episode).repeat(len(Y_np))]),
                 Y=Y, win=win, update='append')
    return win



def _inspect(hand):
  print("hand as str",
    hand.to_str(), hand._pbn_str(), hand.spades, hand.shape, hand.hcp, hand.pt)


def get_variable_from_np(x, dtype=torch.FloatTensor):
  return Variable(torch.from_numpy(x)).type(dtype)


def var_as_np(l):
  return [i.data.numpy() for i in l]


def remove_older_than(d, filter="", n=10):
  if n > 0:
    ts = [(os.path.join(d, _f), os.stat(os.path.join(d, _f)).st_mtime)
      for _f in os.listdir(d)]

    if filter:
      ts = [t for t in ts if filter in t[0]]

    ts.sort(key=lambda x:x[1])

    for t in ts[:-n]:
      os.remove(t[0])


def save_checkpoint(model, is_best, f='/tmp/checkpoint.pth.tar'):
  torch.save(model.state_dict(), f)
  if is_best:
    shutil.copyfile(f, f.replace(".pth", "_best.pth"))


def load_checkpoint(d):
  torch.load(d)


def rmdir(d):
  if os.path.exists(d):
    print("Removing (all contents) of", d)
    shutil.rmtree(d)


def maybe_create(d):
  if not os.path.exists(d):
    print("Creating", d)
    os.makedirs(d)


def partition(lst, n):
  division = len(lst) / float(n)
  return [lst[int(round(division * i)): int(round(division * (i + 1)))]
          for i in xrange(n) ]


def get_many_hot(indices, size):
  idx = np.array(indices)
  one_hot = np.zeros((size), dtype=np.float32)
  one_hot[idx] = 1
  return one_hot


def ob_to_var(obs, agents, dtype):
  x = {}
  for ta, ob in obs.items():
    if ta[0] in PRED_TEAM:

      vs = []

      for o in ob:
        vs += [get_variable_from_np(o, dtype=dtype)]

      x[ta] = vs
  return x


def scalar_to_var(x, dtype=None):
  y = np.array([x])
  return get_variable_from_np(y, dtype=dtype)


def reset(env, agents):
  for ta, agent in agents.items():
    agent.reset()
    if agent.model:
      agent.model.reset()
  env.reset() # order is important: hitpoints must be set by env.


def pp(x):
  return "{:.3f}".format(x).ljust(8)


def clip(x, h=1.0, l=-1.0):
  return max(min(x, h), l)


def ent(logvar_as_np):
  z_dim = logvar_as_np.shape[0]
  std = np.exp(0.5 * logvar_as_np)
  return 0.5 * (math.log( np.prod(std) + 1e-6) + z_dim * math.log( 2 * math.pi * math.exp(1) ))


def _add(l, e, max_n):
  l += [e]
  if max_n > 0 and len(l) > max_n:
    return l[-max_n:]
  else:
    return l


def _dump(fp, n, obj):
  path = os.path.join(fp, n)
  with open(path, 'wb') as f:
    pickle.dump(obj, f)


class Log(object):
  local_ep = 0
  local_env_step = 0

  local_env_steps = [0]
  local_ep_lens = [0]

  global_eps = [0]
  global_env_steps = [0] # only used by test thread

  r_idx = [(0,0,0,0)]
  r = defaultdict(lambda: [0.])
  ep_r = defaultdict(lambda: [0.])
  mar = defaultdict(lambda: [0.])
  sumr = defaultdict(lambda: [0.])

  allr = [0.]

  lp = defaultdict(lambda: defaultdict(lambda: [.2] * 5))

  R_idx = [(0,0,0,0)]
  R = defaultdict(lambda: [0.])
  maR = defaultdict(lambda: [0.])
  sumR = defaultdict(lambda: [0.])

  V = defaultdict(float)
  fr = []

  last_alive = defaultdict(lambda: 0.)

  zs = defaultdict(lambda: defaultdict(lambda: [(None, None, None)]))
  zs_g = defaultdict(lambda: [])
  ent_z = defaultdict(lambda: [0.])


  def add_lp(self, ep_id, ta, lp, step, max_n=0):
    self.lp[ep_id][ta] += [lp]
    if ep_id > max_n:
      for i in range(ep_id - max_n):
        self.lp.pop(i, None)

  def add_z(self, ep_id, zs_g, zs, step, max_n=0):
    self.zs_g[ep_id] = _add(self.zs_g[ep_id], zs_g, max_n)

    for e, (mu, logvar, z) in enumerate(zip(*zs)):
      self.zs[ep_id][e] = _add(self.zs[ep_id][e], (mu, logvar, z), max_n)

    if ep_id > max_n:
      for i in range(ep_id - max_n):
        self.zs_g.pop(i, None)
        self.zs.pop(i, None)


  def add_r(self, rs, ta, step, global_step, ep, global_ep, max_n=0):
    self.r[ta] += rs
    if max_n > 0 and len(self.r[ta]) > max_n:
      self.r[ta] = self.r[ta][-max_n:]


  def add_r_avg(self, ta, ep, ep_len, max_n=0):

    ep_sumr = np.sum(self.r[ta][-ep_len:])

    self.ep_r[ta] = _add(self.ep_r[ta], ep_sumr, max_n)
    self.sumr[ta] = _add(self.sumr[ta], self.sumr[ta][-1] + ep_sumr, max_n)

    r_avg = ((ep - 1) * self.mar[ta][-1] + ep_sumr) / ep
    self.mar[ta] = _add(self.mar[ta], r_avg, max_n)


  def add_R(self, R, ta, step, global_step, ep, global_ep, max_n=0):
    self.R[ta] = _add(self.R[ta], R, max_n)
    self.sumR[ta] = _add(self.sumR[ta], self.sumR[ta][-1] + R, max_n)

    maR = ((step - 1) * self.maR[ta][-1] + R) / step
    self.maR[ta] = _add(self.maR[ta], maR, max_n)

  def dump_frames(self, fp, tag=None):
    _dump(fp, 'raw_frames_thread-{}.pickle'.format(tag), self.fr)

  def dump(self, fp, tag=None):
    # _dump(fp, 'last_alive_thread-{}.pickle'.format(tag), self.last_alive)
    # _dump(fp, 'r_idx_thread-{}.pickle'.format(tag), self.r_idx)
    # _dump(fp, 'R_idx_thread-{}.pickle'.format(tag), self.R_idx)
    # _dump(fp, 'ep_lens_thread-{}.pickle'.format(tag), self.local_ep_lens)
    # _dump(fp, 'env_steps_thread-{}.pickle'.format(tag), self.local_env_steps)
    # _dump(fp, 'global_eps_thread-{}.pickle'.format(tag), self.global_eps)
    # _dump(fp, 'global_env_steps_thread-{}.pickle'.format(tag), self.global_env_steps)
    _dump(fp, 'ep_r_thread-{}.pickle'.format(tag), self.ep_r)
    _dump(fp, 'allr_thread-{}.pickle'.format(tag), self.allr)
    # _dump(fp, 'mar_thread-{}.pickle'.format(tag), self.mar)
    # _dump(fp, 'sumr_thread-{}.pickle'.format(tag), self.sumr)
    _dump(fp, 'R_thread-{}.pickle'.format(tag), self.R)
    # _dump(fp, 'maR_thread-{}.pickle'.format(tag), self.maR)
    # _dump(fp, 'sumR_thread-{}.pickle'.format(tag), self.sumR)
    # _dump(fp, 'zs_thread-{}.pickle'.format(tag), self.zs)
    # _dump(fp, 'zs_g_thread-{}.pickle'.format(tag), self.zs_g)
    # _dump(fp, 'lp_thread-{}.pickle'.format(tag), self.lp)

  def pprint(self, tag=""):
    print(tag, "r_sum     ", " ".join([pp(sumr[-1]) for ta, sumr in self.sumr.items()]))
    print(tag, "r_avg (ep)", " ".join([pp(mar[-1]) for ta, mar in self.mar.items()]))
    print(tag, "V_avg     ", " ".join([pp(r) for ta, r in self.V.items()]))
    print(tag, "maR   (ep)", " ".join([pp(maRs[-1]) for ta, maRs in self.maR.items()]))
    print(tag, "sum_R     ", " ".join([pp(sumRs[-1]) for ta, sumRs in self.sumR.items()]))


class GlobalLog(Log):
  def __init__(self, initval=0):
    self.global_ep = Value('i', initval)
    self.global_env_step = Value('i', initval)
    self.lock = Lock()
  def _add_r(self, *args, **kwargs):
    self.add_r(*args, **kwargs)
  def _add_r_avg(self, *args, **kwargs):
    self.add_r_avg(*args, **kwargs)
  def _add_R(self, *args, **kwargs):
    self.add_R(*args, **kwargs)
  def add_env_step(self, n=1):
    with self.lock:
      self.global_env_step.value += n
      return self.global_env_step.value
  def get_global_env_step(self):
    with self.lock:
      return self.global_env_step.value
  def add_ep(self, n=1):
    with self.lock:
      self.global_ep.value += n
      return self.global_ep.value
  def get_global_ep(self):
    with self.lock:
      return self.global_ep.value


def _str_to_bool(s):
    """Convert string to bool (in argparse context)."""
    if s.lower() not in ['true', 'false']:
        raise ValueError('Need bool; got %r' % s)
    return {'true': True, 'false': False}[s.lower()]

def add_boolean_argument(parser, name, default=False):
    """Add a boolean argument to an ArgumentParser instance."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--' + name, nargs='?', default=default, const=True, type=_str_to_bool)
    group.add_argument('--no' + name, dest=name, action='store_false')
