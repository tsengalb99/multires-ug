from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.nn import Parameter
from torch.autograd import Variable
from torch.sparse import FloatTensor as STensor
from torch.cuda.sparse import FloatTensor as CudaSTensor
from torch.utils.data import Dataset

from collections import defaultdict
from model import Seq2seq

import time
import os
import numpy as np
import scipy as sp
import scipy.stats as sps
import util
import visdom
import queue
import dill as pickle

class ScalarMonitor():
  def __init__(self, args, name=""):
    self.name = name
    self.args = args
    self.tau_loss = args.tau_loss
    self.cuda = args.cuda
    self.q = queue.Queue(maxsize=args.T)
    self.s = []
    self.e = defaultdict(lambda: [])
    self.t = defaultdict(lambda: [])
    self.b = defaultdict(lambda: [])
    self.vals = defaultdict(lambda: [])

    self.mu = 0
    self.sd = 0
    self.i = 0
    self.vw = None
    self.curr_s = 0

  def add(self, val, key="", s=0, e=0, b=0, t=0, lr=1):
    self.s += [s]
    self.e[s] += [e]
    self.t[s] += [t]
    self.b[s] += [b]
    self.vals[s] += [val]
    self.i += 1

    if self.q.full():
      self.q.get()

    if s >= self.curr_s:
      self.curr_s = s

      if self.q.full():
        self.q.get()

      self.q.put(val / lr if lr > 0. else val)
    else:
      print("Discarding", key, val, s, e, b, t, "This value has come after some thread moved to the next stage.")

  def check_div(self):

    if self.q.qsize() < self.args.early_stop_T:
      return False

    gs = list(self.q.queue)
    g = np.array(gs[-self.args.early_stop_T:-1])
    last_val = gs[-1]
    _mu = np.mean(g)

    tau = 0.05
    check = last_val - _mu > tau
    if check:
      print(self.name, "early stopping: loss going up", last_val, ">", _mu, tau)

    return check

  def check_conv(self):

    if self.q.qsize() < self.args.T:
      return False

    gs = list(self.q.queue)
    g = np.array(gs[:-1])
    last_val = gs[-1]
    _mu = np.mean(g)

    check = abs(last_val - _mu) < self.tau_loss

    print("Check loss", last_val, _mu, abs(last_val - _mu), self.tau_loss, abs(last_val - _mu) < self.tau_loss)

    if check:
      print(self.name, "threshold exceeded", last_val, _mu, abs(last_val - _mu), self.tau_loss)
    return check

  def write(self, d="", suffix=""):
    f = os.path.join(d, self.name + suffix + ".pkl")
    with open(f, "wb") as o:
      pickle.dump({"b": self.b, "s": self.s, "e": self.e, "t": self.t, "vals": self.vals}, o)
    # print(self.name, "Writing", len(self.vals), "vals to", f)

class VariableMonitor():
  def __init__(self, v, args, name=""):
    self.v = v
    self.name = name
    self.args = args
    self.mu = 0
    self.mu_std = 0
    self.sd = 0
    self.sd_std = 0
    self.mu_arr = 0
    self.sd_arr = 0
    self.tau_mu = args.tau_mu
    self.tau_sd = args.tau_sd
    self.tau_ent = args.tau_ent
    self.cuda = args.cuda
    self.s = []
    self.e = defaultdict(lambda: [])
    self.t = defaultdict(lambda: [])
    self.b = defaultdict(lambda: [])
    self.i = 0
    self.vals = defaultdict(lambda: [])

    self.q = queue.Queue(maxsize=args.T)
    self.curr_grad = None

    self.count_q = queue.Queue(maxsize=args.T)

    vT = self.v.data
    if args.cuda: vT = vT.cpu()
    vT = vT.numpy()

    self.n = np.zeros((*vT.shape, self.args.nbins))
    self.p = np.zeros((*vT.shape, self.args.nbins))
    self.log_p = np.zeros((*vT.shape, self.args.nbins))
    self.ent = np.zeros_like(vT)
    self.ent_mu = 0
    self.ent_sd = 0
    self.ent_min = 0
    self.ent_max = 0
    self.ent_num_pos = 0

    self.gd_min = 0
    self.gd_max = 0

    self.vw_mu = None
    self.vw_sd = None
    self.vw_e = None
    self.vw_gd = None

  def add(self, s=0, e=0, b=0, t=0):
    self.s += [s]
    self.e[s] += [e]
    self.t[s] += [t]
    self.b[s] += [b]
    self.i += 1

    self._add_grad()
    self._add_count()

    self.vals[s] += [(self.ent_mu, self.ent_sd, self.ent_min, self.ent_max, self.ent_num_pos)]

  def check_sd_div(self):

    if self.q.qsize() < self.args.T:
      return False

    g = np.array(list(self.q.queue))

    self.sd_arr = np.std(g, axis=0)
    self.sd = np.mean(self.sd_arr)
    self.sd_std = np.std(self.sd_arr)

    num_elem = np.prod(self.sd_arr.shape)

    self.sd_num_pos = 0.
    sd_num_pos = np.sum(self.sd_arr > self.tau_sd)
    self.sd_num_pos = sd_num_pos / num_elem

    print(self.name, "gd min", self.gd_min, "max", self.gd_max)
    print(self.name, "sd", self.sd, "sd_sd", self.sd_std,
                     "tau_sd", self.tau_sd,
                     "sd #ok", self.sd_num_pos, "%")

    meets_cond = self.sd_num_pos > self.args.ent_quantile

    if meets_cond:
      print(self.args.ent_quantile, "Quantile threshold [", self.name, "] reached sd >", self.tau_sd)
    return meets_cond

  def check_mu_sd(self):

    if self.q.qsize() < self.args.T:
      return False

    g = np.array(list(self.q.queue))

    self.mu_arr = np.mean(g, axis=0)
    self.mu = np.mean(self.mu_arr)
    self.mu_std = np.std(self.mu_arr)

    self.sd_arr = np.std(g, axis=0)
    self.sd = np.mean(self.sd_arr)
    self.sd_std = np.std(self.sd_arr)

    num_elem = np.prod(self.mu_arr.shape)

    self.mu_num_pos = 0.
    mu_num_pos = np.sum(self.mu_arr < self.tau_mu)
    self.mu_num_pos = mu_num_pos / num_elem

    self.sd_num_pos = 0.
    sd_num_pos = np.sum(self.sd_arr > self.tau_sd)
    self.sd_num_pos = sd_num_pos / num_elem

    print(self.name, "gd min", self.gd_min, "max", self.gd_max)
    print(self.name, "mu", self.mu, "mu_sd", self.mu_std,
                     "sd", self.sd, "sd_sd", self.sd_std,
                     "tau_mu", self.tau_mu, "tau_sd", self.tau_sd,
                     "mu #ok", self.mu_num_pos, "%"
                     "sd #ok", self.sd_num_pos, "%")

    meets_cond = self.mu_num_pos > self.args.ent_quantile and self.sd_num_pos > self.args.ent_quantile

    if meets_cond:
      print(self.args.ent_quantile, "Quantile threshold [", self.name, "] reached mu <", self.tau_mu, "&& sd >", self.tau_sd)
    return meets_cond

  def check_ent(self):

    if self.q.qsize() < self.args.T:
      return False

    t0 = time.time()

    # Update the entropy
    N = self.i # self.q.qsize() # total number of counts
    self.p = self.n / N # self.args.nbins
    self.log_p = np.log(self.p + 1e-10)
    ee = self.p * self.log_p

    print("N", N, "freq", self.p[0,0,0])

    self.ent = - np.sum(ee, axis=self.ent.ndim - 1)
    self.ent_min = np.min(self.ent)
    self.ent_max = np.max(self.ent)
    self.ent_mu = np.mean(self.ent)
    self.ent_sd = np.std(self.ent)
    self.ent_num_pos = 0.

    num_pos = np.sum(self.ent > self.tau_ent)
    num_elem = np.prod(self.ent.shape)

    self.ent_num_pos = num_pos / num_elem

    print(self.name, "gd min", self.gd_min, "max", self.gd_max)
    print(self.name, "ent min", self.ent_min, "max", self.ent_max,
                     "mu", self.ent_mu, "sd", self.ent_sd,
                     "tau", self.tau_ent,
                     "top-k%", self.args.ent_quantile,
                     "#ok", num_pos, num_pos / num_elem, "%")

    num_nonneg = np.sum(self.ent > 0.)
    print("% nonzero?", num_nonneg / num_elem, num_nonneg, num_elem)

    # meets_cond = self.ent_mu > self.tau_ent

    meets_cond = self.ent_num_pos > self.args.ent_quantile

    if meets_cond:
      print(self.args.ent_quantile, "\% of weights (quantile threshold) [", self.name, "] reached tau ent >", self.tau_ent)
    return meets_cond

  def _add_grad(self):
    if self.v.grad is not None:
      if self.q.full():
        self.q.get()
      gd = self.v.grad.data
      if self.cuda: gd = gd.cpu()
      gd = gd.numpy()

      self.gd_min = np.min(gd)
      self.gd_max = np.max(gd)

      self.q.put(gd)
      self.curr_grad = gd

  def _add_count(self):

    plus_counts = np.zeros_like(self.n)

    gd = self.curr_grad

    lo = self.args.ent_lo
    hi = self.args.ent_hi
    step = (hi - lo) / (self.args.nbins + 1)

    bin_edges = np.arange(lo, hi, step)
    bin_idx = np.digitize(gd, bin_edges[1:-1])

    n_values = self.args.nbins
    plus_counts = np.eye(n_values)[bin_idx]

    self.n += plus_counts # add the count

    # self.count_q.put(plus_counts) # store the count

  def write(self, d="", suffix=""):
    vals = list(self.q.queue)
    f = os.path.join(d, self.name + suffix + ".pkl")
    with open(f, "wb") as o:
      pickle.dump({"b": self.b, "s": self.s, "e": self.e, "t": self.t, "vals": self.vals}, o)
    # print(self.name, "Writing", len(vals), "vals to", f)

  def to_vis(self, vis, params, step, mu=False, sd=False, ent=False):
    """Summary

    Args:
        vis (TYPE): Description
        params (TYPE): Description
        step (TYPE): Description
        mu (bool, optional): Description
        sd (bool, optional): Description
        ent (bool, optional): Description
    """
    title = "{} S{}".format(self.args.dataset, params["stage"])

    legend = "[S{} {}] Bh-{} Def-{} batch-size {}".format(
          params["stage"], params["type"],
          params["scale_B"], params["scale_C"], self.args.batch_size)


    if mu:
      self.vw_mu = util.update_visdom(
        vis,
        step,
        self.vw_mu,
          np.array( [[self.mu, self.mu_std]] ), # 1 x 2 tensor
          xlabel="Epochs", ylabel="",
          title="{} P(grad) mu tau={} T={}".format(
            title, self.args.tau_mu, self.args.T),
          legend=["mu {}".format(legend),
                  "mu-sd {}".format(legend)])
    if sd:
      self.vw_sd = util.update_visdom(
        vis,
        step,
        self.vw_sd,
          np.array( [[self.sd, self.sd_std]] ), # 1 x 2 tensor
          xlabel="Epochs", ylabel="",
          title="{} P(grad) sd tau={} T={}".format(
            title, self.args.tau_sd, self.args.T),
          legend=["sd {}".format(legend),
                  "sd-sd {}".format(legend)])

    if ent:
      self.vw_e = util.update_visdom(
        vis,
        step,
        self.vw_e,
          np.array( [[self.ent_mu, self.ent_sd, self.ent_min, self.ent_max, self.ent_num_pos]] ), # 1 x 2 tensor
          xlabel="Epochs", ylabel="",
          title="{} P(grad) ent tau={} T={}".format(
            title, self.args.tau_ent, self.args.T),
          legend=["e {}".format(legend),
                  "e-sd {}".format(legend),
                  "e-min {}".format(legend),
                  "e-max {}".format(legend),
                  "e-pos {}".format(legend)])

      self.vw_gd = util.update_visdom(
        vis,
        step,
        self.vw_gd,
          np.array( [[self.gd_min, self.gd_max]] ), # 1 x 2 tensor
          xlabel="Epochs", ylabel="",
          title="{} P(grad) ent tau={} T={}".format(
            title, self.args.tau_ent, self.args.T),
          legend=["raw gd-min {}".format(legend),
                  "raw gd-max {}".format(legend)])
