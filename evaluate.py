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

from model import Seq2seq
from manager import ScalarMonitor, VariableMonitor

import time

from collections import defaultdict
from datareader import LorenzDataset

import os
import numpy as np
import util

import visdom
vis = visdom.Visdom(port=11111)

def visualize(rank, args, shared_q, status_q):
  """Summary

  Args:
      rank (TYPE): Description
      args (TYPE): Description
      shared_q (TYPE): Description
      status_q (TYPE): Description
  """
  names = ["train", "valid", "test"]
  idx = {"train": 0, "valid": 1, "test": 2}

  vw = defaultdict(lambda: None)

  batch_idx = defaultdict(lambda: 0)
  acc_loss = defaultdict(lambda: 0.)
  _correct_idx = defaultdict(lambda: 0)
  _correct = defaultdict(lambda: 0.)
  _num_ex = defaultdict(lambda: 0.)

  size = {"train": args.train_size, "valid": args.val_size, "test": args.test_size}
  freq = {"train": args.num_processes, "valid": 1, "test": 1}

  monitors = {}
  # for key in [i+"-loss" for i in names] + [i+"-correct" for i in names]:
  #   monitors[key] = ScalarMonitor(args, name=key)

  title = "{} Run-{}".format(args.dataset, args.exp)

  t0 = time.time()

  curr_ep = 0

  avg_loss = defaultdict(lambda: 0.)
  avg_acc = defaultdict(lambda: 0.)

  while True:

    checks = []

    d = shared_q.get()
    ds = d["dataset"]
    i = idx[ds]

    params = d["params"]

    legend = "batch-size {}".format(args.batch_size)

    x = batch_idx["train"] * args.batch_size / size["train"]

    t = time.time() - t0

    # Process loss
    if "loss" in d.keys():
      batch_idx[ds] += 1
      acc_loss[ds] += d["loss"]
      avg_loss[ds] = (avg_loss[ds] * (batch_idx[ds]-1) + d["loss"]) / batch_idx[ds]

      if batch_idx[ds] % freq[ds] == 0:

        y = acc_loss[ds] / freq[ds]

        key = "loss-x"
        vw[key] = util.update_visdom_scalar(vis, x, vw[key], y,
          xlabel="Time (epochs)",
          ylabel="Loss",
          title="{} loss".format(title),
          legend=[ds],
          name=ds)

        key = "loss-t"
        vw[key] = util.update_visdom_scalar(vis, t, vw[key], y,
          xlabel="Time (s)",
          ylabel="Loss",
          title="{} loss".format(title),
          legend=[ds],
          name=ds)

        acc_loss[ds] = 0

        y = avg_loss[ds]

        key = "avg_loss-x"
        vw[key] = util.update_visdom_scalar(vis, x, vw[key], y,
          xlabel="Time (epochs)",
          ylabel="Average Loss",
          title="{} average loss".format(title),
          legend=[ds],
          name=ds)

        key = "avg_loss-t"
        vw[key] = util.update_visdom_scalar(vis, t, vw[key], y,
          xlabel="Time (s)",
          ylabel="Average Loss",
          title="{} average loss".format(title),
          legend=[ds],
          name=ds)

    # Train stats
    if "lr" in d.keys() and batch_idx[ds] % (args.num_processes * 10) == 0:
      y = d["lr"]
      key = "lr"
      vw[key] = util.update_visdom_scalar(vis, x, vw[key], y,
        xlabel="Epochs",
        ylabel="Learning-rate",
        title="{} learning-rate".format(title),
        legend=[ds],
        name=ds)

    if batch_idx["train"] % args.num_processes == 0:
      y = args.batch_size

      key = "bs"
      vw[key] = util.update_visdom_scalar(vis, x, vw[key], y,
        xlabel="Epochs",
        ylabel="Batchsize",
        title="{} bs".format(title),
        legend=[ds],
        name=ds)

      y = args.batch_size * batch_idx["train"] / (time.time() - t0)

      key = "train-speed"
      vw[key] = util.update_visdom_scalar(vis, x, vw[key], y,
        xlabel="Epochs",
        ylabel="examples/s",
        title="{} speed".format(title),
        legend=[ds],
        name=ds)

    checks = []
    if any(checks):
      print("Some threshold met. Stopping stage!")
      for i in range(args.num_processes):
        status_q.put((i,{"m":True}))


def evaluate(rank, args, shared_model, shared_q, status_q, params, valid=False):
  if args.cuda:
    torch.cuda.manual_seed(args.seed + rank)
  else:
    torch.manual_seed(args.seed + rank)

  kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

  data_loader = torch.utils.data.DataLoader(
    LorenzDataset(args, args.data_dir, train=False),
    batch_size=args.batch_size, shuffle=True, **kwargs)

  model = Seq2seq(args, params)
  if args.cuda: model.cuda()
  model.eval()

  _T = torch.cuda if args.cuda else torch

  ds = "valid" if valid else "test"

  eval_idx = 0

  t00 = time.time()

  while True:
    # print("[eval] running", ds)

    model.load_state_dict(shared_model.state_dict())
    # print("[eval] loaded weights", shared_model.state_dict().keys())

    sum_loss = 0
    num_ex = 0
    N = args.val_size if valid else args.test_size
    Nb = int(N / args.batch_size) + 1

    # print("[eval] Running on", N, ds, "examples")

    t0 = time.time()

    for e, (data, target) in enumerate(data_loader):

      bs = data.size()[0]

      if bs < args.batch_size: continue

      data = data.type(torch.FloatTensor)
      data = torch.transpose(data, 0, 1)
      target = target.type(torch.FloatTensor)
      target = torch.transpose(target, 0, 1)
      if args.cuda: data, target = data.cuda(), target.cuda()
      data, target = Variable(data, volatile=True), Variable(target)

      output = model(data)

      loss = F.mse_loss(output, target, size_average=False).data[0] # sum up batch loss
      sum_loss += loss

      pred = output.data

      num_ex += bs

      if e >= Nb:
        break

    sum_loss /= num_ex

    # print('{} set: average xe: {:.4f}, wxe: {:.4f}, accuracy: {}/{} ({:.0f}%). Compute time: {:.2f} (t/batch {:.2f})'.format(ds, sum_xe, sum_wxe, correct, num_ex, 100. * correct / num_ex, time.time()-t0, (time.time()-t0) / Nb))

    shared_q.put({
      "thread": rank,
      "e": -1,
      "b": -1,
      "dataset": ds,
      "loss": sum_loss,
      "params": params,
      "total_time_this_epoch": time.time() - t0,
      "t": time.time() - t00
      })

    if valid:
      ckpt_dir = os.path.join(args.log_dir, "ckpt")
      f = os.path.join(ckpt_dir, "ckpt_step-eval-step-{}.pth.tar".format(eval_idx))
      util.save_checkpoint(shared_model, False, f=f)
      util.remove_older_than(ckpt_dir, n=10)

    # Visdom spatial
    # eval_idx += 1
    # if args.vis_image_freq > 0 and eval_idx % args.vis_image_freq == 0:
    #   shared_model.to_vis(vis, params, eval_idx)
