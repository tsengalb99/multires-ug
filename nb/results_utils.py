import os, sys, time, struct, json, dill as pickle
from os.path import join as pj
from collections import defaultdict
import numpy as np, scipy, sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import torch

MIN_STEPS = 10

def get_ma(ys, gamma = 0.99, verbose=False):
  y_ma = []
  ma = 0.
  def norm(T, verbose=False):
    return (1-gamma**T) / (1-gamma)
  for e,y in enumerate(ys):
    ma = (norm(e) * ma * gamma + y) / norm(e+1)
    y_ma += [ma]
  return y_ma

def select_res(res, verbose=False):
  """Summary

  Args:
      res (TYPE): Description
      verbose (bool, optional): Description
  """
  for run, vv in res.items():

    stage_fp = sorted(vv.keys())

    if verbose: print("Run", run, "stage_fp", stage_fp)

    num_points = 0
    for fp in stage_fp:
      v = vv[fp]

      if "monitor" in fp:
        for l,w in v.items():
          if not ("train" in l and "loss" in l):
            continue

          if "keys" in w.keys():
            for stage in w["keys"].keys():
              x = w["keys"][stage]
              num_points += len(x)
          elif "t" in w.keys() and isinstance(w["t"], list):
            for stage in w["vals"].keys():
              t = w["t"]
              print("Fixme!")
              num_points += len(t)
          elif "t" in w.keys():
            for stage in w["vals"].keys():
              t = w["t"][stage]
              num_points += len(t)

    vv["plot_this_run"] = num_points >= MIN_STEPS
    if verbose: print("Run", run, "has", num_points, "points. Plot:", vv["plot_this_run"])


def get_data_from_sessions(sessions, verbose=False):

  res_all_sessions = {}
  for session in sessions:
    if not os.path.exists(session): continue

    runs = os.listdir(session)

    res = {}
    for run in runs:

      run_fp = pj(session,run)
      if not os.path.isdir(run_fp): continue

      if verbose: print("Loading data from", run_fp)

      res[run] = defaultdict(lambda: defaultdict())
      for i in os.walk(run_fp):
        fp, subdirs, files = i

        if "ckpt" in fp:
          pass
        elif "monitor" in fp:
          for f in files:
            if "loss" in f or "correct" in f:
              with open(pj(fp, f), "rb") as fi:
                try:
                  res[run][fp][f] = pickle.load(fi)
                except:
                  print("Empty file", fp, f)
            if "B" in f:
              with open(pj(fp, f), "rb") as fi:
                try:
                  res[run][fp][f] = pickle.load(fi)
                except:
                  print("Empty file", fp, f)
        elif "params" in files:
          with open(pj(fp, "args"), "rb") as f:
            args = pickle.load(f)
            res[run][fp]["args"] = args
          with open(pj(fp, "params"), "r") as f:
            params = json.load(f)
            res[run][fp]["params"] = params

    res_all_sessions[session] = res

  return res_all_sessions

def plot_run(run_data, verbose=False):
  pal = sns.color_palette("Paired")
  args, params = None, None
  t0 = 0

  fig, axes = plt.subplots(1, 2, sharex=True, sharey=False, squeeze=False, figsize=(12,4))

  ax = axes[0][0]
  ax.set_title("Training Loss")
  ax.set_xlabel("Time")
  ax.set_ylabel("Negative Log-Likelihood")

  ax = axes[0][1]
  ax.set_title("Training Precision")
  ax.set_xlabel("Time")
  ax.set_ylabel("Accuracy")

  # This is a hack: we need to fetch args, params first, so we ensure they're encountered first.
  ordered_k = sorted(run_data.keys())

  ts = {}     
  xs = {}
  ys = {}
    
  for stage in ordered_k:
    
    # Fetch numbers from exp, session, run, stage
    if "monitor" in stage:

      stage_data = run_data[stage]

      for l, w in stage_data.items():

        if "train" in l and "loss" in l:
          col = 0
        elif "train" in l and "correct" in l:
          col = 1
        else:
          continue

        if verbose: print(stage, l, pal[idx])

        num_points = 0

        if "keys" in w.keys():
          print("Manager had [keys]")
          assert False, "Old data format!"
          for stage in w["keys"].keys():
            x = w["keys"][stage]
            t = x
            y = get_ma(w["vals"][stage], gamma=0.9)
            num_points += len(x)
        elif "t" in w.keys() and isinstance(w["t"], list):          
          print("Manager had [t] as a list")
          assert False, "Old data format!"
          for stage in w["vals"].keys():
            t = w["t"]
            x = t
            y = get_ma(w["vals"][stage], gamma=0.9)
            num_points += len(t)
        elif "t" in w.keys():
          # print("Manager had [t] as a dict {stage: ... }")
          for stage in w["vals"].keys():
            t = w["t"][stage]
            x = w["b"][stage]
            y = get_ma(w["vals"][stage], gamma=0.9)
            num_points += len(t)
        
        y = get_ma(y, gamma = 0.99)
        
        c = pal[stage]
        label = "Stage {} {} {}".format(stage, args.tau_mu, args.tau_sd)
        axes[0][col].plot(t,y, c=c, linewidth=0.1, alpha=0.9)
        axes[0][col].scatter(t,y, c=c, label=label, s=2.0, alpha=1.0)
    
        ts[stage] = t
        xs[stage] = x
        ys[stage] = y
    
    # Fetch meta-data
    elif stage != "plot_this_run":
      stage_data = run_data[stage]
      args = stage_data["args"]
      params = stage_data["params"]


  for ax in axes[0]:
    ax.legend()

  return fig, args, params, ts, xs, ys



def extract_data(run_data, verbose=False):
  
  args, params = None, None
  t0 = 0
  
  # This is a hack: we need to fetch args, params first, so we ensure they're encountered first.
  ordered_k = sorted(run_data.keys())

  ts = {}     
  xs = {}
  loss = {}
  acc = {}
    
  for stage in ordered_k:
    
    # Fetch numbers from exp, session, run, stage
    if "monitor" in stage:

      stage_data = run_data[stage]

      for l, w in stage_data.items():

        if not "train" in l and ("loss" in l or "correct" in l):
          continue

        if verbose: print(stage, l, pal[idx])

        num_points = 0

        if "keys" in w.keys():
          print("Manager had [keys]")
          assert False, "Old data format!"
          for stage in w["keys"].keys():
            x = w["keys"][stage]
            t = x
            y = get_ma(w["vals"][stage], gamma=0.9)
            num_points += len(x)
        elif "t" in w.keys() and isinstance(w["t"], list):          
          print("Manager had [t] as a list")
          assert False, "Old data format!"
          for stage in w["vals"].keys():
            t = w["t"]
            x = t
            y = get_ma(w["vals"][stage], gamma=0.9)
            num_points += len(t)
        elif "t" in w.keys():
          # print("Manager had [t] as a dict {stage: ... }")
          for stage in w["vals"].keys():
            t = w["t"][stage]
            x = w["b"][stage]
            y = get_ma(w["vals"][stage], gamma=0.9)
            num_points += len(t)
        
        y = get_ma(y, gamma = 0.99)
        
        ts[stage] = t
        xs[stage] = x
        
        if "loss" in l:
          loss[stage] = y
        elif "correct" in l:
          acc[stage] = y
        
    
    # Fetch meta-data
    elif stage != "plot_this_run":
      stage_data = run_data[stage]
      args = stage_data["args"]
      params = stage_data["params"]

  return args, params, ts, xs, loss, acc


def interleave(d):
    """d is a (x,y) where x,y are arrays"""
    ptrs = [0] * len(d)

    res = [[], []]

    min_T = min([len(x) for x,y in d])
    most_recent = [0] * len(d)

    all_xs = []
    all_ys = []
    min_ys = []
    max_ys = []
    avg_ys = []
    
    while all([p < min_T for p in ptrs]):

        curr_xs = [d[i][0][ptrs[i]] for i in range(len(d))]
        curr_ys = [d[i][1][ptrs[i]] for i in range(len(d))]

        min_x = min(curr_xs)
        idx = curr_xs.index(min_x)
        # print(ptrs)
        all_xs += [min_x]
        all_ys += [curr_ys]

        min_ys += [min(curr_ys)]
        max_ys += [max(curr_ys)]
        avg_ys += [sum(curr_ys)/len(curr_ys)]

        ptrs[idx] += 1
    return all_xs, all_ys, min_ys, max_ys, avg_ys