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

from datareader import ShotsDataset

import time
import os
import numpy as np
import scipy as sp
import util
import visdom
vis = visdom.Visdom(port=11111)


def train_single(rank, args, model, shared_q, params):
  if args.cuda:
    torch.cuda.manual_seed(args.seed + rank)
  else:
    torch.manual_seed(args.seed + rank)

  kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
  data_loader = torch.utils.data.DataLoader(
    ShotsDataset(args.data_dir, res_bh=params["scale_B"], res_def=params["scale_C"], train=True),
    batch_size=args.batch_size, shuffle=True, **kwargs)

  optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=args.lr,
                        momentum=args.momentum,
                        dampening=0.0,
                        nesterov=False,
                        weight_decay=args.l2)

  print("Training", [p.size() for p in model.parameters() if p.requires_grad])

  vw = {"train_loss": None, "valid_loss": None, "test_loss": None,
        "train_acc": None, "valid_acc": None, "test_acc": None}
  num_ep = {"train": 0., "valid": 0., "test": 0.}
  loss_per_ex = {"train": 0., "valid": 0., "test": 0.}
  batch_idx = {"train": 0, "valid": 0, "test": 0}
  update_idx = {"train": 0, "valid": 0, "test": 0}
  vis_ctrs = (vw, num_ep, loss_per_ex, batch_idx, update_idx)

  monitors = {}
  monitors["TrainLoss"] = ScalarMonitor(args, name="TrainLoss")
  monitors["ValidLoss"] = ScalarMonitor(args, name="ValidLoss")
  monitors["TestLoss"]  = ScalarMonitor(args, name="TestLoss")

  if args.tau_mu > 0 or args.tau_sd > 0:
    print("Using grad dist convergence with tau =", args.tau_mu, args.tau_sd)
    if hasattr(model, "_F"):
      monitors["F"] = VariableMonitor(model._F, args, name="F")
    if hasattr(model, "_B"):
      monitors["B"] = VariableMonitor(model._B, args, name="B")

  should_stop = False
  for epoch in range(1, params["epochs"] + 1):
    should_stop = train_epoch(rank, epoch, args, shared_q, model, data_loader, optimizer, params, vis_ctrs, monitors)
    if should_stop: break

  if False:
    test_loss = evaluate(rank, args, model, shared_q, params, vis_ctrs, test=True)


def train_epoch(rank, epoch, args, shared_q, model, data_loader, optimizer, params, vis_ctrs, monitors):
  _T = torch.cuda if args.cuda else torch

  num_correct = 0
  acc_loss = 0

  # Classes weighted by imbalance
  ratio = 1.0 # 14451032 / 1289692
  weight = torch.from_numpy(np.array([1.0, ratio])).type(torch.FloatTensor)
  if args.cuda: weight = weight.cuda()

  total_train_time = 0
  t0 = time.time()
  t1 = time.time()

  for batch_idx, (data, target) in enumerate(data_loader):

    checks = [False]

    if args.cuda:
      data, target = data.cuda(), target.cuda()
    data, target = Variable(data), Variable(target)

    if data.size()[0] < args.batch_size: break

    optimizer.zero_grad()

    output = model(data)

    # xent loss
    idA = torch.unsqueeze(target[:,0], 1)
    output = torch.gather(output, 1, idA)
    conj_output = 1 - output
    output = torch.cat([conj_output, output], 1)
    # log_output = torch.log(output + 1e-10)

    label_01 = target[:,2]
    # loss = F.nll_loss(log_output, label_01)
    loss = F.cross_entropy(output, label_01, weight=weight)

    e_loss = torch.exp(loss) # minimize the -likelihood (instead of -log-L)

    e_loss.backward()
    # torch.nn.utils.clip_grad_norm(model.parameters(), 40)
    optimizer.step()

    total_train_time += time.time() - t1


    # Validation loss
    if batch_idx % args.valid_freq == 0 and batch_idx > 0:
      valid_loss = evaluate(rank, args, model, shared_q, params, vis_ctrs, valid=True)
      monitors["ValidLoss"].add(valid_loss, s=params["stage"], e=epoch, t=batch_idx)
      # if args.early_stop_T > 0:
      #   if m.check_early_stop():
      #     print("Early stopping due to valid loss increasing.")
      #     return True


    # Training loss
    _loss = loss.data.cpu().numpy()[0] if args.cuda else loss.data.numpy()[0]
    acc_loss += _loss

    monitors["TrainLoss"].add(_loss, s=params["stage"], e=epoch, t=batch_idx)

    # Train Loss divergence
    if args.early_stop_T > 0:
      checks += [monitors["TrainLoss"].check_early_stop()]
      if checks[-1]: print("Early stopping due to train loss increasing.")

    # Loss convergence
    if batch_idx > args.T and args.T > 0 and args.finegrain:
      checks += [monitors["TrainLoss"].check()]


    # Write loss data
    if batch_idx % args.ckpt_freq == 0:
      monitors["TrainLoss"].write(d=os.path.join(args.log_dir, str(params["stage"]), "monitor"))
      monitors["ValidLoss"].write(d=os.path.join(args.log_dir, str(params["stage"]), "monitor"))


    # Accuracy
    pred = output.data.max(1, keepdim=True)[1]
    _label_01 = torch.unsqueeze(label_01, 1)

    correct = pred.eq(_label_01.data)
    if args.cuda: correct = correct.cpu()
    correct = correct.sum()
    num_correct += correct


    # Debug training
    if batch_idx % args.vis_scalar_freq == 0:
      print("avg P(y=1|x) =", torch.mean(output[:,1]).data[0])
      # print("label_01  ", label_01)
      print("data pos/neg =", torch.mean(label_01.type(_T.FloatTensor)).data[0])


    # Visdom
    if batch_idx % args.vis_scalar_freq == 0 and batch_idx > 0:
      x = epoch - 1 + batch_idx / data_loader.__len__()
      legend = "[S{} {}] Bh-{} Def-{} batch-size {}".format(
          params["stage"], params["type"],
          params["scale_B"], params["scale_C"], args.batch_size)

      acc_loss /= args.vis_scalar_freq
      print("Plotting (", x, acc_loss, ")")
      vis_ctrs[0]["train_loss"] = util.update_visdom(vis, x, vis_ctrs[0]["train_loss"],
        np.array([[ acc_loss ]]), # 1 x 1 tensor
        xlabel="Epochs", ylabel="Loss",
        title="[{}] Train loss".format(args.dataset),
        legend=[legend])
      acc_loss = 0

      num_correct /= (args.vis_scalar_freq * args.batch_size)
      print("Plotting acc (", x, num_correct, ")")
      vis_ctrs[0]["train_acc"] = util.update_visdom(vis, x, vis_ctrs[0]["train_acc"],
        np.array([[ num_correct ]]),
        xlabel="Epochs", ylabel="Accuracy",
        title="[{}] Train accuracy".format(args.dataset),
        legend=[legend])
      num_correct = 0

      print('T{} {:.2f}s ({:.3f}s/batch, {:3.1f}ex/s) | Train Epoch: {} [{}/{} examples ({:.1f}%) | ({} ps, total # ex: {})]\tLoss: {:.6f} (acc: {:.6f})'.format(
        rank,
        total_train_time,
        total_train_time / (batch_idx + 1),
        args.batch_size * (batch_idx + 1) / total_train_time,
        epoch,
        batch_idx * len(data), int(len(data_loader.dataset) / args.num_processes),
        100. * batch_idx * args.num_processes / len(data_loader),
        args.num_processes,
        len(data_loader.dataset),
        _loss, acc_loss))


    # Visdom spatial
    if args.vis_image_freq > 0 and batch_idx % args.vis_image_freq == 0 and params["type"] == "parafac":
      model.to_vis(vis, params, batch_idx)


    # Save ckpt
    if batch_idx % args.ckpt_freq == 0:
      ckpt_dir = os.path.join(args.log_dir, str(params["stage"]), "ckpt")
      is_best = False
      util.save_checkpoint(model, is_best,
        f=os.path.join(ckpt_dir, "ckpt_step-{}.pth.tar".format(batch_idx)))
      util.remove_older_than(ckpt_dir, n=100)


    # Spatial entropy
    if "B" in monitors.keys():
      m = monitors["B"]
      m.add(t=total_train_time)

      if batch_idx > args.T and args.T > 0 and args.finegrain:
        checks += [m.check()]

      if batch_idx % args.vis_scalar_freq == 0 and batch_idx > 0:
        m.to_vis(vis, params, epoch + batch_idx / args.train_size * args.batch_size)

      # if batch_idx % 100 == 0:
      #   m.write(d=os.path.join(args.log_dir, str(params["stage"]), "monitor"))


    condition = any(checks)
    if condition:
      return True

    t1 = time.time()

  return False


def visualize(rank, args, shared_model, shared_q, params, vis_ctrs):

  vw, num_ep, loss_per_batch, batch_idx, update_idx = vis_ctrs

  for key in loss_per_batch.keys():
    loss_per_batch[key] = 0.

  while True:
    (key, val) = shared_q.get()
    loss_per_batch[key] += val
    batch_idx[key] += 1

    if shared_q.empty():
      num_batches = args.vis_scalar_freq
      loss_per_batch[key] /= num_batches

      print("Plotting (", num_ep[key], loss_per_batch[key], ")", batch_idx[key] - 1)

      vw["loss"] = util.update_visdom(vis, num_ep[key], vw["loss"],
        np.array([[ loss_per_batch[key] ]]), # 1 x 1 tensor
        xlabel="Epochs", ylabel="Loss",
        title="[{}] {} loss".format(args.dataset, key),
        legend=["[S{} {}] Bh-{} Def-{} batch-size {}".format(
          params["stage"], params["type"],
          params["scale_B"], params["scale_C"], args.batch_size)])
      loss_per_batch[key] = 0.
      return


def evaluate(rank, args, shared_model, shared_q, params, vis_ctrs, valid=False,
  test=False):

  vw, num_ep, loss_per_ex, batch_idx, update_idx = vis_ctrs

  if args.cuda:
    torch.cuda.manual_seed(args.seed + rank)
  else:
    torch.manual_seed(args.seed + rank)

  assert(not (valid and test))

  kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
  data_loader = torch.utils.data.DataLoader(
    ShotsDataset(args.data_dir, res_bh=params["scale_B"], res_def=params["scale_C"], train=valid),
    batch_size=args.batch_size, shuffle=True, **kwargs)

  TensorModel = ParafacTensorModel if params["type"] == "parafac" else FullTensorModel
  model = TensorModel(args, params)
  if args.cuda: model.cuda()
  model.eval()

  model.load_state_dict(shared_model.state_dict())


  eval_loss = 0
  sum_xe = 0
  correct = 0
  num_valid_ex = 0
  N = args.val_size if valid else args.test_size

  # Classes weighted by imbalance
  ratio = 1.0 # 14451032 / 1289692
  weight = torch.from_numpy(np.array([1.0, ratio]))
  weight = weight.type(torch.FloatTensor)
  if args.cuda: weight = weight.cuda()

  print("Running on", N, "[valid]" if valid else "[test]", "examples")

  for data, target in data_loader:
    if args.cuda:
      data, target = data.cuda(), target.cuda()
    data, target = Variable(data, volatile=True), Variable(target)

    if data.size()[0] < args.batch_size: continue
    output = model(data)

    idA = torch.unsqueeze(target[:,0], 1)
    output = torch.gather(output, 1, idA)
    conj_output = 1 - output
    output = torch.cat([output, conj_output], 1)
    log_output = torch.log(output + 1e-10)

    label_01 = target[:,2]

    xe = F.nll_loss(log_output, label_01, size_average=False).data[0]
    sum_xe += xe

    wxe = F.cross_entropy(output, label_01, size_average=False, weight=weight).data[0] # sum up batch loss
    eval_loss += wxe

    # print("eval:", xe, wxe, "(", eval_loss, ")")

    pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
    label_01 = torch.unsqueeze(label_01, 1)
    correct += pred.eq(label_01.data).cpu().sum()

    num_valid_ex += data.size()[0]

  print("Result: eval loss", eval_loss, "(", num_valid_ex, "examples")
  sum_xe /= num_valid_ex
  eval_loss /= num_valid_ex # * args.batch_size

  print('\n{} set: Average xe: {:.4f}, wxe: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    "Valid" if valid else "Test",
    sum_xe, eval_loss, correct, num_valid_ex, 100. * correct / num_valid_ex))


  legend = "[S{} {}] Scales {}-{} batch-size {}".format(
      params["stage"],
      params["type"],
      "valid" if valid else "test",
      params["scale_B"],
      params["scale_C"],
      args.batch_size)

  vw_name = "valid_loss" if valid else "test_loss"
  vw[vw_name] = util.update_visdom(vis, num_ep["train"], vw[vw_name],
    np.array( [[ sum_xe, eval_loss ]] ), # 1 x 1 tensor
    xlabel="Epochs", ylabel="Loss",
    title="[{}] {} loss".format(args.dataset, "Valid" if valid else "Test"),
    legend=["xe {}".format(legend),
            "wxe {}".format(legend)])

  return eval_loss
