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
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau
import torch.multiprocessing as _mp
mp = _mp.get_context('spawn')

from torch.multiprocessing import Manager

from collections import OrderedDict, defaultdict

import numpy as np
import time
import dill as pickle
import json
import os
import util

import experiments

from model import Seq2seq
from evaluate import evaluate, visualize
from train import train
from shared_optim import SharedAdam

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--exp', type=int, default=64, metavar='N')
parser.add_argument('--sess', default="", metavar='N')
parser.add_argument('--train-size', type=int, default=64, metavar='N')
parser.add_argument('--val-size', type=int, default=64, metavar='N')
parser.add_argument('--test-size', type=int, default=64, metavar='N')
parser.add_argument('--batch-size', type=int, default=64, metavar='N')
parser.add_argument('--input-len', type=int, default=10, metavar='N')
parser.add_argument('--output-len', type=int, default=90, metavar='N')
parser.add_argument('--state-dim', type=int, default=1, metavar='N')
parser.add_argument('--hidden-size', type=int, default=64, metavar='N')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                                        help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                                        help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                                        help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=True,
                                        help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                                        help='random seed (default: 1)')
parser.add_argument('--vis-scalar-freq', type=int, default=10, metavar='N',
                                        help='how many batches to wait before logging training status')
parser.add_argument('--ckpt-freq', type=int, default=100, metavar='N',
                                        help='how many batches to wait before logging training status')
parser.add_argument('--num-processes', type=int, default=2, metavar='N')
parser.add_argument('--data-dir', type=str, default="./data/lorenz.npy", metavar='N')
parser.add_argument('--vis-image-freq', type=int, default=0, metavar='N')
parser.add_argument('--valid-freq', type=int, default=500, metavar='N')
parser.add_argument('--log-dir', default="/tmp/stephan/logs/multires", help='')
parser.add_argument('--prev-ckpt', default="", help='')
parser.add_argument('--algo', default="multires", help='')
parser.add_argument('--dataset', default="full", help='')
parser.add_argument('--model-type', default="full", help='')
parser.add_argument('--l2', type=float, default=0.1, metavar='LR')
parser.add_argument('--opt', default="sgd")

util.add_boolean_argument(parser, "early-stop")

args = parser.parse_args()
f = open(str(args.log_dir) + "model.log", "w")
f.write(str(args))
f.close()
args.cuda = not args.no_cuda and torch.cuda.is_available()

def create_shared_model(params, from_ckpt=None):

    print("Creating SHARED model")

    shared_model = Seq2seq(args, params)
    shared_model.share_memory()
    shared_model.train()
    if args.cuda:
        shared_model.cuda()

    if from_ckpt:
        import glob
        list_of_files = glob.glob(os.path.join(from_ckpt, '*.pth.tar'))
        if list_of_files:
            ckpt_fn = max(list_of_files, key=os.path.getctime)
            latest_ckpt = torch.load(ckpt_fn)
            shared_model.load_state_dict(latest_ckpt)
            print("Loading weights from", ckpt_fn)
        else:
            print("No checkpoints found in", from_ckpt)

    print("Creating SHARED optimizer", args.opt, "with lr", init_lr)

    if args.opt == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, shared_model.parameters()),
            lr=init_lr,
            momentum=args.momentum,
            weight_decay=args.l2)
    elif args.opt == "adam":
        optimizer = SharedAdam(
            filter(lambda p: p.requires_grad, shared_model.parameters()),
            lr=init_lr,
            betas=(0.9, 0.999), # (0.,0.), #
            weight_decay=args.l2)
        optimizer.share_memory()
    else:
        print("no valid opt"); exit()

    # Decays lr each batch
    # scheduler = StepLR(optimizer, 1, gamma=0.9, last_epoch=-1)
    # scheduler = ExponentialLR(optimizer, gamma=0.9999, last_epoch=-1)
    scheduler = None
    print("Not using scheduler")

    return shared_model, optimizer, scheduler


if __name__ == "__main__":
    if args.cuda and torch.cuda.is_available(): print("Using CUDA")

    shared_q = mp.Queue()
    status_q = mp.Queue()

    t0 = time.time()

    params = experiments.get_params(args)

    util.maybe_create(os.path.join(args.log_dir, "ckpt"))
    util.maybe_create(os.path.join(args.log_dir, "monitor"))

    pickle.dump(args, open(os.path.join(args.log_dir, "args"),'wb'))
    json.dump(params, open(os.path.join(args.log_dir, "params"),'w'))

    print(args.algo, "training", params["type"], "with", args.num_processes, "threads")

    shared_model = None
    optimizer = scheduler = None

    init_lr = args.lr

    if args.num_processes > 1:
        shared_model, optimizer, scheduler = create_shared_model(params, from_ckpt=args.prev_ckpt)

    processes = []
    for rank in range(args.num_processes):
        print("Starting", rank)
        p = mp.Process(target=train,
                                     args=(rank, args, shared_model, optimizer, scheduler,
                                         init_lr, shared_q, status_q, params, t0),
                                     kwargs={}) #  args.num_processes == 1
        p.start()
        processes.append(p)
        print("here")
    eval_processes = []

    # Visualization process 

    # p = mp.Process(target=visualize, args=(args.num_processes+2, args, shared_q, status_q))
    # p.start()
    # eval_processes.append(p)

    if args.num_processes != 1:

        p = mp.Process(target=evaluate,
                                     args=(args.num_processes, args, shared_model, shared_q, status_q, params),
                                     kwargs={"valid": True})
        p.start()
        eval_processes.append(p)

        p = mp.Process(target=evaluate,
                                     args=(args.num_processes+1, args, shared_model, shared_q, status_q, params),
                                     kwargs={"valid": False})
        p.start()
        eval_processes.append(p)

    for p in processes:
        p.join()

    print("Terminating evaluation process", eval_processes)
    for p in eval_processes:
        p.terminate()
        p.join()
