from __future__ import print_function
import argparse
import random
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

#sdtw 
#from sdtw import SoftDTW
#from sdtw.distance import SquaredEuclidean

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau

import matplotlib

from collections import OrderedDict, defaultdict

from model import Seq2seq
from manager import ScalarMonitor, VariableMonitor

import time

from datareader import LorenzDataset

import os
import numpy as np
import util



def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                                                 shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def train(rank, args, shared_model, optimizer, scheduler, init_lr,
    shared_q, status_q, params, t0):
    if args.cuda:
        torch.cuda.manual_seed(args.seed + rank)
    else:
        torch.manual_seed(args.seed + rank)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    print ("LORENZ HERE")
    a = LorenzDataset(args, args.data_dir, train=True)
    print(a)
    data_loader = torch.utils.data.DataLoader(
        LorenzDataset(args, args.data_dir, train=True),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    model = Seq2seq(args, params)
    model.train()
    if args.cuda:
        model.cuda()

    if not optimizer:
        if rank == 0: print("Creating optimizer", args.opt, "with lr", init_lr)

        if args.opt == "sgd":
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=init_lr,
                momentum=args.momentum,
                weight_decay=args.l2)
        elif args.opt == "adam":
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=init_lr,
                betas=(0.9, 0.999), # (0.,0.), #
                weight_decay=args.l2)
        else:
            print("no valid opt"); exit()

    else:
        if rank == 0: print("Train thread using SHARED optimizer!")

    if not scheduler:
        # scheduler = StepLR(optimizer, 1, gamma=0.9, last_epoch=-1)
        # scheduler = ExponentialLR(optimizer, gamma=0.9999, last_epoch=-1)
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
        #   patience=1000, # num batches
        #   verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0,
        #   min_lr=0, eps=1e-08)
        scheduler = None
        print("Not using scheduler")
    else:
        if rank == 0: print("MAYBE Train thread using SHARED scheduler!")

    monitors = {}

    if rank == 0:
        if shared_model: print(rank, "Using SHARED_MODEL", [p.size() for p in shared_model.parameters() if p.requires_grad])
        if model: print(rank, "Rollouts using  model", [p.size() for p in model.parameters() if p.requires_grad])

    # Train
    for epoch in range(1, params["epochs"] + 1):
        should_stop = train_epoch(rank, epoch, args, shared_q, status_q,
            shared_model, model, data_loader, optimizer, scheduler, init_lr, params,
            monitors, t0)
        if should_stop: break


    # Send data back to masters
    if rank == 0:
        state_dict = model.state_dict()
        torch.save(state_dict, '/tmp/model.pth.tar')
        torch.save(optimizer.state, '/tmp/opt_state.pth.tar')


def train_epoch(rank, epoch, args, shared_q, status_q, shared_model, model,
    data_loader, optimizer, scheduler, init_lr, params, monitors, t0):
    _T = torch.cuda if args.cuda else torch

    total_time_this_epoch = 0
    t00 = time.time()

    print(rank, "@epoch", epoch)


    savedImg = False
    for batch_idx, (data, target) in enumerate(data_loader):
#        print(data, target)

        # Decay lr. Careful: each thread controls its own lr
        if scheduler: scheduler.step()

        if not status_q.empty():
            r,m = status_q.get()
            if r == rank and m["m"]:
                print(rank, "Received KILL signal from status_q")
                return True
            else:
                status_q.put((r,m))

        if shared_model: model.load_state_dict(shared_model.state_dict())

        data = data.type(torch.FloatTensor)
        data = torch.transpose(data, 0, 1) # num_step x batch x dim

        pdata =False
        if(pdata):
            for i in range(len(data)):
                for k in range(3):
                    print(data[i][0][k], end = ' ')
                print()

        target = target.type(torch.FloatTensor)
        target = torch.transpose(target, 0, 1)
        if args.cuda: data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
#        print(len(target), len(data))
        if data.size()[1] < args.batch_size: break

        optimizer.zero_grad()

        output = model(data, target)
        if(epoch == 10 and not savedImg):
            savedImg = True
            outputArr = output.data.numpy()
            targetArr = target.data.numpy()
            #plot a random image
            
            randImage = int(random.random()*8)
            from mpl_toolkits import mplot3d
            import matplotlib.pyplot as plt
            xo = outputArr[:, randImage, 0]
            yo = outputArr[:, randImage, 1]
            zo = outputArr[:, randImage, 2]
            
            xt = targetArr[:, randImage, 0]
            yt = targetArr[:, randImage, 1]
            zt = targetArr[:, randImage, 2]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot3D(xo, yo, zt)
            ax.plot3D(xt, yt, zt)
            fig.savefig("img/plot.png")
            
        #loss on only points
        outputChop = np.ndarray.tolist(output.data.cpu().numpy())
        targetChop = np.ndarray.tolist(target.data.cpu().numpy())
        for i in range(len(outputChop)):
            for j in range(len(outputChop[i])):
                outputChop[i][j] = outputChop[i][j][0:3]
                targetChop[i][j] = targetChop[i][j][0:3]
        loss2 = F.mse_loss(Variable(torch.FloatTensor(outputChop)), Variable(torch.FloatTensor(targetChop)))
        loss = F.mse_loss(output, target)

        tn = target.data.cpu().numpy()
        on = output.data.cpu().numpy()
        #5x8x6

        loss.backward()

        if shared_model: ensure_shared_grads(model, shared_model)

        optimizer.step()

        # Training loss
        _loss = loss.data.cpu().numpy()[0] if args.cuda else loss.data.numpy()[0]
        _loss2 = loss2.data.cpu().numpy()[0] if args.cuda else loss2.data.numpy()[0]
        total_time_this_epoch = time.time() - t00

        d = {
            "thread": rank,
            "e": epoch,
            "b": batch_idx,
            "dataset": "train",
            "num_ex": args.batch_size,
            "loss": _loss,
            "lr": optimizer.param_groups[0]["lr"],
            "params": params,
            "total_time_this_epoch": total_time_this_epoch,
            "t": time.time() - t0
            }
        shared_q.put(d)

        if batch_idx % args.vis_scalar_freq == 0:
            if rank == 0:
                print(args.dataset, args.model_type, args.algo, args.opt,'T{} {:.2f}s ({:.3f}s/batch, {:3.1f}ex/s) | Train Epoch: {} [{}/{} examples ({:.1f}%) | ({} ps, total # ex: {})] Loss: {:.6f}'.format(
                    rank,
                    total_time_this_epoch,
                    total_time_this_epoch / (batch_idx + 1),
                    args.batch_size * (batch_idx + 1) / total_time_this_epoch,
                    epoch,
                    batch_idx * len(data), int(len(data_loader.dataset) / args.num_processes),
                    100. * batch_idx * args.num_processes / len(data_loader),
                    args.num_processes,
                    len(data_loader.dataset),
                    _loss2))

        checks = []
        if any(checks):
            print(rank, "Gradient distrib threshold met. Stopping stage!")
            for i in range(args.num_processes):
                status_q.put((i,{"m":True}))
            return True

        # N processes each see 1/N of the dataset per epoch
        if batch_idx >= int(data_loader.__len__() / args.num_processes):
            return False

    return False
