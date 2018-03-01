import random
import numpy as np
import sys
import math

def lorenz(x, y, z, s=10, r=28, b=2.667):
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot

def dist(x, y):
    s = 0
    for i in range(len(x)):
        s += (x[i]-y[i])**2
    return s**0.5

def gen_lorenz_series(x0, y0, z0, dstT, totalStep, genV):
    dt = 0.01
    stepCnt = 1

    # Setting initial values
    pt = [[x0, y0, z0]]
    xd, yd, zd = lorenz(x0, y0, z0)
    ptv = [[xd, yd, zd]]

    pGen = [x0, y0, z0]
    dst = 0
    # Stepping through "time".
    while(stepCnt < totalStep):
#        print(dst)
        dot = [0,0,0]
        dot[0], dot[1], dot[2] = lorenz(pGen[0], pGen[1], pGen[2])
#        print(dot)
        for i in range(3):
            pGen[i] += dot[i]*dt
        dst += dist(pGen, [pGen[0]-dot[0]*dt, pGen[1]-dot[1]*dt, pGen[2]-dot[2]*dt])
        if(dst >= dstT):
#            print(dst, dstT)
            pt.append([pGen[0], pGen[1], pGen[2]])
            dst = 0
            stepCnt += 1
            ptv.append(np.dot(dt,dot))
#    pt = np.add(-1*np.min(pt), pt)
#    pt = np.dot(1/np.max(pt), pt)
    #save the sequence for training
    xss = []; yss = []; zss = []
    xv = []; yv = []; zv = []
    for i in range(len(pt)):
        xss.append(pt[i][0])
        yss.append(pt[i][1])
        zss.append(pt[i][2])
        xv.append(ptv[i][0])
        yv.append(ptv[i][1])
        zv.append(ptv[i][2])
    if(genV):
        lorenz_series = np.transpose(np.vstack((xss,yss,zss,xv,yv,zv)))
    else:
        lorenz_series = np.transpose(np.vstack((xss, yss, zss)))
    return lorenz_series

#random distribution around cent
def rrange(cent):
    return random.random()/10-0.05 + cent

velo = bool(int(sys.argv[2]))
print(velo)
res = []
for iter in range(1):#000):
    if(iter%50 == 0):
        print(iter)
    sx = rrange(0)
    sy = rrange(0)
    sz = rrange(0)
    ptb = []
    xD = []
    yD = []
    zD = []
    for i in range(int(sys.argv[1])):
        xD.append([])
        yD.append([])
        zD.append([])
    for i in range(1000):
        ptb.append(gen_lorenz_series(rrange(sx), rrange(sy), rrange(sz), 10, int(sys.argv[1]), False))
    ptb.append(gen_lorenz_series(sx, sy, sz, 10, int(sys.argv[1]), True))
#    print(np.shape(ptb[0]))
#    print(np.min(ptb))
    amin = math.inf
    for i in ptb:
        amin = np.min(i) if np.min(i) < amin else amin
    for i in range(len(ptb)):
        ptb[i] = np.add(-1*amin, ptb[i])
    mx = 0
    for i in ptb:
        mx = np.max(i) if np.max(i) > mx else mx
    for i in range(len(ptb)):
        ptb[i] = np.dot(1/mx, ptb[i])
    for i in range(1000):
        for it in range(len(ptb[i])):
            xD[it].append(ptb[i][it][0])
            yD[it].append(ptb[i][it][1])
            zD[it].append(ptb[i][it][2])
    tA = []
    for i in range(int(sys.argv[1])):
        tA.append(np.ndarray.tolist(ptb[-1][i]) + [np.var(xD[i]), np.var(yD[i]), np.var(zD[i])])
#        print(xD[i][0], yD[i][0], zD[i][0])
        print(np.var(xD[i]), np.var(yD[i]), np.var(zD[i]))
    print("actual")
    for i in range(int(sys.argv[1])):
        print(ptb[-1][i])
    res.append(tA)

np.save("data/lorenz.npy", np.array(res))
