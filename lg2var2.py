import random
import numpy as np
import sys

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

def gen_series(x0\ #initial x point
               , y0\ #initial y point
               , z0\ #initial z point
               , dstT\ #distance between points 
               , totalStep\ #total number of steps
               , varzero = True\ #if variance is calcuated from zeroth time step or every timestep 
               , pbC = 0.05\ #range for pertubations of variance
               , npts = 200\ # number of variance points to simulate
               , multires = False\ #if true then use dstT/variacne for multiresolution
               , series=lorenz): #series to produce

    dt = 1e-2 #finest resolution
    stepCnt = 1

    # Setting initial values
    pt = [[x0, y0, z0]]#list of points generated
    xd, yd, zd = series(x0, y0, z0)#initial deltas
    ptv = [[xd, yd, zd]]#veo
    varA = [[0,0,0]]
    pGen = [x0, y0, z0]
    dst = 0
    lastVar = 1
    tSum = 0
    # Stepping through "time".
    while(stepCnt < totalStep):
        dot = [0,0,0]
        dot[0], dot[1], dot[2] = lorenz(pGen[0], pGen[1], pGen[2])
        for i in range(3):
            pGen[i] += dot[i]*dt
        dst += dist(pGen, [pGen[0]-dot[0]*dt, pGen[1]-dot[1]*dt, pGen[2]-dot[2]*dt])
        if(dst >= 0.3*dstT/lastVar):
            tSum += dst
#            print(0.3*dstT/lastVar)
            pt.append([pGen[0], pGen[1], pGen[2]])
            #VARIANCE PREDICTION DATA
            pertub = []
            for pIt in range(1000):
                pertub.append([random.random()*2*pbC-pbC + pGen[0],random.random()*2*pbC-pbC + pGen[1],random.random()*2*pbC-pbC + pGen[2]])
            nxx = []; nxy = []; nxz = [];
            for p in pertub:
                nx, ny, nz = lorenz(p[0], p[1], p[2])
                nxx.append(nx)
                nxy.append(ny)
                nxz.append(nz)
            varx = np.var(nxx)
            vary = np.var(nxy)
            varz = np.var(nxz)
            lastVar = max((varx**2 + vary**2 + varz**2)**0.5, 0.2) #limit how long this distance can go (15 units)
            varA.append([varx, vary, varz])
#            print(np.mean(nxx), np.mean(nxy), np.mean(nxz))
#            print(dot)
            dst = 0
            stepCnt += 1
            ptv.append(np.dot(dt,dot))
#    print(varA)
    pt = np.add(-1*np.min(pt), pt)
    pt = np.dot(1/np.max(pt), pt)
    #save the sequence for training
    xss = []; yss = []; zss = []
    xv = []; yv = []; zv = []
    vx = []; vy = []; vz = []
    for i in range(len(pt)):
        xss.append(pt[i][0])
        yss.append(pt[i][1])
        zss.append(pt[i][2])
        xv.append(ptv[i][0])
        yv.append(ptv[i][1])
        zv.append(ptv[i][2])
        vx.append(varA[i][0])
        vy.append(varA[i][1])
        vz.append(varA[i][2])
    lorenz_series = np.transpose(np.vstack((xss,yss,zss,vx,vy,vz)))#xv,yv,zv,vx,vy,vz)))
    print(tSum)
    return lorenz_series

def rrange():
    return random.random()*0.4-0.2

velo = bool(int(sys.argv[2]))
#print(velo)
res = []
for iter in range(1000):
    a = gen_lorenz_series(rrange(), rrange(), rrange(), 10, int(sys.argv[1]))
    tA = []
    for i in a:
        if(velo):
            tA.append(i)
        else:
            tA.append(i[0:3])
#    res.append(np.ndarray.tolist(a))
#    print(tA)
    res.append(tA)

np.save("data/lorenz.npy", np.array(res))
