import numpy as np
import torch
import cv2
import datetime
import argparse
# rescale the ray trace information in each block into our desired size
# fetch the information of our interested area
# result: ray trace info. in interested 128x128 image area and 128x128x128 volume area

parser = argparse.ArgumentParser(description='raysample rescale')
parser.add_argument('--b1','-b1',type=int, default = 0,required=True,help="batch id 1")
parser.add_argument('--b2','-b2',type=int, default = 1080,required=True,help="batch id 2")
parser.add_argument('--cam','-cam',type=int, default=0,required=True,help='camera id')
 
args = parser.parse_args()
batch1=args.b1
batch2=args.b2
camid=args.cam

densityshape=128

if camid==-1:
    basew=32
    baseh=32
else:
    basew=40
    baseh=40

hwlist=[]
xyzlist=[]
valuelist=[]

# cam0
if camid==0:
    hmin=390
    hmax=hmin+640
    wmin=230
    wmax=wmin+640
    sm_hmin=0
    sm_hmax=densityshape
    sm_wmin=0
    sm_wmax=densityshape

# cam1
if camid==1:
    hmin=321
    hmax=hmin+768
    wmin=205
    wmax=wmin+640
    sm_hmin=0
    sm_hmax=densityshape
    sm_wmin=0
    sm_wmax=densityshape

# cam2
if camid==2:
    hmin=456
    hmax=hmin+768
    wmin=190
    wmax=wmin+640
    sm_hmin=0
    sm_hmax=densityshape
    sm_wmin=0
    sm_wmax=densityshape

# proj
if camid==-1:
    hmin=0
    hmax=512
    wmin=0
    wmax=640
    sm_hmin=0
    sm_hmax=densityshape
    sm_wmin=0
    sm_wmax=densityshape

hwshape=[wmax-wmin,hmax-hmin]
sm_hwshape=[sm_wmax-sm_wmin,sm_hmax-sm_hmin]
scale_hwshape=[sm_hwshape[0]/hwshape[0],sm_hwshape[1]/hwshape[1]]

def rescale(idx1,idx2):
    idx1=torch.floor((idx1-wmin)*scale_hwshape[0])
    idx2=torch.floor((idx2-hmin)*scale_hwshape[1])
    return idx1*sm_hwshape[1]+idx2

def rescalexyz(idx1,idx2,idx3):
    if densityshape==128:
        return (idx1*128+idx2)*128+idx3
    shape=densityshape
    scaleshape=shape/128
    idx1=torch.floor((idx1)*scaleshape)
    idx2=torch.floor((idx2)*scaleshape)
    idx3=torch.floor((idx3)*scaleshape)
    return (idx1*shape+idx2)*shape+idx3

w1=batch1*basew
w2=w1+basew
h1=batch2*baseh
h2=h1+baseh
temp=torch.load("./ray_trace_info/temp/w"+str(w1)+"to"+str(w2)+"_h"+str(h1)+"to"+str(h2)+"cam"+str(camid)+"_20sample.tsr")
indices=temp._indices()
values=temp._values()
if(indices.shape[1]!=0):
    hwidx=rescale(indices[0,:],indices[1,:])   #indices[0,:]*hwshape[1]+indices[1,:]
    hwmin=hwidx.min()
    hwmax=hwidx.max()
    xyzidx=rescalexyz(indices[2,:],indices[3,:],indices[4,:])
    for k in range(indices.shape[1]):
        if(indices[0,k]<wmin or indices[0,k]>=wmax or indices[1,k]<hmin or indices[1,k]>=hmax):
            continue
        hwlist.append(hwidx[k])
        xyzlist.append(xyzidx[k])
        valuelist.append(values[k])
indices = torch.tensor([hwlist,xyzlist])
values = torch.tensor(valuelist, dtype=torch.float32)
ray_trace_info=torch.sparse_coo_tensor(indices=indices, values=values, size=[sm_hwshape[0]*sm_hwshape[1], densityshape*densityshape*densityshape])
ray_trace_info=ray_trace_info.coalesce()
torch.save(ray_trace_info,"./ray_trace_info/temp/batch"+str(batch1)+"_"+str(batch2)+"cam"+str(camid)+"_20sample.tsr")
