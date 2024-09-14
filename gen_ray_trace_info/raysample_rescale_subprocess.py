import subprocess
import numpy as np
import datetime
import torch
import cv2
# given cpu num, do rescale in small blocks in dynamic parallelism
# then merge information in blocks into one tensor

cam_idx_gl=[-1,0,1,2]
for camid in cam_idx_gl:
    # camid=-1
    if camid!=-1:
        batch1=27
        batch2=36
    else:
        batch1=20
        batch2=16

    densityshape=128

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

    sm_hwshape=[sm_wmax-sm_wmin,sm_hmax-sm_hmin]

    cpunum=30

    cpustatues=np.zeros(cpunum)
    plist=[]
    p_cpuid=[]
    start_time_total = datetime.datetime.now()

    def normalization(x):
        if x.max()==0:
            return x
        return x/x.max()

    for i in range(batch1):
        for j in range(batch2):
            while(1):
                if cpustatues.sum()<cpunum:
                    break
                for k in range(len(plist)):
                    if(plist[k].poll() is not None and p_cpuid[k] !=-1): 
                        cpustatues[p_cpuid[k]]=0
                        p_cpuid[k]=-1
            for k in range(cpunum):
                if(cpustatues[k]==0):
                    cmd='python ./gen_ray_trace_info/raysample_rescale.py --b1 '+str(i)+' --b2 '+str(j)+" --cam "+str(camid)
                    print("start cmd:",cmd)
                    p = subprocess.Popen(cmd, shell=True)
                    plist.append(p)
                    p_cpuid.append(k)
                    cpustatues[k]=1
                    print("start on:",k)
                    break

    # wait until all the rescale is done
    while(cpustatues.sum()!=0):
        if cpustatues.sum()<cpunum:
                    break
        for k in range(len(plist)):
            if(plist[k].poll() is not None and p_cpuid[k] !=-1):
                cpustatues[p_cpuid[k]]=0
                p_cpuid[k]=-1

    print("add start")
    hwlist=[]
    xyzlist=[]
    valuelist=[]
    indices = torch.tensor([hwlist,xyzlist])
    values = torch.tensor(valuelist, dtype=torch.float32)
    ray_trace_info=torch.sparse_coo_tensor(indices=indices, values=values, size=[sm_hwshape[0]*sm_hwshape[1], densityshape*densityshape*densityshape])
    for i in range(batch1):
        for j in range(batch2):
            print(i,j)
            temp=torch.load("./ray_trace_info/temp/batch"+str(i)+"_"+str(j)+"cam"+str(camid)+"_20sample.tsr")
            ray_trace_info+=temp

    torch.save(ray_trace_info,"./ray_trace_info/0913/128cam"+str(camid)+"_20sample.tsr")
    end_time = datetime.datetime.now()
    print("total time:",end_time-start_time_total)
    # img=np.asarray(ray_trace_info.sum(axis=1).to_dense()).reshape(sm_hwshape[0],sm_hwshape[1])
    # cv2.imwrite("./test128.bmp",255*normalization(img))
