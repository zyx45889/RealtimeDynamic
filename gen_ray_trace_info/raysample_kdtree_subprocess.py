import subprocess
import numpy as np
import datetime

cam_idx_gl=[-1,0,1,2]
for cam in cam_idx_gl:
    # cam=-1
    if cam!=-1:
        batch1=27
        batch2=36
        basew=40
        baseh=40
    else:
        batch1=20
        batch2=16
        basew=32
        baseh=32

    cpunum=30

    cmd = 'python ./gen_ray_trace_info/raysample_kdtree.py --w1 0 --w2 0 --h1 0 --h2 0 --cam '+str(cam)+" --bt 1"
    p = subprocess.Popen(cmd, shell=True)
    while(1):
        if(p.poll() is not None):
            break

    cpustatues=np.zeros(cpunum)
    plist=[]
    p_cpuid=[]
    start_time_total = datetime.datetime.now()

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
                    w1=i*basew
                    w2=w1+basew
                    h1=j*baseh
                    h2=h1+baseh
                    cmd='python ./gen_ray_trace_info/raysample_kdtree.py --w1 '+str(w1)+' --w2 '+str(w2)+' --h1 '+str(h1)+' --h2 '+str(h2)+" --cam "+str(cam)
                    print("start cmd:",cmd)
                    p = subprocess.Popen(cmd, shell=True)
                    plist.append(p)
                    p_cpuid.append(k)
                    cpustatues[k]=1
                    print("start on:",k)
                    break

    while(cpustatues.sum()!=0):
        if cpustatues.sum()<cpunum:
                    break
        for k in range(len(plist)):
            if(plist[k].poll() is not None and p_cpuid[k] !=-1):
                cpustatues[p_cpuid[k]]=0
                p_cpuid[k]=-1

    end_time = datetime.datetime.now()
    print("total time:",end_time-start_time_total)