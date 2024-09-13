import numpy as np
import cv2 as cv
from annoy import AnnoyIndex
import argparse
import datetime
import torch
import h5py

# dump ply points for debug visualization
def dump_points(
    points: np.array,  # [N, 3]
    dump_fp: str,
    colors: np.array = None,
    normals: np.array = None,
):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(dump_fp, pcd)

def cal_index(x,step1,step2,step3):
    x1=x%step3
    x2=int(x/step3)
    x3=int(x2/step2)
    x2=x2%step2
    return [x3,x2,x1]

def save_data(filename,density):
    f=h5py.File(filename,"w")
    f.create_dataset('data/tomography/data',data=density)

    f.close()

def cal_thres(origin,x,y):
    dir1=np.asarray([origin[0]-x,origin[1]+y,0])
    dir2=np.asarray([origin[0]-x,origin[1],0])
    return (dir1*dir2).sum()/(np.linalg.norm(dir1)*np.linalg.norm(dir2))

def get_camera_relative(file_path):
    calib_results = cv.FileStorage(file_path, cv.FileStorage_READ)
    rvec = calib_results.getNode('E_rvec').mat()
    tvec = calib_results.getNode('E_tvec').mat()
    R_mat = cv.Rodrigues(rvec)[0]
    E = np.hstack((R_mat, tvec.reshape(3,1)))
    return np.vstack((E, np.array([[0.,0.,0.,1.]])))

def cal_RT(E_mat):
    R_mat = E_mat[:3,:3]
    tvec = E_mat[:3,3]
    rvec = cv.Rodrigues(R_mat)[0]
    return rvec,tvec

def cal_E(rvec,tvec):
    R_mat = cv.Rodrigues(rvec)[0]
    E = np.hstack((R_mat, tvec.reshape(3,1)))
    return np.vstack((E, np.array([[0.,0.,0.,1.]])))

def cal_extrinsic_avg(E_mat):
    rvec_avg=np.zeros([3,1],dtype=np.float64)
    tvec_avg=np.zeros([1,3],dtype=np.float64)
    for i in range(5):
        rvec,tvec=cal_RT(E_mat[i])
        rvec_avg+=rvec
        tvec_avg+=tvec
    return cal_E(rvec_avg/5,tvec_avg/5)

def loadE(path):
    calib_results = cv.FileStorage(path, cv.FileStorage_READ)
    E_rvec=calib_results.getNode('E_rvec').mat().mean(axis=0)
    E_tvec=calib_results.getNode('E_tvec').mat().mean(axis=0)
    return cal_E(E_rvec,E_tvec)

parser = argparse.ArgumentParser(description='raysample kdtree')
parser.add_argument('--w1','-w1',type=int, default = 0,required=True,help="w start")
parser.add_argument('--w2','-w2',type=int, default = 1080,required=True,help="w end")
parser.add_argument('--h1','-h1',type=int, default=0,required=True,help='h start')
parser.add_argument('--h2','-h2',type=int, default=1440,required=True,help='h end')
parser.add_argument('--cam','-cam',type=int, default=0,required=True,help='camera id')
parser.add_argument('--bt','-bt',type=int, default=0,required=False,help='build tree or not')
 
args = parser.parse_args()
wstart=args.w1
wend=args.w2
hstart=args.h1
hend=args.h2
camid=args.cam
dobuildtree=args.bt

# volume resolution
size=128
# real-world volume size in mm
start = 0
end = 96
sample_num = 20
step = (end - start)/size
volume_grid = np.mgrid[start:end:step, start:end:step, start:end:step].reshape(3,-1) + np.array([[-end/2],[-end/2],[-end/2]])
volume_grid = volume_grid.transpose(1,0)

proj2vol = np.eye(4,4, dtype=np.float32)
# real-world projector location with respect to volume center
proj2vol[:3,3] = -np.array([0.5,-60.5,555.5])
no_proj=2
no_cam0=2
no_cam1=2
no_cam2=0

# calibration information
path="./ray_trace_info/intrinsic_proj_cam_42000.yml"
calib_results=cv.FileStorage(path, cv.FileStorage_READ)
E_proj = calib_results.getNode('extrinsic_proj').mat()
proj_matrix=calib_results.getNode('projector_matrix').mat()
dist_proj=calib_results.getNode('dist_proj').mat()
E_cam1 = calib_results.getNode('extrinsic_cam0').mat()
camera1_matrix=calib_results.getNode('camera0_matrix').mat()
dist_cam1=calib_results.getNode('dist_cam0').mat()
E_cam0 = calib_results.getNode('extrinsic_cam1').mat()
camera0_matrix=calib_results.getNode('camera1_matrix').mat()
dist_cam0=calib_results.getNode('dist_cam1').mat()
E_cam2 = calib_results.getNode('extrinsic_cam2').mat()
camera2_matrix=calib_results.getNode('camera2_matrix').mat()
dist_cam2=calib_results.getNode('dist_cam2').mat()
E_proj=E_proj[no_proj]
world2vol = proj2vol @ E_proj

vol_2_cam0 = E_cam0[no_cam0] @ np.linalg.inv(world2vol)
cam0_2_vol=np.linalg.inv(vol_2_cam0)

vol_2_cam1 = E_cam1[no_cam1] @ np.linalg.inv(world2vol)
cam1_2_vol=np.linalg.inv(vol_2_cam1)

vol_2_cam2 = E_cam2[no_cam2] @ np.linalg.inv(world2vol)
cam2_2_vol=np.linalg.inv(vol_2_cam2)

if camid==0:
    cam2vol=cam0_2_vol
    camera_matrix=camera0_matrix
    dist_cam=dist_cam0
if camid==1:
    cam2vol=cam1_2_vol
    camera_matrix=camera1_matrix
    dist_cam=dist_cam1
if camid==2:
    cam2vol=cam2_2_vol
    camera_matrix=camera2_matrix
    dist_cam=dist_cam2
if camid==-1:  # projector
    cam2vol=proj2vol
    camera_matrix=proj_matrix
    dist_cam=dist_proj

# camera center in volume coordinate
origin_rot=np.asarray([0,0,0,1])
origin_rot=cam2vol@origin_rot
origin=origin_rot[0:3].reshape(1,3)

hv_num, _ = volume_grid.shape  # Length of item vector that will be indexed

path="./ray_trace_info/"

def buildtree(volume_grid,origin,camid):
    volume_grid_to_origin=volume_grid-origin
    ANN_tree = AnnoyIndex(3, 'angular')  # angular = dot if two vectors are both normalized
    for i in range(hv_num):
        ANN_tree.add_item(i, volume_grid_to_origin[i]/np.linalg.norm(volume_grid_to_origin[i]))
    print("tree added")

    ANN_tree.build(64) # 64 trees
    print("tree built")
    ANN_tree.save(path+'ANN_tree'+str(camid)+'.ann')

if dobuildtree:
    buildtree(volume_grid,origin,camid)

def cal_dist(x,y):
    return (x*y).sum()/np.linalg.norm(x)/np.linalg.norm(y)

ANN_tree = AnnoyIndex(3, 'angular')
ANN_tree.load(path+'ANN_tree'+str(camid)+'.ann') 

wlist=[]
hlist=[]
xlist=[]
ylist=[]
zlist=[]
valuelist=[]
threshold = cal_dist(origin-np.asarray([48,0,0]),origin-np.asarray([48,step/2,step/2]))
diag=step*0.5*np.sqrt(3)
n = size*2
search_k = 4096
start_time_total = datetime.datetime.now()
for w in range(wstart,wend):
    # w=0-1080
    for h in range(hstart,hend):
        # h=0-1440
        tempxyzlist=[]
        tempxyzid=[]
        rand_s = np.random.rand(sample_num, 2)
        delta = rand_s
        x_0, y_0 = np.array([h, w], dtype=np.float32)
        dir = np.asarray([[x_0, y_0, 1,1]]).repeat(sample_num,axis=0)
        dir[:,:2] += delta

        temp=dir[:,:2].reshape(sample_num*2).reshape(sample_num,2)
        dir[:,:2] = cv.undistortPoints(temp, camera_matrix, dist_cam).reshape(sample_num,2)
        dir_rot=(cam2vol@dir.T).T
        dir=dir_rot[:,:3]
        dir=dir-origin

        tempxyzlist=[]
        tempxyzid=[]
        for s in range(sample_num):
            index, dist = ANN_tree.get_nns_by_vector(dir[s]/np.linalg.norm(dir[s]), n, search_k, include_distances=True)
            distnp=np.asarray(dist)
            cosdist=np.sqrt(1-distnp*distnp)
            sindist=distnp
            for i in range(len(dist)):
                if(cosdist[i]>threshold):
                    idx=cal_index(index[i],size,size,size)
                    zvalue=idx[2]*step-end/2-origin[0,2]
                    weight=zvalue*sindist[i]
                    if(weight>=diag):
                        continue
                    weight=np.sqrt(diag*diag-weight*weight)
                    idx_xyz=index[i]
                    if idx_xyz in tempxyzlist:
                        idnow=tempxyzlist.index(idx_xyz)
                        valuelist[tempxyzid[idnow]]+=weight/sample_num
                    else:
                        hlist.append(h)
                        wlist.append(w)
                        xlist.append(idx[0])
                        ylist.append(idx[1])
                        zlist.append(idx[2])
                        valuelist.append(weight/sample_num)
                        tempxyzlist.append(idx_xyz)
                        tempxyzid.append(len(zlist)-1)
            

end_time = datetime.datetime.now()
indices = torch.tensor([wlist,hlist,xlist,ylist,zlist])
values = torch.tensor(valuelist, dtype=torch.float32)
if camid!=-1:
    whxyz_sparse=torch.sparse_coo_tensor(indices=indices, values=values, size=[1080,1440, 128,128,128])
else:
    whxyz_sparse=torch.sparse_coo_tensor(indices=indices, values=values, size=[640,512, 128,128,128])
whxyz_sparse=whxyz_sparse.coalesce()
torch.save(whxyz_sparse,"./ray_trace_info/temp/w"+str(wstart)+"to"+str(wend)+"_h"+str(hstart)+"to"+str(hend)+"cam"+str(camid)+"_20sample.tsr")
