import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import datetime
import os
from net.net import MyNet
import math

name="6pattern"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
load_modelpath='./checkpoints/checkpoints_'+name+"/bestmodel.pth"
import h5py

frame_num=20
size=[128,128,128]
patternnum=6
cameranum=3
dataname="./data/"
donorm=1        # apply normalized to the measurement
donorm2=1       # apply normal factor back to the predicted volume
sizedata=128

def normalization(x):
    if(x.max()==x.min()):
        return x/(x.min()+1e-7)
    return (x-x.min())/(x.max()-x.min()+1e-7)

# normalization function for light pattern
def pattern_normalization(x):
    x=F.sigmoid(x)
    x=x/torch.linalg.norm(x)
    return x

def reademd(filename):
    with h5py.File(filename, 'r') as f:
        D=f["data"]
        T = D['tomography']
        data=T['data']
        return data[:]

def read_data_predict(filepath):
    path_list=os.listdir(filepath)
    train_num=180
    valid_num=20
    dataset_valid=np.zeros((valid_num*frame_num,sizedata,sizedata,sizedata))
    for i in range(train_num,train_num+valid_num):
        for j in range(frame_num):
            filename=filepath+str(i-train_num)+"/"+f"density_{str(j).zfill(4)}.emd"
            if os.path.exists(filename):
                volume=reademd(filename)
            else:
                volume=np.load(f"./{filepath}/{path_list[i]}/density_{str(j).zfill(4)}.npz")["data"]
            volume=volume.reshape(128,128,128)
            volume=np.asarray(volume)
            dataset_valid[(i-train_num)*frame_num+j]=volume
    return dataset_valid

def save_data(filename,density):
    f=h5py.File(filename,"w")
    f.create_dataset('data/tomography/data',data=density)

    f.close()

def predict_net(encoder,decoder,device):
    batch_size=1
    filename=dataname
    valid_datasets,scaler = read_data_predict(filename)
    n_valid=valid_datasets.shape[0]
    print("data loaded")

    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    valid_iter = DataLoader(valid_datasets, shuffle=False, drop_last=True, **loader_args)
    id=0
    loss_avg=0
    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        cnt=0
        for volume in valid_iter:
            volume=volume.to(device=device, dtype=torch.float32)
            measurement=encoder(volume).reshape((batch_size,patternnum*cameranum,size[0],size[1]))
            if donorm:
                # record the max value to rescale predicted volume density back to real scale
                recmax=measurement.max()
                measurement=measurement/recmax
            v1,v2,v3,volume_pred=decoder(measurement,encoder.light_pattern)
            if donorm2:
                volume_pred=volume_pred*recmax
            if(volume.sum()==0):
                volume_pred=volume

            loss=torch.sqrt((torch.square(volume-volume_pred)).mean())
            loss_avg+=loss.item()
            print(id,loss.item())
            savepath="./our_result/"
            if not os.path.exists(savepath+"gt/"):
                os.mkdir(savepath+"gt/")
            if not os.path.exists(savepath+"measurement/"):
                os.mkdir(savepath+"measurement/")
            save_data(savepath+str(id).zfill(4)+".emd",volume_pred[0].cpu())
            save_data(savepath+"gt/"+str(id).zfill(4)+".emd",volume[0].cpu())
            path1=savepath+"measurement/"+str(id)+"/"
            if not os.path.exists(path1):
                os.mkdir(path1)
            imgs=np.asarray(measurement[0].to(device=torch.device('cpu')))
            for i in range(patternnum*cameranum):
                cv2.imwrite(path1+str(i)+".bmp",255*normalization(imgs[i]))
            id=id+1
            cnt+=batch_size
    print("avg:",loss_avg/n_valid*batch_size)
    return 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = MyNet(pattern_num=patternnum,pattern_size=(size[1],size[2]),camera_num=cameranum,density_shape=size,cuda=device)
net.to(device=device)
ckpt=torch.load(load_modelpath, map_location=device)
net.load_state_dict(ckpt["net"])

print("model loaded")

start_time = datetime.datetime.now()
predict_net(encoder=net.encoder,decoder=net.decoder,device=device)
end_time = datetime.datetime.now()
print("predict time: ", end_time-start_time)
