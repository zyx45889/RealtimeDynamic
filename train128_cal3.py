import torch
import numpy as np
import wandb
from tqdm import tqdm
import time
import datetime
import os
import h5py
from pathlib import Path
from net.net import MyNet

name="6pattern_previousdata"
modelpath = './checkpoints/checkpoints_'+name+'/'
if not os.path.exists(modelpath):
    os.mkdir(modelpath)
dir_checkpoint = Path(modelpath)

frame_num=20            # how many frames are generated in data for each subfolder
size=[128,128,128]      # volume size
patternnum=6            # pattern num
camera_num=3            # camera num
dowandb=1               # use wandb to visualize training or not
dosavecheckpoint=1      # save ckpt or not
doloadalldata=1         # load all data or part of them to debug
miu1=0.5                # training loss param
miu2=0.5
miu3=0.5
donorm=0                # use normalized measurement or not
datapath="../laserbeam/data128"      # data path

def reademd(filename):
    with h5py.File(filename, 'r') as f:
        D=f["data"]
        T = D['tomography']
        data=T['data']
        return data[:]

def read_data(filepath):
    path_list=os.listdir(filepath)
    num_inited=200
    if doloadalldata:
        train_num=int(0.8*num_inited)
        valid_num=num_inited-train_num
    else :
        train_num=10
        valid_num=10
    dataset_train=np.zeros((train_num*frame_num,size[0],size[1],size[2]))
    for i in range(train_num):
        for j in range(frame_num):
            volume=reademd(f"{filepath}/{path_list[i]}/density_{str(j).zfill(4)}.emd")
            dataset_train[i*frame_num+j]=volume
    dataset_valid=np.zeros((valid_num*frame_num,size[0],size[1],size[2]))
    for i in range(train_num,train_num+valid_num):
        # name_list=os.listdir("")
        for j in range(frame_num):
            volume=reademd(f"{filepath}/{path_list[i]}/density_{str(j).zfill(4)}.emd")
            dataset_valid[(i-train_num)*frame_num+j]=volume
    return dataset_train,dataset_valid

def normalization(x):
    if(x.max()==x.min()):
        return x/(x.min()+1e-7)
    return (x-x.min())/(x.max()-x.min()+1e-7)

def save_data(filename,density):
    f=h5py.File(filename,"w")
    f.create_dataset('data/tomography/data',data=density)

    f.close()

def train_net(net,device,save_checkpoint,local_rank):
    epochs = 1000
    batch_size =5
    learning_rate = 0.004
    print(f"aug batch size: {batch_size}")
    
    if dowandb and local_rank==0:
        experiment = wandb.init(project='laserbeam',  
                                name="prototype-"+name+"-"+time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())),
                                resume='allow', anonymous='must', 
                                settings=wandb.Settings(start_method="fork"))
        experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                    save_checkpoint=save_checkpoint,gpu_num=os.environ["CUDA_VISIBLE_DEVICES"] ))

    train_datasets,valid_datasets = read_data(datapath)
    n_train=train_datasets.shape[0]
    n_valid=valid_datasets.shape[0]
    print("data loaded")

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_datasets)
    train_iter = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size  ,sampler=train_sampler)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_datasets)
    valid_iter = torch.utils.data.DataLoader(valid_datasets, batch_size=batch_size, shuffle=False,drop_last=True, sampler=valid_sampler)

    optimizer = torch.optim.SGD(
        [
            {"params":net.module.encoder.parameters(),"lr":10*learning_rate},
            {"params":net.module.decoder.parameters()},
        ], 
        lr=learning_rate, 
        weight_decay=1e-8, 
        momentum=0.98
    )

    global_step=0
    min_valid_loss=1e7
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        train_sampler.set_epoch(epoch)
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for volume in train_iter:
                volume=volume.to(device=device, dtype=torch.float32)
                v1,v2,v3,volume_pred=net(volume,donorm)
                loss1=torch.sqrt((torch.square(volume-volume_pred)).mean())
                loss2=torch.sqrt((torch.square(volume-v1)).mean())
                loss3=torch.sqrt((torch.square(volume-v2)).mean())
                loss4=torch.sqrt((torch.square(volume-v3)).mean())
                loss=loss4*miu3+loss3*miu2+loss2*miu1+loss1

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                pbar.update(volume.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                if dowandb and local_rank==0:
                    experiment.log({
                        'train loss1': loss1.item(),
                        'train loss2': loss2.item(),
                        'train loss3': loss3.item(),
                        'train loss4': loss4.item(),
                        'step': global_step,
                        'epoch': epoch
                    })
                pbar.set_postfix(**{'loss (batch)': loss.item()})
            
            # valid
            avg_loss1=0
            avg_loss2=0
            avg_loss3=0
            avg_loss4=0
            cntloss=0
            with torch.no_grad():
                net.eval()
                for volume in valid_iter:
                    volume=volume.to(device=device, dtype=torch.float32)
                    v1,v2,v3,volume_pred=net(volume,donorm)
                    loss1=torch.sqrt((torch.square(volume-volume_pred)).mean())
                    loss2=torch.sqrt((torch.square(volume-v1)).mean())
                    loss3=torch.sqrt((torch.square(volume-v2)).mean())
                    loss4=torch.sqrt((torch.square(volume-v3)).mean())
                    avg_loss1+=loss1.item()
                    avg_loss2+=loss2.item()
                    avg_loss3+=loss3.item()
                    avg_loss4+=loss4.item()
                    cntloss=cntloss+1
                
                avg_loss1/=cntloss
                avg_loss2/=cntloss
                avg_loss3/=cntloss
                avg_loss4/=cntloss
                if avg_loss1 < min_valid_loss:
                    min_valid_loss = avg_loss1
                    ckpt={"net":net.module.state_dict()}
                    torch.save(ckpt, str(dir_checkpoint / 'bestmodel.pth'.format(epoch)))

                if dowandb and local_rank==0:
                    experiment.log({
                        'learning rate': optimizer.param_groups[0]['lr'],
                        'validation loss1': avg_loss1,
                        'validation loss2': avg_loss2,
                        'validation loss3': avg_loss3,
                        'validation loss4': avg_loss4,
                        'step': global_step,
                        'epoch': epoch,
                    })

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            ckpt={"net":net.module.state_dict()}
            torch.save(ckpt, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
    return 

local_rank=int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
torch.distributed.init_process_group(
    backend='nccl',
    init_method='env://'
)
device = torch.device(f'cuda:{local_rank}')

net = MyNet(pattern_num=patternnum,pattern_size=(size[1],size[2]),camera_num=camera_num,density_shape=size,cuda=device)
net=net.cuda()
net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net).to(device)
net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank],find_unused_parameters=True)
print("model loaded")

start_time = datetime.datetime.now()
train_net(net=net,device=device,save_checkpoint=dosavecheckpoint,local_rank=local_rank)
end_time = datetime.datetime.now()
print("train time: ", end_time-start_time) 