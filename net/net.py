import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter
from net.unet_part import *
import scipy

def pattern_normalization(x):
    x=F.sigmoid(x)
    x=x/torch.linalg.norm(x)
    return x

class MyEncoder(nn.Module):
    def __init__(self,pattern_num,pattern_size,camera_num,density_shape,cuda):
        super(MyEncoder, self).__init__()
        self.device=cuda
        self.density_shape=density_shape
        self.light_pattern=Parameter(torch.randn((pattern_num,pattern_size[1]),requires_grad=True).cuda(cuda))

        temp1=torch.load("./ray_trace_info/128cam"+str(0)+"_20sample.tsr")
        temp2=torch.load("./ray_trace_info/128cam"+str(1)+"_20sample.tsr")
        temp3=torch.load("./ray_trace_info/128cam"+str(2)+"_20sample.tsr")
        temp1=temp1/temp1._values().max()
        temp2=temp2/temp2._values().max()
        temp3=temp3/temp3._values().max()

        self.ray_trace_info=[temp1.to(device=cuda),temp2.to(device=cuda),temp3.to(device=cuda)]

        temp=torch.load("./ray_trace_info/light_20_sample.tsr")
        temp=temp/temp.max()
        self.light_info=temp.to(device=cuda)

        self.camera_num=camera_num
    
    def forward(self,density_volume):
        batchsize=density_volume.shape[0]
        patternnum=self.light_pattern.shape[0]
        cameranum=self.camera_num
        density_shape=self.density_shape
        # volume: 20 x 128 x 128 x 128 (with batchsize)
        # light: 6 x 128 (with patternnumN)
        # camera: 3 x 128 x 128 x 128 (cameranumM x (x)H x (x) x (z))
        
        # ray_trace_info: cameranum x hw x xyz
        x=torch.zeros(batchsize,patternnum,cameranum,density_shape[0],density_shape[2]).to(device=self.device)
        light=pattern_normalization(self.light_pattern)
        light=torch.einsum("HXZ,NH->NXZ",self.light_info,light)
        # apply light intensity to density volume
        density_encode=torch.einsum("BXYZ,NXZ->BNXYZ",density_volume,light)
        density_encode=density_encode.reshape(batchsize*patternnum,density_shape[0]*density_shape[1]*density_shape[2]).permute(1,0)

        # for each camera, apply ray_trace_info to render camera image
        for i in range(cameranum):
            temp=torch.matmul(self.ray_trace_info[i],density_encode).permute(1,0).view(batchsize,patternnum,density_shape[0],density_shape[1])
            x[:,:,i,:,:]=temp

        # apply noise modeled from real device
        param={'K': 1.790987517302874, 'loc': -0.08208043639065216, 'scale': 0.05720079131427573}
        noise=scipy.stats.exponnorm.rvs(param["K"], loc=param["loc"], scale=param["scale"], size=x.shape)
        noise=torch.tensor(noise)
        x=x+x*noise.to(device=self.device, dtype=torch.float32)

        return x

def fc_block(in_channels: int, out_channels: int, activation: nn.Module = nn.LeakyReLU(), drop_out: float = 0,bn=0):
    layers = [nn.Linear(in_channels, out_channels)]
    if bn==1:
        layers.append(nn.BatchNorm1d(out_channels))
    if activation is not None:
        layers.append(activation)
    if drop_out is not None:
        layers.append(nn.Dropout(drop_out))
    return nn.Sequential(*layers)

class mlp_block(nn.Module):
    def __init__(self, innum,outnum,dropout,featurelist):
        super(mlp_block,self).__init__()
        bn=1
        self.decoder1=fc_block(innum,featurelist[1],bn=bn)
        self.decoder2=fc_block(featurelist[1],featurelist[2], drop_out=dropout,bn=bn)
        self.decoder3=fc_block(featurelist[2],featurelist[3], drop_out=dropout,bn=bn)
        self.decoder5=fc_block(featurelist[3],outnum,bn=bn)
    
    def forward(self,x,retfeature):
        featurenum=x.shape[-1]
        x0=x.view(-1,featurenum)
        x1=self.decoder1(x0)
        x2=self.decoder2(x1)
        x3=self.decoder3(x2)
        x4=self.decoder5(x3)
        if retfeature:
            return x4,x3

        return x4
    
class unet3d(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size=3):
        super(unet3d, self).__init__()
        self.firstchannel=16
        self.inc = DoubleConv3d(in_channel, self.firstchannel)
        self.down1 = Down3d(self.firstchannel, self.firstchannel*2)
        self.down2 = Down3d(self.firstchannel*2, self.firstchannel*4)
        self.down3 = Down3d(self.firstchannel*4, self.firstchannel*8)
        self.up2 = Up3d(self.firstchannel*8, self.firstchannel*4)
        self.up3 = Up3d(self.firstchannel*4, self.firstchannel*2)
        self.up4 = Up3d(self.firstchannel*2, self.firstchannel)
        self.outc = OutConv3d(self.firstchannel, out_channel)

    def forward(self, x):
        xx1 = self.inc(x)
        xx2 = self.down1(xx1)
        xx3 = self.down2(xx2)
        xx4 = self.down3(xx3)
        x = self.up2(xx4, xx3)
        x = self.up3(x, xx2)
        x = self.up4(x, xx1)
        return self.outc(x)
    
class MyDecoder(nn.Module):
    def __init__(self,cuda,density_shape,camera_num,in_channel=10,out_channel=32):
        super(MyDecoder, self ).__init__()
        self.device=cuda
        self.patternnum=int(in_channel/camera_num)
        self.cameranum=camera_num

        self.ray_trace_info=self.init_mapping(density_shape)
        self.pattern_coo=self.init_patterncoo(density_shape)
        self.light_info=self.init_light(density_shape)
        
        self.cond_num1=density_shape[2]*self.patternnum
        self.cond_num3=density_shape[2]
        self.cond3=None

        feature1=[self.patternnum+self.cond_num1,384,192,192,density_shape[2]]
        self.decoder1=mlp_block(feature1[0],feature1[-1],0,feature1)
        self.decoder3=unet3d(3,1)
        self.out_channel=out_channel
        self.density_shape=density_shape

    def init_patterncoo(self,density_shape):
        temp_sparse2=torch.load("./ray_trace_info/patterncoo_20sample_cam2_x.tsr",map_location=self.device)
        temp_sparse1=torch.load("./ray_trace_info/patterncoo_20sample_cam1_x.tsr",map_location=self.device)
        temp_sparse0=torch.load("./ray_trace_info/patterncoo_20sample_cam0_x.tsr")
        temp_sparse2=temp_sparse2/temp_sparse2._values().max()
        temp_sparse1=temp_sparse1/temp_sparse1._values().max()
        temp_sparse0=temp_sparse0/temp_sparse0._values().max()
        return [temp_sparse0.to(device=self.device),temp_sparse1.to(device=self.device),temp_sparse2.to(device=self.device)]
    
    def init_mapping(self,density_shape):
        temp=torch.load("./ray_trace_info/128cam"+str(0)+"_20sample.tsr")
        temp1=torch.load("./ray_trace_info/128cam"+str(1)+"_20sample.tsr")
        temp2=torch.load("./ray_trace_info/128cam"+str(2)+"_20sample.tsr")
        temp=temp/temp._values().max()
        temp1=temp1/temp1._values().max()
        temp2=temp2/temp2._values().max()
        return [temp.to(device=self.device),temp1.to(device=self.device),temp2.to(device=self.device)]

    def init_light(self,density_shape):
        temp=torch.load("./ray_trace_info/light_20_sample.tsr")
        temp=temp/temp.max()
        return temp.to(device=self.device)
    
    def cal_cond_idealA(self,id,lightpattern,density_shape,batchsize):
        patternnum=lightpattern.shape[0]
        ray_trace_info=self.pattern_coo[id]
        # ray_trace_info: sparse, hw x xyz
        light=pattern_normalization(lightpattern)
        light=torch.einsum("HXZ,NH->NXZ",self.light_info,light)
        light=light.view(patternnum,density_shape[0],1,density_shape[2])
        light=light.expand(patternnum,density_shape[0],density_shape[1],density_shape[2])
        light=light.reshape(patternnum,density_shape[0]*density_shape[1]*density_shape[2])
        light=light.permute(1,0)
        # light: dense, xyz x patternum
        realpattern=torch.matmul(ray_trace_info,light)
        realpattern=realpattern.view(1,density_shape[0],density_shape[1],density_shape[2]*patternnum)
        ret=realpattern.expand(batchsize,density_shape[0],density_shape[1],density_shape[2]*patternnum)
        # hwz x patternum
        return ret

    def cal_cond_zonehot(self,density_shape,batchsize):
        if self.cond3==None:
            z_cond=torch.eye((density_shape[2]),dtype=torch.float32,requires_grad=True).cuda(self.device)
            z_cond=z_cond.view(1,density_shape[2],density_shape[2])
            self.cond3=z_cond.to(device=self.device)
        
        ret=self.cond3.expand(batchsize,density_shape[0],density_shape[1],density_shape[2],density_shape[2])

        return ret

    def forward(self,measurement,lightpattern):
        batchsize=measurement.shape[0]
        density_shape=self.density_shape
        patternnum=self.patternnum
        cameranum=self.cameranum
        x=measurement.view(batchsize,patternnum,cameranum,density_shape[0],density_shape[1])
        
        x1=x[:,:,0,:,:]
        x1=x1.permute(0,2,3,1)
        # x1: B x X x Y x patternnum
        x1_cond=self.cal_cond_idealA(0,lightpattern,density_shape,batchsize)
        x1_input=torch.cat([x1,x1_cond],dim=3)

        x2=x[:,:,1,:,:]
        x2=x2.permute(0,2,3,1)
        # x2: B x X x Y x patternnum
        x2_cond=self.cal_cond_idealA(1,lightpattern,density_shape,batchsize)
        x2_input=torch.cat([x2,x2_cond],dim=3)

        x3=x[:,:,2,:,:]
        x3=x3.permute(0,2,3,1)
        # x3: B x X x Y x patternnum
        x3_cond=self.cal_cond_idealA(2,lightpattern,density_shape,batchsize)
        x3_input=torch.cat([x3,x3_cond],dim=3)

        # send to decoder
        input=torch.cat([x1_input,x2_input,x3_input],dim=0)
        y123=self.decoder1(input,0)
        y123=y123.view(batchsize*3,density_shape[0],density_shape[1],density_shape[2])
        y1_beforerot=y123[0:batchsize,:,:,:]
        y2_beforerot=y123[batchsize:batchsize*2,:,:,:]  
        y3_beforerot=y123[batchsize*2:batchsize*3,:,:,:]

        # resample result from cam3 to volume
        mapping3to3=self.pattern_coo[2].permute(1,0)
        y3_beforerot=y3_beforerot.permute(1,2,3,0)
        y3_beforerot=y3_beforerot.view(density_shape[0]*density_shape[1]*density_shape[2],batchsize)
        y3=torch.matmul(mapping3to3,y3_beforerot).view(*density_shape,batchsize)
        y3=y3.permute(3,0,1,2)

        # resample result from cam2 to volume
        mapping3to3=self.pattern_coo[1].permute(1,0)
        y2_beforerot=y2_beforerot.permute(1,2,3,0)
        y2_beforerot=y2_beforerot.view(density_shape[0]*density_shape[1]*density_shape[2],batchsize)
        y2=torch.matmul(mapping3to3,y2_beforerot).view(*density_shape,batchsize)
        y2=y2.permute(3,0,1,2)

        # resample result from cam1 to volume
        mapping3to3=self.pattern_coo[0].permute(1,0)
        y1_beforerot=y1_beforerot.permute(1,2,3,0)
        y1_beforerot=y1_beforerot.view(density_shape[0]*density_shape[1]*density_shape[2],batchsize)
        y1=torch.matmul(mapping3to3,y1_beforerot).view(*density_shape,batchsize)
        y1=y1.permute(3,0,1,2)

        # send to aggregation module
        x4=torch.cat([y1.reshape(batchsize,1,*density_shape),y2.reshape(batchsize,1,*density_shape),y3.reshape(batchsize,1,*density_shape)],dim=1)
        y4=self.decoder3(x4)

        output1=y1.view(batchsize,density_shape[0],density_shape[1],density_shape[2])
        output2=y2.view(batchsize,density_shape[0],density_shape[1],density_shape[2])
        output3=y3.view(batchsize,density_shape[0],density_shape[1],density_shape[2])
        output4=y4.view(batchsize,density_shape[0],density_shape[1],density_shape[2])

        return output1,output2,output3,output4

class MyNet(nn.Module):
    def __init__(self,pattern_num,pattern_size,camera_num,density_shape,cuda):
        super(MyNet, self ).__init__()
        size=density_shape
        self.encoder=MyEncoder(pattern_num=pattern_num,pattern_size=(size[1],size[2]),
                               camera_num=camera_num,density_shape=size,cuda=cuda)
        self.decoder=MyDecoder(cuda=cuda,density_shape=size,camera_num=camera_num,
                               in_channel=pattern_num*camera_num,out_channel=size[2])

    def forward(self,volume,donorm):
        measurement=self.encoder(volume)
        # if normalized, in real capture the measurement should be normalized as well
        if donorm:
            recmax=measurement.max()+1e-7
            measurement=measurement/recmax
        v1,v2,v3,volume_pred=self.decoder(measurement,self.encoder.light_pattern)
        if donorm:
            volume_pred=volume_pred*recmax
            v1=v1*recmax
            v2=v2*recmax
            v3=v3*recmax
        
        return v1,v2,v3,volume_pred