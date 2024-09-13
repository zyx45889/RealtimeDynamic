import torch

density_shape=[128,128,128]

basepath="./ray_trace_info/0913/"
temp=torch.load(basepath+"128cam"+str(-1)+"_20sample.tsr")
# light： H x XZ
temp=temp.to_dense().reshape(128,128,128,128,128).mean(axis=[0,3])
torch.save(temp,basepath+"light_20_sample.tsr")

temp=torch.load(basepath+"128cam"+str(0)+"_20sample.tsr")
ray_trace_info=temp
indices=ray_trace_info._indices()
values=ray_trace_info._values()
idx_z=indices[1,:]%density_shape[2]
idx_x=torch.floor(indices[1,:]/density_shape[1]/density_shape[2])
indices[0,:]=indices[0,:]*density_shape[2]+idx_x
shape=density_shape[0]*density_shape[1]*density_shape[2]
temp_sparse=torch.sparse_coo_tensor(indices=indices, values=values, size=[shape, shape])
torch.save(temp_sparse,basepath+"patterncoo_20sample_cam0_x.tsr")

temp=torch.load(basepath+"128cam"+str(1)+"_20sample.tsr")
ray_trace_info=temp
indices=ray_trace_info._indices()
values=ray_trace_info._values()
idx_z=indices[1,:]%density_shape[2]
idx_x=torch.floor(indices[1,:]/density_shape[1]/density_shape[2])
indices[0,:]=indices[0,:]*density_shape[2]+idx_x
shape=density_shape[0]*density_shape[1]*density_shape[2]
temp_sparse=torch.sparse_coo_tensor(indices=indices, values=values, size=[shape, shape])
torch.save(temp_sparse,basepath+"patterncoo_20sample_cam1_x.tsr")

temp=torch.load(basepath+"128cam"+str(2)+"_20sample.tsr")
ray_trace_info=temp
indices=ray_trace_info._indices()
values=ray_trace_info._values()
idx_z=indices[1,:]%density_shape[2]
idx_x=torch.floor(indices[1,:]/density_shape[1]/density_shape[2])
indices[0,:]=indices[0,:]*density_shape[2]+idx_x
shape=density_shape[0]*density_shape[1]*density_shape[2]
temp_sparse=torch.sparse_coo_tensor(indices=indices, values=values, size=[shape, shape])
torch.save(temp_sparse,basepath+"patterncoo_20sample_cam2_x.tsr")