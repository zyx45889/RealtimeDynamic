## Real-time Dynamic

This is the code for CVPR2024 paper:

**Real-time Acquisition and Reconstruction of Dynamic Volumes with Neural Structured Illumination**

by Yixin Zeng, Zoubin Bi, Mingrui Yin, Xiang Feng, Kun Zhou and Hongzhi Wu*.

[Project Page](https://svbrdf.github.io/publications/realtimedynamic/project.html)

### Introduction

We propose a novel framework for real-time acquisition and reconstruction of temporally-varying 3D phenomena with high quality. The core of our framework is a deep neural network, with an encoder that directly maps to the structured illumination during acquisition, a decoder that predicts a 1D density distribution from single-pixel measurements under the optimized lighting, and an aggregation module that combines the predicted densities for each camera into a single volume. It enables the automatic and joint optimization of physical acquisition and computational reconstruction, and is flexible to adapt to different hardware configurations. Using as few as 6 pre-optimized structured light patterns, we capture and reconstruct high-quality, dynamic 3D volumes from corresponding image measurements at different views, with a lightweight projector-camera setup. We achieve a performance of 40 volumes per second for both acquisition and reconstruction.

<img src="./imgs/teaser.jpg" alt="teaser" style="zoom: 15%;" />

We utilize a auto-encoder pipeline. Starting from a synthetic/physical 3D density volume, we first project the pre-optimized light patterns (i.e., weights in the encoder) to the volume. For each valid pixel at each camera view, we send all its measurements along with the resampled local illumination conditions to a decoder, to predict a 1D density distribution over the corresponding camera ray. All density distributions for one camera are then collected and resampled into a single 3D volume. In the multi-camera case, the predicted volumes for each camera are fused to obtain the final result.

<img src="./imgs/netpipeline.jpg" alt="netpipeline" style="zoom: 15%;" />

Our network consists of 3 parts:

- The encoder simulates the measurement process, linking the light patterns, the input density volume and the output image measurements in a differentiable manner. 
- Our decoder consists of 4 fc layers and works on a per-pixel basis.  It takes as input the measurements at the same pixel location and the corresponding local incident lighting, and outputs a 1D density distribution along the corresponding camera ray. 
- In the multi-camera case, we take as input the 3D volumes predicted by the decoder for each camera, and fuse the multi-view information to output a high-quality volume with a 3D UNet.

<img src="./imgs/decoder.jpg" alt="decoder" style="zoom:10%;" />

### Data Generation

We generate our data with mantaflow, based the code from  [tempoGAN](https://github.com/thunil/tempoGAN).

#### Installation

- h5py
- python-numpy
- [Mantaflow](http://mantaflow.com/install.html)

Note: for ubuntu version after 20.04, it is likely package ‘qt5-default’ can not be installed. Instead you can use:

```
sudo apt-get install build-essential
sudo apt-get install qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools
sudo apt-get install qtcreator
```

And numpy should be allowed in mantaflow installation, so the cmake sentence should be changed into:

```
cmake .. -DNUMPY=ON -DOPENMP=ON
```

Then use the following command to generate data:

```
./mantaflow/mantaflow-master/build/manta ./gen_data/gen_data.py
```

It takes about 3.5 hours to generate all the data.

### Training

#### Environment

- pytorch-cuda
- numpy
- wandb
- tqdm
- h5py
- opencv-python
- annoy

#### Train

After wandb login, run the following command with multiple gpus:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train128_cal3.py
```

#### Test

Run the following command:

```
python predict.py
```

The result is in `./our_result/`.

### Apply to Your Device

For specific real-world device, the ray trace information used for resample process in training and testing need to be precomputed before the training. The ray trace information for our device is provided in `./ray_trace_info`.

Take camera-projector device for example, with calibrated camera and projector(also see calibration algorithm in supplementary material of our paper), run the following code:

```
./gen_ray_trace_info/raysample_kdtree_subprocess.py
./gen_ray_trace_info/raysample_rescale_subprocess.py
./gen_ray_trace_info/cal_ray_trace_info.py
```

The camera and projector calibration information of ours is provided in `./ray_trace_info/intrinsic_proj_cam_42000.yml`. The volume location with respect to the projector in line 109 of `./gen_ray_trace_info/raysample_kdtree.py` is roughly estimated from real scene. The `hmin` and `hmax` parameter in `./gen_ray_trace_info/raysample_rescale.py` and  `./gen_ray_trace_info/raysample_rescale_subprocess.py` is roughly estimated from the volume location in the real photos taken by the camera. The `cpunum` parameter in `./gen_ray_trace_info/raysample_rescale_subprocess.py` and `./gen_ray_trace_info/raysample_kdtree_subprocess.py` is decided by the number of cpu in your machine.

It takes about 2 hour to finish the precomputation.