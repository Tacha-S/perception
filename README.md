# PERCH 2.0 : Fast and Accurate CUDA-based 3-Dof and 6-Dof Pose Estimation

![Image of 6-Dof](images/6dof_flow.png)

Overview
--------
This library provides implementations for single and multi-object 3-Dof and 6-Dof pose estimation from RGB-D sensor data and 3D CAD models. It can evaluate thousands of poses in parallel on a GPU in order to find the pose that best explains the input scene using CUDA. Each pose is refined in parallel through CUDA based GICP. PERCH 2.0 works in conjunction with an instance segmentation CNN for 6-Dof pose estimation (Tested with **YCB Video Dataset**). 

The libray is the official implementation of "PERCH 2.0 : Fast and Accurate GPU-based Perception via Search for Object Pose Estimation" accepted at **IROS 2020** [[PDF](https://arxiv.org/abs/2008.00326)]

Notable Features
----------------
- CUDA Rendering : Render thousands of RGB and depth images of multiple-objects in parallel
- CUDA Point clouds : Convert rendered images to point clouds in parallel
- CUDA GICP : Adjust thousands of poses of different objects accurately under occlusion and simultanesouly using parallel GICP
- CUDA KNN : Do a parallel KNN search between an input point cloud and thousands of rendered point clouds
- Works without a CNN for 3-Dof pose estimation
- Python/C++ pipeline for running instance segmentation CNN (Mask-RCNN) and PERCH 2.0 for 6-Dof pose estimation 
- Python interface for running experiments on large datasets and computing pose accuracy metrics (AUC, ADD-S, ADD)
- ROS interface for running with robotic platforms like PR2 etc.

System Requirements
------------
- Ubuntu (>= 16.04) 
- NVidia GPU (>= 4GB)
- NVidia Drivers > 440.33
- Docker
- NVidia-Docker toolkit

Docker Setup
------------
Follow the steps outlined in this [Wiki](https://github.com/SBPL-Cruz/perception/wiki/Running-With-Docker#using-docker-image) to setup the code on your machine. The code will be built and run from the Docker image and no local installation of dependencies (apart from NVidia drivers and Docker itself) is needed.

Running with YCB Video Dataset
-----------------------
Follow the steps outlined in this [Wiki](https://github.com/SBPL-Cruz/perception/wiki/Running-With-Docker#running-6-dof--ycb_video_dataset) to run the code on YCB Video Dataset. It can run using PoseCNN masks, ground truth masks or a custom MaskRCNN model trained by us. The model is trained to detect full bounding boxes and instance segmentation masks of YCB objects in the dataset.

Results : 
![](https://cdn.mathpix.com/snip/images/oUibumUIATzIIYEr81i_wcgp7rs0HyF109AcUCspE3Q.original.fullsize.png)

Running with Robot
------------------
PERCH 2.0 communicates with the robot's camera using ROS. Follow the steps outlined in this [Wiki](https://github.com/SBPL-Cruz/perception/wiki/Running-on-Robot) to first test the code with bagfiles. You can then use the bagfile setup of your choice and modify it as per the robot requirements.

Author
------
Created by [Aditya Agarwal](http://adityaagarwal.in) at the [Search Based Planning Lab](http://sbpl.net), Robotics Institute, CMU. Please direct any questions about the code or paper to the [Issues](https://github.com/SBPL-Cruz/perception/issues) section of this repo. 

Citation
----
Please use the citation below for our [paper](https://arxiv.org/abs/2008.00326) accepted at **IROS 2020**, if you use our code :
```
@inproceedings{Agarwal2020PERCH2,
  title={PERCH 2.0 : Fast and Accurate GPU-based Perception via Search for Object Pose Estimation},
  author={Aditya Agarwal and Yupeng Han and Maxim Likhachev},
  year={2020},
  booktitle={IROS}
}
```
