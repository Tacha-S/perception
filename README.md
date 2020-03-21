# PERCH 2.0 : Fast and High Quality GPU-based Perception via Search for Object Pose Estimation

Overview
--------
This library provides implementations for single and multi-object pose estimation from RGB-D sensor (MS Kinect, ASUS Xtion etc.) data. It renders 100s of poses in parallel on a GPU in order to find the pose that best explains the observed scene.

Features
------------
* Detect 3Dof poses (in a tabletop setting) in under 1s
* No pretraining required
* Works with depth data from typical RGBD cameras
* Get high detection accuracies required for tasks such as robotic manipulation 

Requirements
------------
- Ubuntu OS 
- ROS Kinetic
- Object 3D Mesh Models
- NVidia GPU (>= 4GB)

Setup (For running with a robot camera or bagfile recorded from robot)
-----
1. Install Docker using the steps [here](https://github.com/fmidev/smartmet-server/wiki/Setting-up-Docker-and-Docker-Compose-(Ubuntu-16.04-and-18.04.1)) for Ubuntu

2. Install Nvidia-Docker toolkit for GPU access. Follow steps [here](https://github.com/NVIDIA/nvidia-docker/) (for Docker version > 19) or [here](https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0)) (for Docker version < 19) depending on your Docker version.
   
3. Pull the latest Docker image that contains all dependencies required to run PERCH 2.0 :
    ```
    docker pull thecatalyst25/perch_debug:2.0
    ```
4.  Clone this repo which contains the code locally (skip this step if you already have it cloned) : 
    ```
    git clone https://github.com/SBPL-Cruz/perception -b gpu
    ```
5. Create a folder for visualization of the output and to store the 3D models. This example works with 3D models of YCB objects :
   ```
   cd perception/sbpl_perception
   mkdir visualization
   mkdir -p data/ycb_models
   ```
6. Download the YCB models from this [link](https://drive.google.com/file/d/1gmcDD-5bkJfcMKLZb3zGgH_HUFbulQWu/view?usp=sharing) and extract them in the ```ycb_models``` folder created above
7. Download the example bag file from this [link]() 

8. Run the Docker while mounting the cloned repo to the workspace inside the Docker and build the workspace :
```
docker run --runtime nvidia -it --net=host -v perception:/ros_python3_ws/src/perception  perch_debug:2.0
source /opt/ros/kinetic/setup.bash
cd ros_python3_ws/
catkin build object_recognition_node
```
9. Run the code. This will start the service which will wait for camera input and request to locate a given object :
```
roscore&
source /ros_python3_ws/devel/setup.bash 
roslaunch object_recognition_node pr2_conveyor_object_recognition.launch
```
10. Request for objects and run the bag file :
```
rosbag play 3dof_1_2020-03-04-14-17-00.bag

rostopic pub /requested_object std_msgs/String "data: '004_sugar_box 005_tomato_soup_can 002_master_chef_can 006_mustard_bottle 010_potted_meat_can'"
```
11. Modify required parameters in the files below if needed :
```
sbpl_perception/config/camera_config.yaml
sbpl_perception/config/pr2_conv_env_config.yaml
```

Tweaking Params
------------
1. Tweek table_height such that no points of the table or floor are visible in /perch/input_point_cloud
2. Tweak xmin, xmax, ymin, ymax such that no visible points of required objects get excluded from the point cloud
3. Tweak downsampling leaf size to get desired speed and accuracy



