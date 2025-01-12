ARG CUDA="10.2"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu18.04

# RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# install basics
RUN apt-get update -y \
 && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ \
 && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev

RUN apt-get update && apt-get install -y --allow-unauthenticated\
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --allow-unauthenticated\
    vim \
    && rm -rf /var/lib/apt/lists/*
    
# Upgrade boost to work with cuda 9
# RUN add-apt-repository ppa:mhier/libboost-latest && apt-get update \
#     && rm -rf /var/lib/apt/lists/*

# RUN apt-get update && apt-get install -y --allow-unauthenticated\
#     libboost1.67-dev \
#     && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN  apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*

RUN wget -O /miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh \
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p /miniconda \
 && rm /miniconda.sh

RUN ls /miniconda/
ENV PATH=/miniconda/bin:$PATH

# Create a Python 3.6 environment
RUN conda install -y conda-build \
 && conda create -y --name py36 python=3.6.7 \
 && conda clean -ya

ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN conda install -y ipython
RUN pip install ninja yacs cython matplotlib opencv-python tqdm tensorboardX


# Install PyTorch 1.0 Nightly
ARG CUDA
RUN conda install pytorch==1.0.0 torchvision==0.2.1 cudatoolkit=${CUDA} -c pytorch \
 && conda clean -ya

RUN conda install pillow==6.1

# install pycocotools
RUN git clone https://github.com/cocodataset/cocoapi.git \
 && cd cocoapi/PythonAPI \
 && python setup.py build_ext install



##### ROS #####

# install packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu bionic main" > /etc/apt/sources.list.d/ros-latest.list

RUN wget http://packages.ros.org/ros.key -O - | apt-key add -
#RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# install bootstrap tools
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y --allow-unauthenticated\
    python-rosdep \
    python-rosinstall \
    python-vcstools \
    && rm -rf /var/lib/apt/lists/*

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# bootstrap rosdep
RUN rosdep init \
    && rosdep update

# install ros packages
ENV ROS_DISTRO melodic
RUN apt-get update && apt-get install -y --allow-unauthenticated\
    # ros-melodic-ros-core=1.3.2-0* \
    ros-melodic-desktop-full \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --allow-unauthenticated\
    python-catkin-tools \
    && rm -rf /var/lib/apt/lists/*

# RUN apt-get update && apt-get install -y --allow-unauthenticated\
#     # ros-melodic-pcl-* \
#     ros-melodic-desktop-full \
#     && rm -rf /var/lib/apt/lists/*

### STuff for PERCH ###
# For ROS python 3
RUN pip install dipy git+git://github.com/waspinator/coco.git@2.1.0
RUN pip install git+git://github.com/waspinator/pycococreator.git@0.2.0
RUN pip install glumpy pyopengl rospkg python-pcl defusedxml plyfile sklearn empy pyyaml


RUN apt-get update && apt-get install -y --allow-unauthenticated\
    libbullet-dev \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --allow-unauthenticated\
    python3.6-dev \
    && rm -rf /var/lib/apt/lists/*

# For PERCH
RUN apt-get update && apt-get install -y --allow-unauthenticated\
    libgsl-dev libvtk6-dev libglew-dev libsdl2-dev ros-melodic-octomap\
    && rm -rf /var/lib/apt/lists/*


# Install workspace
ENV CPLUS_INCLUDE_PATH=/usr/include/python3.6m/:$CPLUS_INCLUDE_PATH

RUN git clone https://github.com/SBPL-Cruz/ros_python3_ws \
 && cd ros_python3_ws \
 && rm -r vision_opencv \
 && git clone https://github.com/ros-perception/vision_opencv.git -b melodic \
 && catkin init \
 && catkin config -DPYTHON_EXECUTABLE=/miniconda/envs/py36/bin/python -DPYTHON_INCLUDE_DIR=/miniconda/envs/py36/include  -DPYTHON_LIBRARY=/miniconda/envs/py36/lib/libpython3.6m.so

# RUN chmod +x /opt/ros/melodic/setup.sh \
#     /opt/ros/melodic/setup.sh \
#     && cd /ros_python3_ws \
#     && catkin build

# Install PyTorch Detection
ARG CACHEBUST=1
ARG FORCE_CUDA="1"
ENV FORCE_CUDA=${FORCE_CUDA}
RUN git clone https://github.com/sbpl-cruz/maskrcnn-benchmark.git -b fat_pose_dataset_new \
 && cd maskrcnn-benchmark \
 && python setup.py build develop


# Install apex (Doesnt cpp ext with Cuda 8.0)
RUN git clone https://github.com/NVIDIA/apex.git \
 && cd apex \
 && pip install -v --no-cache-dir ./
#  && python setup.py install --cuda_ext --cpp_ext

# Install PERCH
RUN pip install catkin_pkg
# For PERCH
RUN apt-get update && apt-get install -y --allow-unauthenticated\
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Install Latest Eigen, for fast_gicp
RUN git clone https://github.com/eigenteam/eigen-git-mirror.git \
 && cd eigen-git-mirror \
 && mkdir build \
 && cd build \
 && cmake .. -DCMAKE_INSTALL_PREFIX=/usr/ \
 && sudo make install 

#  && python setup.py install --cuda_ext --cpp_ext


### Release Version below, code cloned inside docker ###
# RUN cd /ros_python3_ws/src \
#     && git clone https://github.com/SBPL-Cruz/improved-mha-planner -b renamed \
#     && git clone https://github.com/SBPL-Cruz/sbpl_utils.git -b renamed \
#     && git clone https://github.com/SBPL-Cruz/perception -b gpu_icp_ycb

# # If release then clone the main code, else it will be mounted by user during runtime
# # RUN echo "$TYPE"
# # RUN if [[ "$TYPE" = "RELEASE" ]; then "cd /ros_python3_ws/src \
# #     && git clone https://github.com/SBPL-Cruz/perception -b gpu_icp_ycb"; else echo "Not cloning code"; fi

# RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; cd /ros_python3_ws/; catkin build sbpl_perception'
# ENTRYPOINT /bin/bash -c '. /opt/ros/melodic/setup.bash; \
#                          . /ros_python3_ws/devel/setup.bash; \
#                          cd /ros_python3_ws/src/perception/sbpl_perception/src/scripts/tools/fat_dataset; \
#                          Xvfb :5 -screen 0 800x600x24 & export DISPLAY=:5; \
#                          python fat_pose_image.py --config config_docker.yaml'

RUN wget -qO- "https://cmake.org/files/v3.17/cmake-3.17.0-Linux-x86_64.tar.gz" | tar --strip-components=1 -xz -C /usr/local

RUN pip install pandas glfw

# From https://stackoverflow.com/questions/54483960/pyopengl-headless-rendering for PyOpenGL
RUN mkdir /glfw \
    && git clone https://github.com/glfw/glfw.git /glfw \
    && cd /glfw \
    && cmake -DBUILD_SHARED_LIBS=ON . \
    && make
ENV PYGLFW_LIBRARY=/glfw/src/libglfw.so

### Debug Version below, code not cloned inside docker ###
RUN cd /ros_python3_ws/src \
    && git clone https://github.com/SBPL-Cruz/improved-mha-planner -b renamed \
    && git clone https://github.com/SBPL-Cruz/sbpl_utils.git -b renamed 

# RUN /bin/bash -c 'Xvfb :5 -screen 0 800x600x24 & export DISPLAY=:5; \
#                         . /opt/ros/melodic/setup.bash; \
#                         cd /ros_python3_ws/;'
