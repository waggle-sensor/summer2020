## Build Docker Image with Everything inside

### Using plugin-tensorflow-ros:v2.0.7 as base image

https://github.com/waggle-sensor/edge-plugins

Edmward's base image:
```
#Download base image ubuntu 18.04
FROM ubuntu:18.04
#Author
MAINTAINER Eduard Gibert Renart
#Extra metadata
LABEL version="1.0"
LABEL description="Edge Processor Dockerfile."
#Install and source ansible
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -qq install -y --no-install-recommends \
     htop \  
     iotop \  
     iftop \  
     bwm-ng \  
     screen \  
     git \  
     python-dev \  
     python-pip \  
     python3-dev \  
     python3-pip \  
     dosfstools \  
     parted \  
     bash-completion \  
     v4l-utils \  
     network-manager \  
     usbutils \  
     nano \  
     stress-ng \  
     rabbitmq-server \  
     python-psutil \  
     python3-psutil \  
     fswebcam \  
     alsa-utils \  
     portaudio19-dev \  
     libavcodec-extra57 \
     libavformat57 \
     libavutil55 \  
     libc6 \  
     libcairo2 \  
     libgdk-pixbuf2.0-0 \  
     libglib2.0-0 \  
     libstdc++6 \  
     zlib1g \  
     python3-tabulate \  
     python3-pika \  
     lsof \
     python3-exif \
     libv4l-dev \
     libdc1394-22 \
     libgtk2.0-0 \
     python3 \
     ffmpeg \
     libasound2-dev \
     portaudio19-dev \  
     libportaudio2 \
     libportaudiocpp0 \  
     python3-pyaudio \
     mc \
     tmux \
     rsync \
     python3-pip \
     libstdc++6 \
     libc6 \
     libgcc1 \
     gcc-8-base \    
     libstdc++6 \
     build-essential \
     cmake \
     git \
     wget \
     unzip \
     yasm \
     pkg-config \
     libswscale-dev \
     libtbb2 \
     libtbb-dev \
     libjpeg-dev \
     libpng-dev \
     libtiff-dev \
     libavformat-dev \
     libpq-dev \
     libatlas-base-dev \
     libilmbase-dev \
     libopenexr-dev \
     libgstreamer1.0-0 \
     libqtgui4 \
     libqt4-test \
     libjpeg62 \
     libopenblas-dev \
     libhdf5-serial-dev \
     hdf5-tools \
     libhdf5-dev \
     zlib1g-dev \
     zip \
     libjpeg8-dev \
     lame \
     && rm -rf /var/lib/apt/lists/*
#Add repo for deprecated packages
RUN echo "deb http://ports.ubuntu.com/ubuntu-ports/ xenial main restricted" >> /etc/apt/sources.list
RUN echo "deb http://ports.ubuntu.com/ubuntu-ports/ xenial-updates main restricted" >> /etc/apt/sources.list
RUN echo "deb http://ports.ubuntu.com/ubuntu-ports/ xenial universe" >> /etc/apt/sources.list
RUN echo "deb http://ports.ubuntu.com/ubuntu-ports/ xenial-updates universe" >> /etc/apt/sources.list
#Install deprecated packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -qq install -y \
    libpng12-0 \
    libswscale-ffmpeg3 \
    libhdf5-10 \
    libhdf5-serial-dev \
    libjasper1 \
    libvtk6.2 \
    gfortran \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install setuptools --upgrade wheel
COPY cuda-repo.tar.gz /var
COPY nvidia_drivers.tbz2 /
RUN cd /var && tar -xzvf cuda-repo.tar.gz && cd cuda-repo-10-0-local-10.0.166 && dpkg-scanpackages . | gzip > Packages.gz && echo "deb [trusted=yes] file:///var/cuda-repo-10-0-local-10.0.166 /" >> /etc/apt/sources.list.d/cuda-10-0-local-10.0.166.list && rm /var/cuda-repo.tar.gz
RUN apt update && apt install -y cuda-toolkit-10-0 \
    libcudnn7 \
    && rm -rf /var/lib/apt/lists/*
RUN cd / && tar -jxvf nvidia_drivers.tbz2 && ldconfig /usr/lib/aarch64-linux-gnu/tegra && rm nvidia_drivers.tbz2
RUN wget https://developer.download.nvidia.com/compute/redist/jp/v42/tensorflow-gpu/tensorflow_gpu-1.13.1+nv19.5-cp36-cp36m-linux_aarch64.whl && pip3 install tensorflow_gpu-1.13.1+nv19.5-cp36-cp36m-linux_aarch64.whl && rm tensorflow_gpu-1.13.1+nv19.5-cp36-cp36m-linux_aarch64.whl
RUN wget https://nvidia.box.com/shared/static/j2dn48btaxosqp0zremqqm8pjelriyvs.whl -O torch-1.1.0-cp36-cp36m-linux_aarch64.whl && pip3 install torch-1.1.0-cp36-cp36m-linux_aarch64.whl && rm torch-1.1.0-cp36-cp36m-linux_aarch64.whl
RUN pip3 install pydub \
    pyaudio \
    piexif \
    tinydb \
    wave \
    v4l2 \
    imutils \
    keras
WORKDIR /tmp
RUN git clone https://github.com/opencv/opencv.git \
  && git clone https://github.com/opencv/opencv_contrib.git \
  && cd /tmp/opencv_contrib \
  && git checkout 4.1.0 \
  && cd /tmp/opencv \
  && git checkout 4.1.0 \
  && mkdir build \
  && cd build \
  && numpy_loc=$(pip3 show numpy | grep Location | cut -d ':' -f 2 | tr -d ' ') \
  && cmake -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX=/usr/local \
  -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib/modules \
  -D PYTHON3_EXECUTABLE=/usr/bin/python3 \
  -D PYTHON3_NUMPY_INCLUDE_DIRS=${numpy_loc}/numpy/core/include \
  -D ENABLE_NEON=ON \
  -D WITH_FFMPEG=ON \
  -D WITH_LIBV4L=ON \
  -D WITH_OPENCL=ON \
  -D CPACK_BINARY_DEB:BOOL=OFF .. \
  && number_of_core=$(cat /proc/cpuinfo | grep processor | wc -l) \
  && make -j${number_of_core}
RUN cd /tmp/opencv/build \
  && make install \
  && make package \
  && rm -Rf /tmp/opencv_contrib && rm -Rf /tmp/opencv
RUN git clone https://github.com/waggle-sensor/waggle_image.git && pip3 install waggle_image/var/cache/pip3/archives/v4l2-0.2.tar.gz && rm -rf waggle-sensor
```

https://github.com/waggle-sensor/edge-plugins/blob/master/plugin-base-light/Dockerfile

### Using plugin-tensorflow-ros:v2.0.7 as base image