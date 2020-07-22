## Build Docker Image with Everything inside

### Using plugin-tensorflow-ros:v2.0.7 as base image

```
sudo apt-get install nano
pip3 install git+https://github.com/waggle-sensor/pywaggle # waggle-0.31.0
pip3 --no-cache-dir install torch torchvision tensorboard tqdm wandb
pip3 --no-cache-dir install tqdm fcn
sudo apt-get update  && apt-get install -y  libjpeg8-dev  zlib1g-dev
pip3 install terminaltables tensorboardx tqdm pillow

sudo apt-get install htop iotop iftop bwm-ng screen git python-dev python-pip python3-dev python3-pip dosfstools parted bash-completion v4l-utils network-manager usbutils nano stress-ng rabbitmq-server python-psutil python3-psutil fswebcam alsa-utils portaudio19-dev libavcodec-extra57 libavformat57 libavutil55 libc6 libcairo2 libgdk-pixbuf2.0-0 libglib2.0-0 libstdc++6 zlib1g python3-tabulate python3-pika lsof python3-exif libv4l-dev libdc1394-22 libgtk2.0-0 python3 ffmpeg libasound2-dev portaudio19-dev libportaudio2 libportaudiocpp0 python3-pyaudio mc tmux rsync python3-pip libstdc++6 libc6 libgcc1 gcc-8-base libstdc++6 build-essential cmake git wget unzip yasm pkg-config libswscale-dev libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libavformat-dev libpq-dev libatlas-base-dev libilmbase-dev libopenexr-dev libgstreamer1.0-0 libqtgui4 libqt4-test libjpeg62 libopenblas-dev libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev lame

echo "deb http://ports.ubuntu.com/ubuntu-ports/ xenial main restricted" >> /etc/apt/sources.list
echo "deb http://ports.ubuntu.com/ubuntu-ports/ xenial-updates main restricted" >> /etc/apt/sources.list
echo "deb http://ports.ubuntu.com/ubuntu-ports/ xenial universe" >> /etc/apt/sources.list
echo "deb http://ports.ubuntu.com/ubuntu-ports/ xenial-updates universe" >> /etc/apt/sources.list

sudo apt-get install libpng12-0 libswscale-ffmpeg3 libhdf5-10 libhdf5-serial-dev libjasper1 libvtk6.2 gfortran libopenmpi1.10
rm -rf /var/lib/apt/lists/*

pip3 install setuptools --upgrade wheel
pip3 install pydub pyaudio piexif tinydb wave v4l2 imutils keras
```

After flattening, the size of the docker image is 8.09GB. The mega-arm64-docker image can be pulled using:
```
sudo docker pull liangkailiu/mega-arm64-docker-image:v0.0.3
```

#### issues 

```
Collecting scikit-image (from fcn)
  Downloading https://files.pythonhosted.org/packages/54/fd/c1b0bb8f6f12ef9b4ee8d7674dac82cd482886f8b5cd165631efa533e237/scikit-image-0.17.2.tar.gz (29.8MB)
    100% |################################| 29.8MB 4.1MB/s 
    Complete output from command python setup.py egg_info:
    Traceback (most recent call last):
      File "<string>", line 1, in <module>
      File "/tmp/pip-build-g3sgf9qe/scikit-image/setup.py", line 30, in <module>
        LONG_DESCRIPTION = f.read()
      File "/usr/lib/python3.6/encodings/ascii.py", line 26, in decode
        return codecs.ascii_decode(input, self.errors)[0]
    UnicodeDecodeError: 'ascii' codec can't decode byte 0xc3 in position 4029: ordinal not in range(128)
    
    ----------------------------------------
Command "python setup.py egg_info" failed with error code 1 in /tmp/pip-build-g3sgf9qe/scikit-image/
```
Solution: install ```pip3 install scikit-image==0.16.2 -vv```

### Using nvidia-l4t-base as base image to create micro-arm64-docker-image:

Pull nvidia-l4t-base and launch the Docker container:
```
sudo docker pull nvcr.io/nvidia/l4t-base:r32.4.3
sudo docker run -it nvcr.io/nvidia/l4t-base:r32.4.3 /bin/bash
```
Install ROS2 based on https://github.com/waggle-sensor/summer2020/blob/master/liu/doc/Docker-ROS2-Setup.md and set up workspace based on https://index.ros.org/doc/ros2/Tutorials/Colcon-Tutorial/.

Desktop Install (Recommended): ROS, RViz, demos, tutorials.
```sudo apt install ros-dashing-desktop```

ROS-Base Install (Bare Bones): Communication libraries, message packages, command line tools. No GUI tools.
```sudo apt install ros-dashing-ros-base```

Notes: if choose ros-dashing-desktop version, then almost 2GB of space will needed. For ros-dashing-ros-base, only 486MB of space is needed.

Install PyWaggle:
```
apt install python3-pip
pip3 install git+https://github.com/waggle-sensor/pywaggle
```
After flatterning, the size is 1.24GB. The micro-arm64-docker image can be pulled using:
```
sudo docker pull liangkailiu/micro-arm64-docker-image:v0.0.2
```

### Flatten Docker image layers to decrease the size of the docker image

Before flattening, the layers of the docker images are:
```
nvidia@nvidia-desktop:~$ sudo docker history liangkailiu/mega-arm64-docker-image:v0.0.2
[sudo] password for nvidia: 
IMAGE               CREATED             CREATED BY                                      SIZE                COMMENT
e70976910065        7 hours ago         /bin/sh -c /bin/bash -c "jupyter lab --ip 0.…   370kB               
80d28f1ef9d4        7 hours ago         /bin/sh -c /bin/bash -c "jupyter lab --ip 0.…   599MB               
c133f1cb2a63        13 days ago         /bin/sh -c /bin/bash -c "jupyter lab --ip 0.…   796MB               
f1219da6ad6d        13 days ago         /bin/sh -c /bin/bash -c "jupyter lab --ip 0.…   568MB               
<missing>           2 weeks ago         /bin/sh -c /bin/bash -c "jupyter lab --ip 0.…   1.55GB              
<missing>           3 weeks ago         /bin/sh -c /bin/bash -c "jupyter lab --ip 0.…   323MB               
<missing>           3 weeks ago         /bin/sh -c /bin/bash -c "jupyter lab --ip 0.…   953MB               
<missing>           3 weeks ago         /bin/sh -c /bin/bash -c "jupyter lab --ip 0.…   6.32GB              
<missing>           7 weeks ago         /bin/sh -c /bin/bash -c "jupyter lab --ip 0.…   395kB               
<missing>           7 weeks ago         /bin/sh -c /bin/bash -c "jupyter lab --ip 0.…   99MB                
<missing>           7 weeks ago         /bin/sh -c /bin/bash -c "jupyter lab --ip 0.…   13.4GB              
<missing>           2 months ago        /bin/sh -c #(nop)  CMD ["/bin/sh" "-c" "/bin…   0B                  
<missing>           2 months ago        /bin/sh -c python3 -c "from notebook.auth.se…   129kB               
<missing>           2 months ago        /bin/sh -c jupyter lab --generate-config        164kB               
<missing>           2 months ago        /bin/sh -c pip3 install jupyter jupyterlab -…   148MB               
<missing>           2 months ago        /bin/sh -c pip3 install pycuda --verbose        128kB               
<missing>           2 months ago        /bin/sh -c echo "$PATH" && echo "$LD_LIBRARY…   128kB               
<missing>           2 months ago        /bin/sh -c #(nop)  ENV LD_LIBRARY_PATH=/usr/…   0B                  
<missing>           2 months ago        /bin/sh -c #(nop)  ENV PATH=/usr/local/cuda/…   0B                  
<missing>           2 months ago        /bin/sh -c pip3 install pandas --verbose        138MB               
<missing>           2 months ago        /bin/sh -c pip3 install scikit-learn --verbo…   86.3MB              
<missing>           2 months ago        /bin/sh -c pip3 install scipy --verbose         229MB               
<missing>           2 months ago        /bin/sh -c pip3 install onnx --verbose          20.7MB              
<missing>           2 months ago        /bin/sh -c pip3 install pybind11 --ignore-in…   1.55MB              
<missing>           2 months ago        /bin/sh -c #(nop) COPY dir:065feff9072dd9e1b…   1.07GB              
<missing>           2 months ago        /bin/sh -c #(nop) COPY dir:786926de804e53a4c…   0B                  
<missing>           2 months ago        /bin/sh -c #(nop) COPY dir:639a8b35b101bed5e…   1.01GB              
<missing>           2 months ago        /bin/sh -c #(nop) COPY dir:786926de804e53a4c…   0B                  
<missing>           2 months ago        /bin/sh -c apt-get update &&     apt-get ins…   676MB               
<missing>           3 months ago        /bin/sh -c #(nop)  ENV DEBIAN_FRONTEND=nonin…   0B                  
<missing>           3 months ago        /bin/sh -c #(nop) CMD ["/bin/bash"]             0B                  
<missing>           3 months ago        /bin/sh -c #(nop) ENV NVIDIA_DRIVER_CAPABILI…   0B                  
<missing>           3 months ago        /bin/sh -c #(nop) ENV NVIDIA_VISIBLE_DEVICES…   0B                  
<missing>           3 months ago        /bin/sh -c ldconfig                             84.8kB              
<missing>           3 months ago        /bin/sh -c #(nop) ENV LD_LIBRARY_PATH /usr/l…   0B                  
<missing>           3 months ago        /bin/sh -c #(nop) ENV PATH /usr/local/cuda-1…   0B                  
<missing>           3 months ago        /bin/sh -c ln -s /usr/local/cuda-10.2 /usr/l…   116B                
<missing>           3 months ago        /bin/sh -c #(nop) COPY file:8c7652fcd59c81ab…   786kB               
<missing>           3 months ago        /bin/sh -c #(nop) COPY file:9b0a50749343d692…   677kB               
<missing>           3 months ago        /bin/sh -c #(nop) COPY dir:49012ace4817c3e14…   1.31MB              
<missing>           3 months ago        /bin/sh -c #(nop) COPY dir:f0f880bb838f752eb…   22.1MB              
<missing>           3 months ago        /bin/sh -c #(nop) COPY dir:e194caebd4a9db75d…   51.2MB              
<missing>           3 months ago        /bin/sh -c echo "/usr/local/cuda-10.2/target…   47B                 
<missing>           3 months ago        /bin/sh -c mkdir -p /usr/share/egl/egl_exter…   110B                
<missing>           3 months ago        /bin/sh -c mkdir -p /usr/share/glvnd/egl_ven…   102B                
<missing>           3 months ago        /bin/sh -c rm /usr/share/glvnd/egl_vendor.d/…   0B                  
<missing>           3 months ago        /bin/sh -c echo "/usr/lib/aarch64-linux-gnu/…   70B                 
<missing>           3 months ago        /bin/sh -c apt-get update && apt-get --only-…   40.1MB              
<missing>           3 months ago        /bin/sh -c apt-get update && apt-get install…   212MB               
<missing>           3 months ago        /bin/sh -c #(nop) ADD file:e3b0c44298fc1c149…   717MB               
```

Flattening the layers to decrease the image size:
```
sudo docker run --name mycontainer -it liangkailiu/mega-arm64-docker-image:v0.0.2 bash
sudo docker export --output=mycontainer.tar mycontainer # the container to a tarball (mycontainer.tar is just an example)

sudo cat mycontainer.tar | sudo docker import - liangkailiu/mega-arm64-docker-image:v0.0.3
```

After flattening, the layers of the docker image:
```
nvidia@nvidia-desktop:~$ sudo docker history liangkailiu/mega-arm64-docker-image:v0.0.3
IMAGE               CREATED             CREATED BY          SIZE                COMMENT
31852356adc7        About an hour ago   /bin/bash           369kB               
4e4629cbed08        5 hours ago                             8.09GB              Imported from -
```

## Reference
 - https://github.com/waggle-sensor/edge-plugins
 - https://github.com/waggle-sensor/edge-plugins/blob/master/plugin-base-light/Dockerfile
 - https://stackoverflow.com/questions/27981124/docker-combine-docker-layers-into-image
 - https://pypi.org/project/docker-squash/
 - http://jasonwilder.com/blog/2014/08/19/squashing-docker-images/
