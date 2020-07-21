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
Install and set up ROS2 based on https://github.com/waggle-sensor/summer2020/blob/master/liu/doc/Docker-ROS2-Setup.md.

## Reference
 - https://github.com/waggle-sensor/edge-plugins
 - https://github.com/waggle-sensor/edge-plugins/blob/master/plugin-base-light/Dockerfile
