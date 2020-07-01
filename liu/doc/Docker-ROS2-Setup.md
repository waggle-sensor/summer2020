## Set up ROS2 with YOLOv3 applications running inside

### Pull base Docker image (L4T with ML tools)
```
$ docker pull nvcr.io/nvidia/l4t-ml:r32.4.2-py3
$ sudo docker run -it nvcr.io/nvidia/l4t-ml:r32.4.2-py3 /bin/bash
```

### Install ROS2 
 - Install based on https://index.ros.org/doc/ros2/Installation/Dashing/Linux-Install-Debians/.
 - Set up workspace based on https://index.ros.org/doc/ros2/Tutorials/Colcon-Tutorial/.

### Install YOLOv3 application:
```
$ cd ~/ros2_example_ws/src
$ git clone https://github.com/ros2/openrobotics_darknet_ros.git
$ cd ..
$ colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
```

### Install Darknet_vendor and vision_msgs
```

```

### Install ros2_usb_camera


### Issues and solutions

#### No Darknet vendor
```
root@nvidia-desktop:~/ros2_example_ws# colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-select openrobotics_darknet_ros
Starting >>> openrobotics_darknet_ros
--- stderr: openrobotics_darknet_ros                         
CMake Error at CMakeLists.txt:15 (find_package):
  By not providing "Finddarknet_vendor.cmake" in CMAKE_MODULE_PATH this
  project has asked CMake to find a package configuration file provided by
  "darknet_vendor", but CMake did not find one.

  Could not find a package configuration file provided by "darknet_vendor"
  with any of the following names:

    darknet_vendorConfig.cmake
    darknet_vendor-config.cmake

  Add the installation prefix of "darknet_vendor" to CMAKE_PREFIX_PATH or set
  "darknet_vendor_DIR" to a directory containing one of the above files.  If
  "darknet_vendor" provides a separate development package or SDK, be sure it
  has been installed.
  
  
---
Failed   <<< openrobotics_darknet_ros [0.79s, exited with code 1]
                                
Summary: 0 packages finished [1.23s]
  1 package failed: openrobotics_darknet_ros
  1 package had stderr output: openrobotics_darknet_ros

```
Solution: add darknet_vendor and build it.

#### No cuda_runtime.h:
```
root@nvidia-desktop:~/ros2_example_ws/src# cd ..
root@nvidia-desktop:~/ros2_example_ws# colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-select darknet_vendor
Starting >>> darknet_vendor
--- stderr: darknet_vendor                               
In file included from /root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/blas.h:3:0,
                 from /root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/blas.c:1:
/root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/include/darknet.h:11:14: fatal error: cuda_runtime.h: No such file or directory
     #include "cuda_runtime.h"
              ^~~~~~~~~~~~~~~~
In file included from /root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/activations.h:3:0,
                 from /root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/activations.c:1:
/root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/include/darknet.h:11:14: fatal error: cuda_runtime.h: No such file or directory
     #include "cuda_runtime.h"
              ^~~~~~~~~~~~~~~~
compilation terminated.
compilation terminated.
In file included from /root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/cuda.h:4:0,
                 from /root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/convolutional_layer.h:4,
                 from /root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/batchnorm_layer.c:1:
/root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/include/darknet.h:11:14: fatal error: cuda_runtime.h: No such file or directory
     #include "cuda_runtime.h"
              ^~~~~~~~~~~~~~~~
compilation terminated.
```
Solution: Add "include_directories("${CUDA_INCLUDE_DIRS}")" into CMakeList.txt.

#### OpenCV4 issue 

```
root@nvidia-desktop:~/ros2_example_ws# colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-select darknet_vendor
Starting >>> darknet_vendor
--- stderr: darknet_vendor                               
/root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/data.c: In function 'load_regression_labels_paths':
/root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/data.c:613:13: warning: ignoring return value of 'fscanf', declared with attribute warn_unused_result [-Wunused-result]
             fscanf(file, "%f", &(y.vals[i][j]));
             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

......

/root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/utils.c: In function 'read_file':
/root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/utils.c:270:5: warning: ignoring return value of 'fread', declared with attribute warn_unused_result [-Wunused-result]
     fread(text, 1, size, fp);
     ^~~~~~~~~~~~~~~~~~~~~~~~
/root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/utils.c: In function 'fgetl':
/root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/utils.c:358:9: warning: ignoring return value of 'fgets', declared with attribute warn_unused_result [-Wunused-result]
         fgets(&line[curr], readsize, fp);
         ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/image_opencv.cpp:12:1: error: 'IplImage' does not name a type; did you mean 'image'?
 IplImage *image_to_ipl(image im)
 ^~~~~~~~
 image
/root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/image_opencv.cpp:28:20: error: 'IplImage' was not declared in this scope
 image ipl_to_image(IplImage* src)
                    ^~~~~~~~
/root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/image_opencv.cpp:28:20: note: suggested alternative: 'image'
 image ipl_to_image(IplImage* src)
                    ^~~~~~~~
                    image
/root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/image_opencv.cpp:28:30: error: 'src' was not declared in this scope
 image ipl_to_image(IplImage* src)
                              ^~~
/root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/image_opencv.cpp:28:30: note: suggested alternative: 'sec'
 image ipl_to_image(IplImage* src)
                              ^~~
                              sec
/root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/image_opencv.cpp:29:1: error: expected ',' or ';' before '{' token
 {
 ^
/root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/image_opencv.cpp: In function 'cv::Mat image_to_mat(image)':
/root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/image_opencv.cpp:54:5: error: 'IplImage' was not declared in this scope
     IplImage *ipl = image_to_ipl(copy);
     ^~~~~~~~
/root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/image_opencv.cpp:54:5: note: suggested alternative: 'image'
     IplImage *ipl = image_to_ipl(copy);
     ^~~~~~~~
     image
/root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/image_opencv.cpp:54:15: error: 'ipl' was not declared in this scope
     IplImage *ipl = image_to_ipl(copy);
               ^~~
/root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/image_opencv.cpp:54:21: error: 'image_to_ipl' was not declared in this scope
     IplImage *ipl = image_to_ipl(copy);
                     ^~~~~~~~~~~~
/root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/image_opencv.cpp:54:21: note: suggested alternative: 'image_to_mat'
     IplImage *ipl = image_to_ipl(copy);
                     ^~~~~~~~~~~~
                     image_to_mat
/root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/image_opencv.cpp:55:13: error: 'cvarrToMat' was not declared in this scope
     Mat m = cvarrToMat(ipl, true);
             ^~~~~~~~~~
/root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/image_opencv.cpp:56:5: error: 'cvReleaseImage' was not declared in this scope
     cvReleaseImage(&ipl);
     ^~~~~~~~~~~~~~
/root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/image_opencv.cpp: In function 'image mat_to_image(cv::Mat)':
/root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/image_opencv.cpp:63:5: error: 'IplImage' was not declared in this scope
     IplImage ipl = m;
     ^~~~~~~~
/root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/image_opencv.cpp:63:5: note: suggested alternative: 'image'
     IplImage ipl = m;
     ^~~~~~~~
     image
/root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/image_opencv.cpp:64:30: error: 'ipl' was not declared in this scope
     image im = ipl_to_image(&ipl);
                              ^~~
/root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/image_opencv.cpp: In function 'void* open_video_stream(const char*, int, int, int, int)':
/root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/image_opencv.cpp:75:20: error: 'CV_CAP_PROP_FRAME_WIDTH' was not declared in this scope
     if(w) cap->set(CV_CAP_PROP_FRAME_WIDTH, w);
                    ^~~~~~~~~~~~~~~~~~~~~~~
/root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/image_opencv.cpp:76:20: error: 'CV_CAP_PROP_FRAME_HEIGHT' was not declared in this scope
     if(h) cap->set(CV_CAP_PROP_FRAME_HEIGHT, w);
                    ^~~~~~~~~~~~~~~~~~~~~~~~
/root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/image_opencv.cpp:77:22: error: 'CV_CAP_PROP_FPS' was not declared in this scope
     if(fps) cap->set(CV_CAP_PROP_FPS, w);
                      ^~~~~~~~~~~~~~~
/root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/image_opencv.cpp:77:22: note: suggested alternative: 'CV_CPU_POPCNT'
     if(fps) cap->set(CV_CAP_PROP_FPS, w);
                      ^~~~~~~~~~~~~~~
                      CV_CPU_POPCNT
/root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/image_opencv.cpp: In function 'void make_window(char*, int, int, int)':
/root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/image_opencv.cpp:126:33: error: 'CV_WND_PROP_FULLSCREEN' was not declared in this scope
         setWindowProperty(name, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
                                 ^~~~~~~~~~~~~~~~~~~~~~
/root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/image_opencv.cpp:126:57: error: 'CV_WINDOW_FULLSCREEN' was not declared in this scope
         setWindowProperty(name, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
                                                         ^~~~~~~~~~~~~~~~~~~~
/root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/image_opencv.cpp: In function 'image load_image_cv(char*, int)':
/root/ros2_example_ws/build/darknet_vendor/_deps/darknet-download-src/src/image_opencv.cpp:105:15: warning: ignoring return value of 'int system(const char*)', declared with attribute warn_unused_result [-Wunused-result]
         system(buff);
         ~~~~~~^~~~~~
make[2]: *** [CMakeFiles/darknet.dir/_deps/darknet-download-src/src/image_opencv.cpp.o] Error 1
make[2]: *** Waiting for unfinished jobs....
make[1]: *** [CMakeFiles/darknet.dir/all] Error 2
make: *** [all] Error 2
---
Failed   <<< darknet_vendor [24.4s, exited with code 2]

Summary: 0 packages finished [24.8s]
  1 package failed: darknet_vendor
  1 package had stderr output: darknet_vendor

```
Solution: downgrade the OpenCV to 3.4.0 can solve the problem.


### Reference
 - https://github.com/ros2/openrobotics_darknet_ros
 - https://github.com/ros2/darknet_vendor
