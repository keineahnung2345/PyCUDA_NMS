# Instruction
To run this file, you should have installed OpenCV.

If you don't have OpenCV installed, follow the instructions here: [Instructions on installing OpenCV on Ubuntu](https://github.com/keineahnung2345/Parallel_NMS/blob/master/GPU/How%20to%20install%20OpenCV%20on%20Ubuntu.md)

## Build a .o file from .cu file
```sh
$ nvcc -o nms.o -std=c++11 nms.cu  `pkg-config opencv --cflags --libs`
```

Notes:
1. -std=c++11: compile with c++11
2. `pkg-config opencv --cflags --libs`: specify the route of libraries

## Run a .o file
```sh
$ chmod 755 nms.o
$ ./nms.o
```
