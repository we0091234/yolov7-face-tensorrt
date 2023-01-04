cmake_minimum_required(VERSION 3.10)
project(yolov7_face)

add_definitions(-std=c++11)
add_definitions(-w)
# option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
find_package(CUDA REQUIRED)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

#cuda 
include_directories(/mnt/Gu/softWare/cuda-11.0/targets/x86_64-linux/include)
link_directories(/mnt/Gu/softWare/cuda-11.0/targets/x86_64-linux/lib)

#tensorrt 
include_directories(/mnt/Gpan/tensorRT/TensorRT-8.2.0.6//include/)
link_directories(/mnt/Gpan/tensorRT/TensorRT-8.2.0.6/lib/)

#opencv 
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

#onnx2trt
add_subdirectory(${PROJECT_SOURCE_DIR}/onnx2trt)

cuda_add_executable(yolov7_face yolov7-face.cpp utils.cpp preprocess.cu postprocess.cu)

target_link_libraries(yolov7_face nvinfer)
target_link_libraries(yolov7_face cudart)
target_link_libraries(yolov7_face nvonnxparser)
target_link_libraries(yolov7_face ${OpenCV_LIBS})

add_definitions(-O2 -pthread)