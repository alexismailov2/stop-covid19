cmake_minimum_required(VERSION 3.10)
project(yolov3_opencv)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(OpenCV REQUIRED PATHS)# /home/oleksandr_ismailov/2/release NO_DEFAULT_PATH)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME}
   YOLOv3.hpp
   YOLOv3.cpp
   TimeMeasuring.hpp
   yolov3-opencv.cpp
   VisualizeDangerIntersection.hpp
   VisualizeDangerIntersection.cpp)

message(STATUS OpenCV_LIBS=${OpenCV_LIBS})

target_link_libraries(${PROJECT_NAME}
   ${OpenCV_LIBS})

if(PLAYGROUND_NEEDED)
add_subdirectory(playground)
endif()