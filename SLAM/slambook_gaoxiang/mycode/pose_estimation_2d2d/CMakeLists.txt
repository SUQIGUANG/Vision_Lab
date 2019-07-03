cmake_minimum_required(VERSION 3.13)
project(pose_estimation_2d2d)

set(CMAKE_CXX_STANDARD 11)

set(OpenCV_DIR /opt/ros/kinetic/share/OpenCV-3.3.1-dev)
find_package( OpenCV 3 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(pose_estimation_2d2d main.cpp)
target_link_libraries( pose_estimation_2d2d ${OpenCV_LIBS} )