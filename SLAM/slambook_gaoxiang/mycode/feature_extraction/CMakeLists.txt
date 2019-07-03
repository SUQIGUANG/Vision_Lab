cmake_minimum_required(VERSION 3.13)
project(feature_extraction)

set(CMAKE_CXX_STANDARD 11)

set(OpenCV_DIR /opt/ros/kinetic/share/OpenCV-3.3.1-dev)
find_package( OpenCV 3 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(feature_extraction main.cpp)
target_link_libraries( feature_extraction ${OpenCV_LIBS} )