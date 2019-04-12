# cmake needs this line
cmake_minimum_required(VERSION 2.8)

# Define project name
project(libfreenect_read_data)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)

# FIND_PACKAGE(PkgConfig REQUIRED)

# OpenCV
set(OpenCV_DIR /opt/ros/kinetic/share/OpenCV-3.3.1-dev)
FIND_PACKAGE(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

# freenect2
FIND_PACKAGE(freenect2 REQUIRED)
include_directories( ${freenect2_INCLUDE_DIRS} )
include_directories("/usr/include/libusb-1.0/")

# Eigen3
FIND_PACKAGE(Eigen3 REQUIRED)
include_directories("/usr/include/eigen3")

# PCL
FIND_PACKAGE(PCL REQUIRED)
include_directories( ${PCL_INCLUDE_DIRS} )
link_directories(${PCL_LIBRARY_DIRS})

# Boost
FIND_PACKAGE(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

# VTK
FIND_PACKAGE(VTK REQUIRED)
include_directories(${VTK_INCLUDE_DIRS})
link_directories(${VTK_LIBRARY_DIRS})


# Declare the executable target built from your sources
add_executable(libfreenect_read_data libfreenect_read_data.cpp)


# Link your application with OpenCV libraries
target_link_libraries(libfreenect_read_data
                      "/usr/lib/x86_64-linux-gnu/libpcl_visualization.so.1.7.2"
                      libboost_system.so libboost_filesystem.so
                      ${VTK_LIBRARIES}
                      ${freenect2_LIBRARIES}
                      ${OpenCV_LIBS})
