cmake_minimum_required(VERSION 3.13)
project(BA_3d2d)

set(CMAKE_CXX_STANDARD 11)

# 添加cmake模块以使用g2o
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

set(OpenCV_DIR /opt/ros/kinetic/share/OpenCV-3.3.1-dev)
find_package( OpenCV 3 REQUIRED )
# find_package( OpenCV REQUIRED ) # use this if in OpenCV2
find_package( G2O REQUIRED )
find_package( CSparse REQUIRED )

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        ${CSPARSE_INCLUDE_DIR}
        "/usr/include/eigen3/"
)

add_executable(BA_3d2d main.cpp)
target_link_libraries( BA_3d2d
        ${OpenCV_LIBS}
        ${CSPARSE_LIBRARY}
        g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension
        )