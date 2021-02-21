cmake_minimum_required(VERSION 2.8.3)
project(guess_ros)

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# To uncomment in the futute
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -DNDEBUG -fPIC --std=c++14 -Wall")

if(NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES arm)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

set(CMAKE_BUILD_TYPE Release)

message( "${PROJECT_NAME}|compiling with these flags: ${CMAKE_CXX_FLAGS}")


## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  geometry_msgs
  cv_bridge
  image_transport
  compressed_image_transport
# guess
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

# Find Eigen3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Find OpenCV
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

#YAML-CPP
find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIR})


###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS src
  LIBRARIES 
  CATKIN_DEPENDS tf image_transport std_msgs sensor_msgs geometry_msgs cv_bridge
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/src
)


add_subdirectory(${PROJECT_SOURCE_DIR}/src)

