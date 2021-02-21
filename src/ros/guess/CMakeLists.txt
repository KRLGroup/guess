cmake_minimum_required(VERSION 2.8.3)
project(guess)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=gnu++11")
set(CMAKE_BUILD_TYPE Release)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories(guess_lib)

add_subdirectory(guess_lib)
add_subdirectory(guess_app)
