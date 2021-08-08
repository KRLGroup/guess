cmake_minimum_required(VERSION 3.0.2)
project(guess_simulation)

find_package(catkin REQUIRED gazebo_ros lms1xx)

catkin_package()

install(
  DIRECTORY worlds maps urdf models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)