add_executable(dumper_node
  dumper_node.cpp
)

target_link_libraries(dumper_node
  ${catkin_LIBRARIES}
)

add_dependencies(dumper_node
  ${catkin_EXPORTED_TARGETS}
)

