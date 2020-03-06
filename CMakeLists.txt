cmake_minimum_required(VERSION 2.8.3)

project(grasp_top)
add_subdirectory(ros_message_test)
add_subdirectory(carla_bridge)
add_subdirectory(grasp_path_planner)
add_subdirectory(rl_node)
add_subdirectory(simple_simulator)
add_subdirectory(sb_rl_node)