# MRSD_TeamD_Project

After building the system on your machine:

1. Start the path_planner_node
2. Start the rl_node
3. Start Carla
4. Start demo_environment_state_extraction.py

Or add the following to a roslaunch file (for eg. waypoint_follow.launch). Make sure to start the CARLA simulator before running the launch file. 

<launch>
	<node name = "path_planning_node" pkg="grasp_path_planner" type="path_planner_node" output="screen"/>
	<node name = "rl_node" pkg="rl_node" type="rl_node.py" />
	<node name = "demo_node" pkg="carla_bridge" type="demo_environment_state_extraction.py" output="screen"/>
</launch>

This should be added to a folder called 'launch' in the workspace folder. (The workspace folder will now have build, devel, launch and src folders. ). An example launch file is added in the root folder of the project.

