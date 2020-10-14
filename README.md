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

TO RUN:

1. GoTo /Carla/CARLA_0.9.8/CarlaUE4/Config/DefaultEngine.ini
2. Replace Town0* with Town04 in the .ini file. Use replace all to be sure.
3. Run ./CarlaUE4.sh to start CARLA.
4. Make sure ROSMASTER is running
5. GoTo carla_bridge/scripts/
6. Run carla_node.py
7. Wait till you see ros spin on the terminal.
8. Remove/comment everything from waypoint_follow.roslaunch in base folder. Keep full_path_planner only.
9. roslaunch waypoint_follow.roslaunch
10. Done.


## Things to do
* Replace dummy objects in VehicleInFront and VehicleBehind
* Use common dummy values [x = 1000, y = 1000, theta = 0 , speed = -1]
* Shift functionality to extract front and back vehicles to current_lane utility
* sync settings and options
* Fix convertdecision based on the new enum in rl_manager.py
* Remove Pedestrian from scenario manager and settings
* Add init after new
* Add right_most_lane into env_desc
