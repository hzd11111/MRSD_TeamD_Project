import time
import subprocess
import sys
sys.path.insert(0, "/home/mayank/Mayank/MRSD_TeamD_Project")
sys.path.insert(0, "/home/mayank/Carla/CARLA_0.9.8/PythonAPI/carla/")
# sys.path.insert(0, "/home/mayank/Carla/carla/Dist/0.9.7.4/PythonAPI/carla/dist/")
import rospy
import copy
import random
import threading

sys.path.append("/home/mayank/Carla/CARLA_0.9.8/PythonAPI/carla/dist/carla-0.9.8-py3.6-linux-x86_64.egg")

import carla


import agents.navigation.controller
import numpy as np

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from carla_handler import CarlaHandler
sys.path.insert(0, "/home/mayank/Mayank/GRASP_ws/src/MRSD_TeamD_Project/carla_bridge/scripts")
from grasp_controller import GRASPPIDController
sys.path.insert(0, '/opt/ros/kinetic/lib/python2.7/dist-packages')

from std_msgs.msg import String
from grasp_path_planner.msg import LanePoint
from grasp_path_planner.msg import LanePoint
from grasp_path_planner.msg import Lane
from grasp_path_planner.msg import VehicleState
from grasp_path_planner.msg import RewardInfo
from grasp_path_planner.msg import EnvironmentState
from grasp_path_planner.msg import PathPlan

client = carla.Client('localhost', 2000)
client.set_timeout(2.0)

carla_handler = CarlaHandler(client)
world_map = client.get_world().get_map()
current_road_ID = 10

for actor in client.get_world().get_actors().filter('walker.*'):
	actor_nearest_waypoint = world_map.get_waypoint(actor.get_location(), project_to_road=True)
	print(actor.get_velocity(), actor.bounding_box)
	break

actor_nearest_waypoint = random.choice(world_map.get_spawn_points())
# actor_nearest_waypoint.transform.location.z = actor_nearest_waypoint.transform.location.z + 2

carla_handler.spawn_vehicle(spawn_point=actor_nearest_waypoint)