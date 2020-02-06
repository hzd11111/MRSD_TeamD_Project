#!/usr/bin/env python

""" Description of steps to follow to obtain environment state from carla, and publish the information as a ROS message. """

__author__ = "Mayank Singal"
__maintainer__ = "Mayank Singal"
__email__ = "mayanksi@andrew.cmu.edu"
__version__ = "0.1"

#####################################################################################

import time
import subprocess
import sys
import sys
sys.path.insert(0, "/home/mayank/Mayank/MRSD_TeamD_Project")


import carla
import rospy
import copy

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from carla_handler import CarlaHandler
sys.path.insert(0, '/opt/ros/kinetic/lib/python2.7/dist-packages')

from std_msgs.msg import String
from grasp_path_planner.msg import LanePoint
from grasp_path_planner.msg import LanePoint
from grasp_path_planner.msg import Lane
from grasp_path_planner.msg import VehicleState
from grasp_path_planner.msg import RewardInfo
from grasp_path_planner.msg import EnvironmentState
from grasp_path_planner.msg import PathPlan


#####################################################################################

NODE_NAME = 'carla_bridge'
SIM_TOPIC_NAME = 'environment_state'
PATH_PLAN_TOPIC_NAME = 'path_plan'

class CarlaManager:
	def __init__(self):
		self.env_pub = None
		self.path_sub = None

		## CARLA Deps ##
		self.client = None
		self.carla_handler = None

	def getVehicleState(self, actor):
		vehicle = VehicleState()
		vehicle.vehicle_location.x = actor.get_transform().location.x
		vehicle.vehicle_location.y = actor.get_transform().location.y
		vehicle.vehicle_location.theta = actor.get_transform().rotation.yaw #TODO : This has roll, pith and yaw. Need 2 more variables?
		vehicle.vehicle_speed = 3 # TODO : This has x, y, and z components. Need 2 more variables. 
		vehicle.length = 5
		vehicle.width = 2

		return vehicle

	def getLanePoints(self, waypoints):

		lane_cur = Lane()
		lane_points = []
		for waypoint in waypoints:
			lane_point = LanePoint()
			lane_point.pose.y = waypoint.transform.location.y
			lane_point.pose.x = waypoint.transform.location.x
			lane_point.pose.theta = waypoint.transform.rotation.yaw # TODO : this has roll, pitch and yaw
			lane_point.width = 3 # TODO
			lane_points.append(lane_point)
		lane_cur.lane = lane_points

		return lane_cur



	def pathCallback(self, data):
		# sample path load code ToDo: Delete
		tracking_pose = data.tracking_pose
		tracking_speed = data.tracking_speed
		reset_sim = data.reset_sim
		
		print("New Path Plan")
		print("Tracking Pose:",tracking_pose.x,",",tracking_pose.y,",",tracking_pose.theta)
		print("Tracking Speed:", tracking_speed)
		print("Reset Sim:", reset_sim)

		# ToDo: load the path data

		# ToDo: call the controller to follow the path

		# ToDo: make CARLA step for one frame and reset if necessary

		# ToDo: extract environment data from CARLA

		# ToDo: fillin the ros msg with environment data

		# sample code for pulbishing the msg ToDo:Delete these when the above are done
		# current Lane
		lane_cur = Lane()
		lane_points = []
		for i in range(100):
			lane_point = LanePoint()
			lane_point.pose.y = 0
			lane_point.pose.x = i/10.
			lane_point.pose.theta = 0
			lane_point.width = 3
			lane_points.append(lane_point)
		lane_cur.lane = lane_points
		# next Lane
		lane_next = Lane()
		lane_points_next = []
		for i in range(100):
			lane_point = LanePoint()
			lane_point.pose.y = 3
			lane_point.pose.x = i/10.
			lane_point.pose.theta = 0
			lane_point.width = 3
			lane_points_next.append(lane_point)
		lane_next.lane = lane_points_next
	
		# current vehicle	
		vehicle = VehicleState()
		vehicle.vehicle_location.x = 2
		vehicle.vehicle_location.y = 0
		vehicle.vehicle_location.theta = 0
		vehicle.vehicle_speed = 10
		vehicle.length = 5
		vehicle.width = 2

		# sample enviroment state
		env_state = EnvironmentState()
		env_state.cur_vehicle_state = copy.copy(vehicle)
		env_state.front_vehicle_state = copy.copy(vehicle)
		env_state.back_vehicle_state = copy.copy(vehicle)
		env_state.adjacent_lane_vehicles = [copy.copy(vehicle), copy.copy(vehicle)]
		env_state.max_num_vehicles = 2
		env_state.speed_limit = 40
		rate = rospy.Rate(2000)

		# publish environment state
		self.env_pub.publish(env_state)
		rate.sleep()#ToDo: Delete this line	

	def initialize(self):
		# initialize node
		rospy.init_node(NODE_NAME, anonymous=True)
	
		# initialize publisher
		self.env_pub = rospy.Publisher(SIM_TOPIC_NAME, EnvironmentState, queue_size = 10)
		
		# initlize subscriber
		self.path_sub = rospy.Subscriber(PATH_PLAN_TOPIC_NAME, PathPlan, self.pathCallback)

		################################## Initialize environment in CARLA #################################################
		##
		##	1. Spawn n = 70 actors in the map.
		##	2. Spawn ego vehicle.
		##
		####

		# Spawn some vehicles in the environment using spawn_vehicles.py, if environment doesn't have vehicles already.
		# subprocess.Popen(["python", "/home/mayank/Mayank/GRASP_ws/src/MRSD_TeamD_Project/carla_bridge/scripts", "-n" , "70"])

		# Start Client. Make sure Carla server is running before starting.

		client = carla.Client('localhost', 2000)
		client.set_timeout(2.0)
		print("Connection to CARLA server established!")

		# Create a CarlaHandler object. CarlaHandler provides some cutom built APIs for the Carla Server.

		self.carla_handler = CarlaHandler(client)
 
		# Spawn ego vehicle on road 
		filtered_waypoints = self.carla_handler.filter_waypoints(self.carla_handler.get_waypoints(), road_id=12)
		spawn_point = filtered_waypoints[100].transform # Select random point from filtered waypoint list #TODO Initialization Scheme Design
		spawn_point.location.z = spawn_point.location.z + 2 # To avoid collision during spawn
		ego_vehicle, ego_vehicle_ID = self.carla_handler.spawn_vehicle(spawn_point=spawn_point)

		time.sleep(3)
		rate = rospy.Rate(10000)
		rate.sleep()#ToDo: Delete this line	

		state_information = self.carla_handler.get_state_information(ego_vehicle)
		current_lane_waypoints, left_lane_waypoints, right_lane_waypoints, front_vehicle, rear_vehicle, actors_in_current_lane, actors_in_left_lane, actors_in_right_lane = state_information

		print("Spawn road ID:", self.carla_handler.world_map.get_waypoint(ego_vehicle.get_location(), project_to_road=True).road_id)
		
		##############################################################################################################
		# publish the first frame ToDo: change it to an actual one
		

		print("Spawn road ID:", self.carla_handler.world_map.get_waypoint(ego_vehicle.get_location(), project_to_road=True).road_id)
		# Current Lane
		lane_cur = self.getLanePoints(current_lane_waypoints)
		
		# Left Lane
		lane_left = self.getLanePoints(left_lane_waypoints)
		
		# Right Lane
		lane_right = self.getLanePoints(right_lane_waypoints)
		

		# TODO : Can wrap this as a function member of the class. //Done  
		# Ego vehicle	
		vehicle_ego = self.getVehicleState(ego_vehicle);
		
		# Front vehicle	
		vehicle_front = self.getVehicleState(front_vehicle)
		
		# Rear vehicle	
		vehicle_rear = self.getVehicleState(rear_vehicle)
	
		# sample enviroment state
		env_state = EnvironmentState()
		env_state.cur_vehicle_state = copy.copy(vehicle_ego)
		env_state.front_vehicle_state = copy.copy(vehicle_front)
		env_state.back_vehicle_state = copy.copy(vehicle_rear)
		env_state.adjacent_lane_vehicles = [copy.copy(self.getVehicleState(actor)) for actor in actors_in_left_lane] #TODO : Only considering left lane for now. Need to make this more general 
		env_state.max_num_vehicles = 2
		env_state.speed_limit = 40
		self.env_pub.publish(env_state)
		
		print("Start Ros Spin")	
		# spin
		rospy.spin()
	
if __name__ == '__main__':
	try:
		carla_manager = CarlaManager()
		carla_manager.initialize()
	except rospy.ROSInterruptException:
		pass
	
 











