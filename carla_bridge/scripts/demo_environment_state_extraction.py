#!/usr/bin/env python

""" Description of steps to follow to obtain environment state from carla, and publish the information as a ROS message. """

__author__ = "Mayank Singal"
__maintainer__ = "Mayank Singal"
__email__ = "mayanksi@andrew.cmu.edu"
__version__ = "0.1"

#####################################################################################

import threading
import time
import subprocess
import sys
import sys
sys.path.insert(0, "/home/mayank/Mayank/MRSD_TeamD_Project")
sys.path.insert(0, "/home/mayank/Carla/carla/PythonAPI/carla/")


import carla
import agents.navigation.controller
import rospy
import copy
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


#####################################################################################

NODE_NAME = 'carla_bridge'
SIM_TOPIC_NAME = 'environment_state'
PATH_PLAN_TOPIC_NAME = 'path_plan'

class CarlaManager:
	def __init__(self):
		self.env_pub = None
		self.path_sub = None
		#id
		self.id = 0
		self.id_waiting = 0
		## CARLA Deps ##
		self.client = None
		self.carla_handler = None
		self.ego_vehicle = None
		self.vehicle_controller = None
		self.lock = threading.Lock()
		self.env_msg = False

	def getVehicleState(self, actor):

		if(actor == None):
			return None

		vehicle = VehicleState()
		vehicle.vehicle_location.x = actor.get_transform().location.x
		vehicle.vehicle_location.y = actor.get_transform().location.y
		vehicle.vehicle_location.theta = actor.get_transform().rotation.yaw * np.pi / 180 #CHECK : Changed this to radians. 
		vehicle.vehicle_speed = np.sqrt(actor.get_velocity().x**2 + actor.get_velocity().y**2 + actor.get_velocity().z**2) 
		vehicle.length = 5
		vehicle.width = 2

		return vehicle

	def getLanePoints(self, waypoints, flip=False):

		lane_cur = Lane()
		lane_points = []
		for i,waypoint in enumerate(waypoints):
			lane_point = LanePoint()
			lane_point.pose.y = waypoint.transform.location.y
			lane_point.pose.x = waypoint.transform.location.x
			lane_point.pose.theta = waypoint.transform.rotation.yaw * np.pi / 180# CHECK : Changed this to radians.
			if(flip == True):
				lane_point.pose.theta += 0#np.pi
				#lane_point.pose.theta = lane_point.pose.theta % (2*np.pi)
			lane_point.width = 3 # TODO
			lane_points.append(lane_point)
		lane_cur.lane = lane_points

		return lane_cur



	def pathCallback(self, data):
		self.lock.acquire()
		# sample path load code ToDo: Delete
		tracking_pose = copy.copy(data.tracking_pose)
		tracking_speed = copy.copy(data.tracking_speed)
		reset_sim = copy.copy(data.reset_sim)
		
		if not data.id == self.id_waiting:
			self.lock.release()
			return
		# print("New Path Plan")
		# print("Tracking Pose:",tracking_pose.x,",",tracking_pose.y,",",tracking_pose.theta)
		# print("Tracking Speed:", tracking_speed)
		# print("Reset Sim:", reset_sim)


		# current_location = carla.Location()
		# current_rotation = carla.Rotation()

		# current_location.x = tracking_pose.x
		# current_location.y = tracking_pose.y
		# current_location
		# current_rotation.yaw = tracking_pose.theta

		# required_transform = carla.Transform()
		# required_transform.location = current_location
		# required_transform.rotation = current_rotation


		# # required_waypoint = Waypoint()
		# # required_waypoint.transform = required_transform

		# end_waypoint = self.carla_handler.world_map.get_waypoint(current_location)

		# ToDo: load the path data

		# ToDo: call the controller to follow the path

		#nearest_waypoint = self.carla_handler.world_map.get_waypoint(self.ego_vehicle.get_location(), project_to_road=True)
		#nearest_waypoint.transform = required_transform
		
		self.ego_vehicle.apply_control(self.vehicle_controller.run_step(tracking_speed, tracking_pose))
		# print("Control Called")

		tracking_loc = carla.Location(x=tracking_pose.x, y=tracking_pose.y, z=self.ego_vehicle.get_location().z)
		self.carla_handler.world.debug.draw_string(tracking_loc, 'O', draw_shadow=False,
			                               color=carla.Color(r=255, g=0, b=0), life_time=20,
			                               persistent_lines=True)
		
		flag = 0
		while(flag == 0):
			try:
				self.carla_handler.world.tick()
				flag = 1
				print("Passed Tick....................................................................................")
			except:
				continue
			
		#self.carla_handler.world.wait_for_tick()
		# ToDo: make CARLA step for one frame and reset if necessary

		# ToDo: extract environment data from CARLA
		########################################################################################3
		#print("Time End-4:", time.time())
		state_information = self.carla_handler.get_state_information(self.ego_vehicle)
		current_lane_waypoints, left_lane_waypoints, right_lane_waypoints, front_vehicle, rear_vehicle, actors_in_current_lane, actors_in_left_lane, actors_in_right_lane = state_information

		# Current Lane
		lane_cur = self.getLanePoints(current_lane_waypoints)
		
		# Left Lane
		lane_left = self.getLanePoints(left_lane_waypoints, flip=False) ##TODO Check this. Reversed 

		
		
		# Right Lane
		lane_right = self.getLanePoints(right_lane_waypoints)
		

		# TODO : Can wrap this as a function member of the class. //Done  
		# Ego vehicle	
		vehicle_ego = self.getVehicleState(self.ego_vehicle)

		#print("Location:", vehicle_ego.vehicle_location.x, vehicle_ego.vehicle_location.y)


		#print("Time End-3:", time.time())
		
		# Front vehicle	
		if(front_vehicle == None):
			vehicle_front = vehicle_ego
		else:
			vehicle_front = self.getVehicleState(front_vehicle)
		
		# Rear vehicle
		if(rear_vehicle == None):
			vehicle_rear = vehicle_ego
		else:	
			vehicle_rear = self.getVehicleState(rear_vehicle)
	
		#print("Time End-2:", time.time())
		# sample enviroment state
		env_state = EnvironmentState()
		env_state.cur_vehicle_state = vehicle_ego
		env_state.front_vehicle_state = vehicle_front
		env_state.back_vehicle_state = vehicle_rear
		env_state.current_lane = lane_cur
		env_state.next_lane = lane_left
		env_state.adjacent_lane_vehicles = [self.getVehicleState(actor) for actor in actors_in_left_lane] #TODO : Only considering left lane for now. Need to make this more general 
		env_state.max_num_vehicles = 2
		env_state.speed_limit = 40
		env_state.id = self.id
		#print("Publishing Id:",self.id)
		self.id_waiting = self.id
		self.id += 1
		if self.id > 100000:
			self.id = 0
		#print("Time End-1:", time.time())
		rate = rospy.Rate(100)
		# publish environment state
		self.env_msg = env_state
		self.env_pub.publish(env_state)
		np.save('/home/mayank/Mayank/test_env_message2.npy', env_state, allow_pickle=True)
		time.sleep(5)
		#rate.sleep()#ToDo: Delete this line	
		####
		self.lock.release()
		####

	def publishFunc(self):
		rate = rospy.Rate(10)
		while not rospy.is_shutdown():
			self.lock.acquire()
			if self.env_msg:
				self.env_pub.publish(self.env_msg)
			self.lock.release()
			rate.sleep()

	def initialize(self):
		# initialize node
		rospy.init_node(NODE_NAME, anonymous=True)
	
		# initialize publisher
		self.env_pub = rospy.Publisher(SIM_TOPIC_NAME, EnvironmentState, queue_size = 1000)
		
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
	
		## Update World Information
		# settings = self.carla_handler.world.get_settings()
		# settings.synchronous_mode = True
		# settings.fixed_delta_seconds = 0.2
		# self.carla_handler.world.apply_settings(settings)



 
 		
		# Spawn ego vehicle on road 
		filtered_waypoints = self.carla_handler.filter_waypoints(self.carla_handler.get_waypoints(1), road_id=12)
		spawn_point = filtered_waypoints[4].transform # Select random point from filtered waypoint list #TODO Initialization Scheme Design
		spawn_point.location.z = spawn_point.location.z + 1 # To avoid collision during spawn
		self.ego_vehicle, ego_vehicle_ID = self.carla_handler.spawn_vehicle(spawn_point=spawn_point)

		self.vehicle_controller = GRASPPIDController(self.ego_vehicle, args_lateral = {'K_P': 0.1, 'K_D': 0.0, 'K_I': 0}, args_longitudinal = {'K_P': 0.5, 'K_D': 0.0, 'K_I': 0.0})

		time.sleep(3)
		rate = rospy.Rate(2000)
		#rate.sleep()#ToDo: Delete this line	

		state_information = self.carla_handler.get_state_information(self.ego_vehicle)
		current_lane_waypoints, left_lane_waypoints, right_lane_waypoints, front_vehicle, rear_vehicle, actors_in_current_lane, actors_in_left_lane, actors_in_right_lane = state_information

		#print("Spawn road ID:", self.carla_handler.world_map.get_waypoint(ego_vehicle.get_location(), project_to_road=True).road_id)
		
		##############################################################################################################
		# publish the first frame ToDo: change it to an actual one
		

		#print("Spawn road ID:", self.carla_handler.world_map.get_waypoint(ego_vehicle.get_location(), project_to_road=True).road_id)
		# Current Lane
		lane_cur = self.getLanePoints(current_lane_waypoints)
		
		# Left Lane
		lane_left = self.getLanePoints(left_lane_waypoints, flip=True)
		#self.carla_handler.draw_arrow(left_lane_waypoints)
		
		# Right Lane
		lane_right = self.getLanePoints(right_lane_waypoints)
		

		# TODO : Can wrap this as a function member of the class. //Done  
		# Ego vehicle	
		vehicle_ego = self.getVehicleState(self.ego_vehicle);
		
		# Front vehicle	
		if(front_vehicle == None):
			vehicle_front = vehicle_ego
		else:
			vehicle_front = self.getVehicleState(front_vehicle)
		
		# Rear vehicle
		if(rear_vehicle == None):
			vehicle_rear = vehicle_ego
		else:	
			vehicle_rear = self.getVehicleState(rear_vehicle)

			
		# sample enviroment state
		env_state = EnvironmentState()
		env_state.cur_vehicle_state = vehicle_ego
		env_state.front_vehicle_state = vehicle_front
		env_state.back_vehicle_state = vehicle_rear
		env_state.adjacent_lane_vehicles = [self.getVehicleState(actor) for actor in actors_in_left_lane] #TODO : Only considering left lane for now. Need to make this more general 
		env_state.current_lane = lane_cur
		env_state.next_lane = lane_left
		env_state.max_num_vehicles = 2
		env_state.speed_limit = 40
		env_state.id = self.id
		#print("Publishing Id:",self.id)
		self.id_waiting = self.id
		self.id += 1
		if self.id > 100000:
			self.id = 0
		#print("Time End-1:", time.time())
		rate = rospy.Rate(100)
		# publish environment state
		self.env_msg = env_state
		self.env_pub.publish(env_state)
		#rate.sleep()#ToDo: Delete this line	
		#print("Publishing Env")

		####
		#self.carla_handler.world.tick()
		####
		
	def spin(self):
		print("Start Ros Spin")	
		# spin
		rospy.spin()
	
if __name__ == '__main__':
	try:
		carla_manager = CarlaManager()
		carla_manager.initialize()
		pub_thread = threading.Thread(target=carla_manager.publishFunc)
		carla_manager.spin()
	except rospy.ROSInterruptException:
		pass
	
 











