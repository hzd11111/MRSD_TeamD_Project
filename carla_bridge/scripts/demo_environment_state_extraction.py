#!/usr/bin/env python

""" Description of steps to follow to obtain environment state from carla, and publish the information as a ROS message. """

__author__ = "Mayank Singal"
__maintainer__ = "Mayank Singal"
__email__ = "mayanksi@andrew.cmu.edu"
__version__ = "0.1"

#####################################################################################

import threading
import os
import time
import subprocess
import sys
import sys
from argparse import RawTextHelpFormatter
sys.path.insert(0, "/home/mayank/Mayank/MRSD_TeamD_Project")
sys.path.insert(0, "/home/mayank/Carla/carla/PythonAPI/carla/")

os.environ["ROOT_SCENARIO_RUNNER"] = "/home/mayank/Mayank/SRunner"


import carla
import agents.navigation.controller
import rospy
import copy
import numpy as np
import argparse
VERSION = 0.6

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from carla_handler import CarlaHandler
sys.path.insert(0, "/home/mayank/Mayank/GRASP_ws/src/MRSD_TeamD_Project/carla_bridge/scripts")
from grasp_controller import GRASPPIDController
sys.path.insert(0, '/opt/ros/kinetic/lib/python2.7/dist-packages')

sys.path.insert(0, '/home/mayank/Mayank/SRunner')

from std_msgs.msg import String
from grasp_path_planner.msg import LanePoint
from grasp_path_planner.msg import LanePoint
from grasp_path_planner.msg import Lane
from grasp_path_planner.msg import VehicleState
from grasp_path_planner.msg import RewardInfo
from grasp_path_planner.msg import EnvironmentState
from grasp_path_planner.msg import PathPlan

from scenario_runner import ScenarioRunner





#####################################################################################

NODE_NAME = 'carla_bridge'
SIM_TOPIC_NAME = 'environment_state'
PATH_PLAN_TOPIC_NAME = 'path_plan'

class CarlaManager:
	def __init__(self, SCENARIORUNNER, SCENARIOMANAGER):
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

		self.SCENARIORUNNER = SCENARIORUNNER
		self.SCENARIOMANAGER = SCENARIOMANAGER

		self.lock = threading.Lock()
		self.env_msg = False
  
	def __del__(self):
     
		for actor in self.client.get_world().get_actors().filter('vehicle.*'): 
				actor.destroy()
		print("All actors destroyed..\n") 


	def resetEnv(self):

		del self.carla_handler
		self.SCENARIORUNNER._cleanup(True)
		self.initialize()

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

	def getLanePoints(self, waypoints, width, flip=False):

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
			lane_point.width = width # TODO
			lane_points.append(lane_point)
		lane_cur.lane = lane_points

		return lane_cur



	def pathCallback(self, data):
		self.lock.acquire()
		# sample path load code ToDo: Delete
		tracking_pose = copy.copy(data.tracking_pose)
		tracking_speed = copy.copy(data.tracking_speed)
		reset_sim = copy.copy(data.reset_sim)

		print("Reset Sim:", reset_sim)
		if(reset_sim == 1):
			env_state = EnvironmentState()
			env_state.id = self.id

			self.id_waiting = self.id
			self.id += 1
			if self.id > 100000:
				self.id = 0
			self.env_msg = env_state
			self.env_pub.publish(env_state)
			self.lock.release()
			self.resetEnv()
		
		if not data.id == self.id_waiting:
			self.lock.release()
			return
		
		self.ego_vehicle.apply_control(self.vehicle_controller.run_step(tracking_speed, tracking_pose))

		tracking_loc = carla.Location(x=tracking_pose.x, y=tracking_pose.y, z=self.ego_vehicle.get_location().z)
		self.carla_handler.world.debug.draw_string(tracking_loc, 'O', draw_shadow=False,
			                               color=carla.Color(r=255, g=0, b=0), life_time=20,
			                               persistent_lines=True)
		
		flag = 0
		while(flag == 0):
			try:
				self.carla_handler.world.tick()
				flag = 1
			except:
				print("Missed Tick....................................................................................")
				continue
			

		# ToDo: extract environment data from CARLA
		########################################################################################3
		#print("Time End-4:", time.time())
		state_information = self.carla_handler.get_state_information(self.ego_vehicle)
		current_lane_waypoints, left_lane_waypoints, right_lane_waypoints, front_vehicle, rear_vehicle, actors_in_current_lane, actors_in_left_lane, actors_in_right_lane = state_information

		#point1 = current_lane_waypoints[0].transform.location
		#point2 = left_lane_waypoints[0].transform.location
		#width = np.sqrt((point1.x-point2.x)**2 + (point1.y-point2.y)**2)
		width = 2
		# Current Lane
		lane_cur = self.getLanePoints(current_lane_waypoints, width)
		
		# Left Lane
		lane_left = self.getLanePoints(left_lane_waypoints[::-1], width, flip=False) ##TODO Check this. Reversed


		# Right Lane
		lane_right = self.getLanePoints(right_lane_waypoints, width)
		

		# TODO : Can wrap this as a function member of the class. //Done  
		# Ego vehicle	
		vehicle_ego = self.getVehicleState(self.ego_vehicle)
		
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
		env_state.speed_limit = 35
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
  
		print("EGO VEHICLES:", self.SCENARIORUNNER.ego_vehicles)
  

		################################## Initialize environment in CARLA #################################################
		##
		##	1. Spawn n = 70 actors in the map.
		##	2. Spawn ego vehicle.
		##
		####

		# Spawn some vehicles in the environment using spawn_vehicles.py, if environment doesn't have vehicles already.
		# subprocess.Popen(["python", "/home/mayank/Mayank/GRASP_ws/src/MRSD_TeamD_Project/carla_bridge/scripts", "-n" , "70"])

		# Start Client. Make sure Carla server is running before starting.

		self.client = carla.Client('localhost', 2000)
		self.client.set_timeout(2.0)
		print("Connection to CARLA server established!")

		# Create a CarlaHandler object. CarlaHandler provides some cutom built APIs for the Carla Server.

		self.carla_handler = CarlaHandler(self.client)

		# Connect Ego Vehicle 
		self.ego_vehicle = self.SCENARIORUNNER.ego_vehicles[0]
		self.ego_vehicle_ID = self.ego_vehicle.id
		
		#filtered_waypoints = self.carla_handler.filter_waypoints(self.carla_handler.get_waypoints(1), road_id=)
		#spawn_point = filtered_waypoints[4].transform # Select random point from filtered waypoint list #TODO Initialization Scheme Design
		#spawn_point.location.z = spawn_point.location.z + 1 # To avoid collision during spawn
		#self.ego_vehicle, ego_vehicle_ID = self.carla_handler.spawn_vehicle(spawn_point=spawn_point)
		#state_information = self.carla_handler.get_state_information(self.ego_vehicle)


		self.vehicle_controller = GRASPPIDController(self.ego_vehicle, args_lateral = {'K_P': 0.015, 'K_D': 0.00, 'K_I': 0.03}, args_longitudinal = {'K_P': 0.3, 'K_D': 0.1, 'K_I': 0.05})

		# time.sleep(2)

		rate = rospy.Rate(2000)
		#rate.sleep()#ToDo: Delete this line	

		state_information = self.carla_handler.get_state_information(self.ego_vehicle)
		current_lane_waypoints, left_lane_waypoints, right_lane_waypoints, front_vehicle, rear_vehicle, actors_in_current_lane, actors_in_left_lane, actors_in_right_lane = state_information

		#print("Spawn road ID:", self.carla_handler.world_map.get_waypoint(ego_vehicle.get_location(), project_to_road=True).road_id)
		
		##############################################################################################################
		# publish the first frame ToDo: change it to an actual one
		

		#print("Spawn road ID:", self.carla_handler.world_map.get_waypoint(ego_vehicle.get_location(), project_to_road=True).road_id)

		point1 = current_lane_waypoints[0].transform.location
		point2 = left_lane_waypoints[0].transform.location
		width = np.sqrt((point1.x-point2.x)**2 + (point1.y-point2.y)**2)

		# Current Lane
		lane_cur = self.getLanePoints(current_lane_waypoints, width)
		
		# Left Lane
		lane_left = self.getLanePoints(left_lane_waypoints[::-1], width, flip=True)
		#self.carla_handler.draw_arrow(left_lane_waypoints)
		
		# Right Lane
		lane_right = self.getLanePoints(right_lane_waypoints, width)
		

		# TODO : Can wrap this as a function member of the class. //Done  
		# Ego vehicle	
		vehicle_ego = self.getVehicleState(self.ego_vehicle)
		
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

    DESCRIPTION = ("CARLA Scenario Runner: Setup, Run and Evaluate scenarios using CARLA\n"
                   "Current version: " + str(VERSION))

    PARSER = argparse.ArgumentParser(description=DESCRIPTION,
                                     formatter_class=RawTextHelpFormatter)
    PARSER.add_argument('--epoches', default=0, help='Number of epoch executions')
    PARSER.add_argument('--host', default='127.0.0.1',
                        help='IP of the host server (default: localhost)')
    PARSER.add_argument('--port', default='2000',
                        help='TCP port to listen to (default: 2000)')
    PARSER.add_argument('--debug', action="store_true", help='Run with debug output')
    PARSER.add_argument('--output', action="store_true", help='Provide results on stdout')
    PARSER.add_argument('--file', action="store_true", help='Write results into a txt file')
    PARSER.add_argument('--junit', action="store_true", help='Write results into a junit file')
    PARSER.add_argument('--outputDir', default='', help='Directory for output files (default: this directory)')
    PARSER.add_argument('--waitForEgo', action="store_true", help='Connect the scenario to an existing ego vehicle')
    PARSER.add_argument('--configFile', default='', help='Provide an additional scenario configuration file (*.xml)')
    PARSER.add_argument('--additionalScenario', default='', help='Provide additional scenario implementations (*.py)')
    PARSER.add_argument('--reloadWorld', action="store_true", default=False, help='Reload the CARLA world before starting a scenario (default=True)')
    # pylint: disable=line-too-long
    PARSER.add_argument(
        '--scenario',  default='OtherLeadingVehicle_3',  help='Name of the scenario to be executed. Use the preposition \'group:\' to run all scenarios of one class, e.g. ControlLoss or FollowLeadingVehicle')
    PARSER.add_argument('--randomize', action="store_true", help='Scenario parameters are randomized')
    PARSER.add_argument('--repetitions', default=1, help='Number of scenario executions')
    PARSER.add_argument('--list', action="store_true", help='List all supported scenarios and exit')
    PARSER.add_argument(
        '--agent', help="Agent used to execute the scenario (optional). Currently only compatible with route-based scenarios.")
    PARSER.add_argument('--agentConfig', type=str, help="Path to Agent's configuration file", default="")
    PARSER.add_argument('--openscenario', help='Provide an OpenSCENARIO definition')
    PARSER.add_argument(
        '--route', help='Run a route as a scenario, similar to the CARLA AD challenge (input: (route_file,scenario_file,[number of route]))', nargs='+', type=str)
    PARSER.add_argument('--challenge', action="store_true", help='Run in challenge mode')
    PARSER.add_argument('--record', action="store_true",
                        help='Use CARLA recording feature to create a recording of the scenario')
    PARSER.add_argument('-v', '--version', action='version', version='%(prog)s ' + str(VERSION))
    PARSER.add_argument('name', default='temp', help='ROS Param')
    PARSER.add_argument('log', default='temp', help='ROS Param')
    ARGUMENTS = PARSER.parse_args()
    # pylint: enable=line-too-long

    if ARGUMENTS.list:
        print("Currently the following scenarios are supported:")
        print(*ScenarioConfigurationParser.get_list_of_scenarios(ARGUMENTS.configFile), sep='\n')
        sys.exit(0)

    if not ARGUMENTS.scenario and not ARGUMENTS.openscenario and not ARGUMENTS.route:
        print("Please specify either a scenario or use the route mode\n\n")
        PARSER.print_help(sys.stdout)
        sys.exit(0)

    if (ARGUMENTS.route and ARGUMENTS.openscenario) or (ARGUMENTS.route and ARGUMENTS.scenario):
        print("The route mode cannot be used together with a scenario (incl. OpenSCENARIO)'\n\n")
        PARSER.print_help(sys.stdout)
        sys.exit(0)

    if ARGUMENTS.agent and (ARGUMENTS.openscenario or ARGUMENTS.scenario):
        print("Agents are currently only compatible with route scenarios'\n\n")
        PARSER.print_help(sys.stdout)
        sys.exit(0)

    if ARGUMENTS.challenge and (ARGUMENTS.openscenario or ARGUMENTS.scenario):
        print("The challenge mode can only be used with route-based scenarios'\n\n")
        PARSER.print_help(sys.stdout)
        sys.exit(0)

    if ARGUMENTS.route:
        ARGUMENTS.reloadWorld = True

	#TODO
    ARGUMENTS.reloadWorld = False
    #ARGUMENTS.reloadWorld = True
    print("Scenario:", ARGUMENTS.scenario)
    
    SCENARIORUNNER = None
    SCENARIOMANAGER = []
    try:

        SCENARIORUNNER = ScenarioRunner(ARGUMENTS)
        SCENARIORUNNER.run_thread(ARGUMENTS, SCENARIOMANAGER)
        #SCENARIOMANAGER[0].stop_scenario()
        # while True:       
        #    time.sleep(3)
        #    print("checkpoint")
        # print("bye")
        time.sleep(5)
        print("running?:=", SCENARIOMANAGER[0].scenario)
        

        carla_manager = CarlaManager(SCENARIORUNNER, SCENARIOMANAGER)
        carla_manager.initialize()
        pub_thread = threading.Thread(target=carla_manager.publishFunc)
        carla_manager.spin()

    except rospy.ROSInterruptException:
        print("Closing....")
        carla_manager.resetEnv()
        pass

    finally:
        if SCENARIORUNNER is not None:
            del SCENARIORUNNER


	
 











