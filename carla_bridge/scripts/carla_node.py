#!/usr/bin/env python

""" CARLA Node for the GRASP System. """

__author__ = "Mayank Singal, Scott Jin"
__maintainer__ = "Mayank Singal, Scott Jin"
__email__ = "mayanksi@andrew.cmu.edu"
__version__ = "0.1"

#######################################################################################


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

from scenario_manager import CustomScenario
from grasp_path_planner.srv import SimService, SimServiceResponse


#######################################################################################

NODE_NAME = 'carla_bridge'
SIM_TOPIC_NAME = 'environment_state'
PATH_PLAN_TOPIC_NAME = 'path_plan'

SIM_SERVICE_NAME = 'carla_simulator'

#######################################################################################

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
        self.vehicles_list = []
        self.original_lane = None
        self.timestamp = 0
        self.collision_marker = 0
        self.first_run = 1
        self.lane_cur = None
        self.lane_left = None
        self.lane_right = None
        self.collision_sensor = None
        self.tm = None
        
        self.simulation_sync_timestep = 0.1
        
        self.first_frame_generated = False
        self.path_planner_terminate = False
        
        self.action_progress = 0
        self.end_of_action = True
  
    def getVehicleState(self, actor):

        if(actor == None):
            return None

        vehicle = VehicleState()
        vehicle.vehicle_location.x = actor.get_transform().location.x
        vehicle.vehicle_location.y = actor.get_transform().location.y
        vehicle.vehicle_location.theta = actor.get_transform().rotation.yaw * np.pi / 180 #CHECK : Changed this to radians. 
        vehicle.vehicle_speed = np.sqrt(actor.get_velocity().x**2 + actor.get_velocity().y**2) 

        vehicle_bounding_box = actor.bounding_box.extent
        vehicle.length = vehicle_bounding_box.x*2
        vehicle.width = vehicle_bounding_box.y*2

        return vehicle
    
    def getPedestrianState(self, actor):

        ## TODO: Processing them like vehicles for now
        if(actor == None):
            return None

        vehicle = VehicleState()
        vehicle.vehicle_location.x = actor.get_transform().location.x
        vehicle.vehicle_location.y = actor.get_transform().location.y
        vehicle.vehicle_location.theta = actor.get_transform().rotation.yaw * np.pi / 180 #CHECK : Changed this to radians. 
        vehicle.vehicle_speed = np.sqrt(actor.get_velocity().x**2 + actor.get_velocity().y**2) 

        vehicle_bounding_box = actor.bounding_box.extent
        vehicle.length = vehicle_bounding_box.x*2
        vehicle.width = vehicle_bounding_box.y*2

        return vehicle
  
    def getLanePoints(self, waypoints, flip=False):

        lane_cur = Lane()
        lane_points = []
        for i,waypoint in enumerate(waypoints):
            lane_point = LanePoint()
            lane_point.pose.y = waypoint.transform.location.y
            lane_point.pose.x = waypoint.transform.location.x
            lane_point.pose.theta = waypoint.transform.rotation.yaw * np.pi / 180 # CHECK : Changed this to radians.
            lane_point.width = 3.5 # TODO
            lane_points.append(lane_point)
        lane_cur.lane = lane_points

        return lane_cur
    
    def pathRequest(self, data):
        
        if self.first_frame_generated:
            data = data.path_plan
            
            ### Get requested pose and speed values ###
            tracking_pose = data.tracking_pose
            tracking_speed = data.tracking_speed / 3.6
            reset_sim = data.reset_sim
            print("Tracking Speed:", tracking_speed, "Current Speed:", self.getVehicleState(self.ego_vehicle).vehicle_speed)
            # print("RESET:",reset_sim)
            
            self.end_of_action = data.end_of_action
            self.action_progress = data.action_progress
            self.path_planner_terminate = data.path_planner_terminate
            
            ### Update ROS Sync ###
            self.id_waiting = self.id
            
            
            if(reset_sim):
                self.resetEnv()
                
            else:
                self.first_run = 0

                # time.sleep(1)
                ### Apply Control ###
                self.ego_vehicle.apply_control(self.vehicle_controller.run_step(tracking_speed, tracking_pose))

                ### Visualize requested waypoint ###
                tracking_loc = carla.Location(x=tracking_pose.x, y=tracking_pose.y, z=self.ego_vehicle.get_location().z)
                self.carla_handler.world.debug.draw_string(tracking_loc, 'O', draw_shadow=False,
                                                    color=carla.Color(r=255, g=0, b=0), life_time=1,
                                                    persistent_lines=True)
                
                #### Check Sync ###
                flag = 0
                while(flag == 0):
                    try:
                        self.carla_handler.world.tick()
                        flag = 1
                    except:
                        print("Missed Tick....................................................................................")
                        continue
                self.timestamp += self.simulation_sync_timestep
        else:
            self.first_frame_generated = True
            self.resetEnv()
            
        ### Extract State Information
        nearest_waypoint = self.carla_handler.world_map.get_waypoint(self.ego_vehicle.get_location(), project_to_road=True)
        state_information = self.carla_handler.get_state_information(self.ego_vehicle, self.original_lane)
        current_lane_waypoints, left_lane_waypoints, right_lane_waypoints, front_vehicle, rear_vehicle, actors_in_current_lane, actors_in_left_lane, actors_in_right_lane = state_information
        
        pedestrians_on_current_road = self.carla_handler.get_pedestrian_information(self.ego_vehicle)
        ####
        
        ####

        # Current Lane
        lane_cur = self.lane_cur#self.getLanePoints(current_lane_waypoints)
        
        # Left Lane
        lane_left = self.lane_left#self.getLanePoints(left_lane_waypoints) 
        
        # Right Lane
        lane_right = self.lane_right#self.getLanePoints(right_lane_waypoints)
        
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
            
        # Contruct enviroment state ROS message
        env_state = EnvironmentState()
        env_state.cur_vehicle_state = vehicle_ego
        env_state.front_vehicle_state = vehicle_front
        env_state.back_vehicle_state = vehicle_rear
        env_state.current_lane = lane_cur
        env_state.next_lane = lane_left
        env_state.adjacent_lane_vehicles = [self.getVehicleState(actor) for actor in actors_in_left_lane]
        env_state.speed_limit = 20
        # env_state.id = self.id
        
        reward_info = RewardInfo()
        reward_info.time_elapsed = self.timestamp
        reward_info.new_run = self.first_run
        reward_info.collision = self.collision_marker
        reward_info.action_progress = self.action_progress
        reward_info.end_of_action = self.end_of_action
        reward_info.path_planner_terminate = self.path_planner_terminate
        env_state.reward = reward_info


        
        # # publish environment state
        # self.env_msg = env_state
        # self.env_pub.publish(env_state)
        return SimServiceResponse(env_state)
            
        
            
        
        
  
  
    def pathCallback(self, data):
        self.lock.acquire()
        # print("Collision:", self.collision_marker)
        ### Check Sync ###
        if not data.id == self.id_waiting:
            self.lock.release()
            return
        
        ### Get requested pose and speed values ###
        tracking_pose = data.tracking_pose
        tracking_speed = data.tracking_speed
        reset_sim = data.reset_sim
        print("Tracking Speed:", tracking_speed, "Current Speed:", self.getVehicleState(self.ego_vehicle).vehicle_speed)
        # print("RESET:",reset_sim)
        
        self.end_of_action = data.end_of_action
        self.action_progress = data.action_progress
        
        ### Update ROS Sync ###
        self.id_waiting = self.id
        self.lock.release()
        
        
        if(reset_sim):
            self.resetEnv()
            
        else:
            self.lock.acquire()
            self.first_run = 0

            # time.sleep(1)
            ### Apply Control ###
            self.ego_vehicle.apply_control(self.vehicle_controller.run_step(tracking_speed, tracking_pose))

            ### Visualize requested waypoint ###
            tracking_loc = carla.Location(x=tracking_pose.x, y=tracking_pose.y, z=self.ego_vehicle.get_location().z)
            self.carla_handler.world.debug.draw_string(tracking_loc, 'O', draw_shadow=False,
                                                color=carla.Color(r=255, g=0, b=0), life_time=1,
                                                persistent_lines=True)
            
            #### Check Sync ###
            flag = 0
            while(flag == 0):
                try:
                    self.carla_handler.world.tick()
                    flag = 1
                except:
                    print("Missed Tick....................................................................................")
                    continue
            self.lock.release()
            self.timestamp += self.simulation_sync_timestep
            
        ### Extract State Information
        nearest_waypoint = self.carla_handler.world_map.get_waypoint(self.ego_vehicle.get_location(), project_to_road=True)
        state_information = self.carla_handler.get_state_information(self.ego_vehicle, self.original_lane)
        current_lane_waypoints, left_lane_waypoints, right_lane_waypoints, front_vehicle, rear_vehicle, actors_in_current_lane, actors_in_left_lane, actors_in_right_lane = state_information
        
        pedestrians_on_current_road = self.carla_handler.get_pedestrian_information(self.ego_vehicle)
        ####
        
        ####

        # Current Lane
        lane_cur = self.lane_cur#self.getLanePoints(current_lane_waypoints)
        
        # Left Lane
        lane_left = self.lane_left#self.getLanePoints(left_lane_waypoints) 
        
        # Right Lane
        lane_right = self.lane_right#self.getLanePoints(right_lane_waypoints)
        
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
            
        # Contruct enviroment state ROS message
        env_state = EnvironmentState()
        env_state.cur_vehicle_state = vehicle_ego
        env_state.front_vehicle_state = vehicle_front
        env_state.back_vehicle_state = vehicle_rear
        env_state.current_lane = lane_cur
        env_state.next_lane = lane_left
        env_state.adjacent_lane_vehicles = [self.getVehicleState(actor) for actor in actors_in_left_lane]
        env_state.speed_limit = 20
        env_state.id = self.id
        
        reward_info = RewardInfo()
        reward_info.time_elapsed = self.timestamp
        reward_info.new_run = self.first_run
        reward_info.collision = self.collision_marker
        reward_info.action_progress = self.action_progress
        reward_info.end_of_action = self.end_of_action
        env_state.reward = reward_info


        
        # publish environment state
        self.env_msg = env_state
        self.env_pub.publish(env_state)
        
        ## CARLA Sync
        self.id += 1
        if self.id > 100000:
            self.id = 0

      
    def publishFunc(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.lock.acquire()
            if self.env_msg:
                self.env_pub.publish(self.env_msg)
            self.lock.release()
            rate.sleep()    

    def destroy_actors_and_sensors(self):

        # for actor in self.carla_handler.world.get_actors().filter('sensor.*'):
        #     actor.destroy()
        if(self.collision_sensor is not None):
            self.collision_sensor.destroy()

        for actor in self.carla_handler.world.get_actors().filter('vehicle.*'):
            actor.destroy()
                
        self.vehicles_list = []
        print("All actors destroyed..\n") 
        
    def collision_handler(self, event):
        # print("collision lol")
        self.collision_marker = 1

    def resetEnv(self):

        settings = self.carla_handler.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.carla_handler.world.apply_settings(settings)
        
        self.destroy_actors_and_sensors()
        self.timestamp = 0
        self.collision_marker = 0
        self.first_run = 1

        
        synchronous_master = False

        try:
    
            if True:
                settings = self.carla_handler.world.get_settings()
                self.tm.traffic_manager.set_synchronous_mode(True)
                if not settings.synchronous_mode:
                    synchronous_master = True
                    settings.synchronous_mode = True
                    settings.fixed_delta_seconds = self.simulation_sync_timestep
                    self.carla_handler.world.apply_settings(settings)
                else:
                    synchronous_master = False
                    
            self.ego_vehicle = self.tm.reset()
            
            ## Handing over control
            del self.collision_sensor
            self.collision_sensor = self.carla_handler.world.spawn_actor(self.carla_handler.world.get_blueprint_library().find('sensor.other.collision'),
                                                    carla.Transform(), attach_to=self.ego_vehicle)

            self.collision_sensor.listen(lambda event: self.collision_handler(event))
            self.vehicle_controller = GRASPPIDController(self.ego_vehicle, args_lateral = {'K_P': 0.4, 'K_D': 0.0, 'K_I': 0}, args_longitudinal = {'K_P': 0.4, 'K_D': 0.0, 'K_I': 0.0})
            # time.sleep(1)
            self.original_lane = 5
                
        except rospy.ROSInterruptException:
            print("failed....")
            pass


                
                
    def initialize(self):
        # initialize node
        rospy.init_node(NODE_NAME, anonymous=True)

        # initialize publisher
        self.env_pub = rospy.Publisher(SIM_TOPIC_NAME, EnvironmentState, queue_size = 10)
        
        # initlize subscriber
        self.path_sub = rospy.Subscriber(PATH_PLAN_TOPIC_NAME, PathPlan, self.pathCallback)
        
        # initialize service
        self.planner_service = rospy.Service(SIM_SERVICE_NAME, SimService, self.pathRequest)

        # Start Client. Make sure Carla server is running before starting.

        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        print("Connection to CARLA server established!")

        # Create a CarlaHandler object. CarlaHandler provides some cutom built APIs for the Carla Server.
        self.carla_handler = CarlaHandler(client)
        self.client = client
        self.tm = CustomScenario(self.client, self.carla_handler)
        # Reset Environment
        self.resetEnv()
                
        # Initialize PID Controller
        self.vehicle_controller = GRASPPIDController(self.ego_vehicle, args_lateral = {'K_P': 0.4, 'K_D': 0.0, 'K_I': 0}, args_longitudinal = {'K_P': 0.4, 'K_D': 0.0, 'K_I': 0.0})


        state_information = self.carla_handler.get_state_information(self.ego_vehicle, self.original_lane)
        current_lane_waypoints, left_lane_waypoints, right_lane_waypoints, front_vehicle, rear_vehicle, actors_in_current_lane, actors_in_left_lane, actors_in_right_lane = state_information
        
        pedestrians_on_current_road = self.carla_handler.get_pedestrian_information(self.ego_vehicle)
        
        ##############################################################################################################
        # publish the first frame 
        # Current Lane
        self.lane_cur = self.getLanePoints(current_lane_waypoints)
        
        
        # Left Lane
        self.lane_left = self.getLanePoints(left_lane_waypoints)
        # Right Lane
        self.lane_right = self.getLanePoints(right_lane_waypoints)

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

            
        # Contruct enviroment state ROS message
        env_state = EnvironmentState()
        env_state.cur_vehicle_state = vehicle_ego
        env_state.front_vehicle_state = vehicle_front
        env_state.back_vehicle_state = vehicle_rear
        env_state.adjacent_lane_vehicles = [self.getVehicleState(actor) for actor in actors_in_left_lane] #TODO : Only considering left lane for now. Need to make this more general 
        env_state.current_lane = self.lane_cur
        env_state.next_lane = self.lane_left
        env_state.max_num_vehicles = 5
        env_state.speed_limit = 20
        env_state.id = self.id
        
        reward_info = RewardInfo()
        reward_info.time_elapsed = self.timestamp
        reward_info.new_run = self.first_run
        reward_info.collision = self.collision_marker
        env_state.reward = reward_info
       

        # Update Sync
        self.id_waiting = self.id
        self.id += 1
        if self.id > 100000:
            self.id = 0

        rate = rospy.Rate(10)

        # publish environment state
        self.env_msg = env_state
        self.env_pub.publish(env_state)
        rate.sleep()

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
	