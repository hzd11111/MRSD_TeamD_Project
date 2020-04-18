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
from grasp_path_planner.msg import Pedestrian
from grasp_path_planner.msg import RewardInfo
from grasp_path_planner.msg import EnvironmentState
from grasp_path_planner.msg import PathPlan

from scenario_manager import CustomScenario
from grasp_path_planner.srv import SimService, SimServiceResponse


#######################################################################################

NODE_NAME = 'carla_bridge'
SIM_TOPIC_NAME = 'environment_state'
PATH_PLAN_TOPIC_NAME = 'path_plan'

SIM_SERVICE_NAME = 'simulator'

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
        
        self.simulation_sync_timestep = 0.05
        
        self.first_frame_generated = False
        self.path_planner_terminate = False
        
        self.action_progress = 0
        self.end_of_action = True
        # self.initialize()
        
        self.max_num_vehicles = 5
        self.last_call_reset = False
        
        self.speed_limit = None
        
        self.pedestrian = None
        self.pedestrian_wait_frames = 0
        self.last_time = time.time()
  
    def getVehicleState(self, actor):

        if(actor == None):
            return None

        vehicle = VehicleState()
        vehicle.vehicle_location.x = actor.get_transform().location.x
        vehicle.vehicle_location.y = actor.get_transform().location.y
        vehicle.vehicle_location.theta = actor.get_transform().rotation.yaw * np.pi / 180  #CHECK : Changed this to radians. 
        vehicle.vehicle_speed = np.sqrt(actor.get_velocity().x**2 + actor.get_velocity().y**2 + actor.get_velocity().z**2) * 3.6

        vehicle_bounding_box = actor.bounding_box.extent
        vehicle.length = vehicle_bounding_box.x*2
        vehicle.width = vehicle_bounding_box.y*2

        return vehicle
    
    def getPedestrianState(self, actor, pedestrian_radius=0.5):

        ## TODO: Processing them like vehicles for now
        if(actor == None):
            return None

        pedestrian_state = Pedestrian()
        pedestrian_state.exist = True
        pedestrian_state.pedestrian_location.x = actor.get_transform().location.x
        pedestrian_state.pedestrian_location.y = actor.get_transform().location.y
        pedestrian_state.pedestrian_location.theta = actor.get_transform().rotation.yaw * np.pi / 180 #CHECK : Changed this to radians. 
        pedestrian_state.radius = pedestrian_radius
        pedestrian_state.pedestrian_acceleration = 0
        pedestrian_state.pedestrian_speed = np.sqrt(actor.get_velocity().x**2 + actor.get_velocity().y**2 + actor.get_velocity().z**2) * 3.6

        return pedestrian_state
  
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
        
        # tmp = self.last_time
        # self.last_time = time.time()
        # print("Time diff:", self.last_time - tmp)
        
        reset_sim = False
        if self.first_frame_generated:
            data = data.path_plan
            
            ### Get requested pose and speed values ###
            tracking_pose = data.tracking_pose
            tracking_speed = data.tracking_speed# / 3.6
            reset_sim = data.reset_sim
            future_poses = data.future_poses
            
            # ego_location = self.ego_vehicle.get_location()
            # future_loc = carla.Location(x=ego_location.x, y=ego_location.y, z=ego_location.z + 1)
            # self.carla_handler.world.debug.draw_string(future_loc, str(tracking_speed), draw_shadow=False,
            #                 color=carla.Color(r=0, g=0, b=255), life_time=0.05,
            #                 persistent_lines=False)
            
            
            
            for future_pose in future_poses:
                future_loc = carla.Location(x=future_pose.x, y=future_pose.y, z=self.ego_vehicle.get_location().z)
                self.carla_handler.world.debug.draw_string(future_loc, 'O', draw_shadow=False,
                                color=carla.Color(r=255, g=0, b=0), life_time=0.05,
                                persistent_lines=True)
            
            
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
                
                #### Pedestrian Spawning
                if(self.tm.pedestrian_mode == True):
                    if(self.pedestrian is None):
                        current_ego_speed = np.sqrt(self.ego_vehicle.get_velocity().x**2 + self.ego_vehicle.get_velocity().y**2 + self.ego_vehicle.get_velocity().z**2) * 3.6
                        if(current_ego_speed > 25):
                            self.tm.pedestrian_controller.waypoints_list = self.carla_handler.get_next_waypoints(self.carla_handler.world_map.get_waypoint(self.ego_vehicle.get_location(), project_to_road=True), None, k=45)[17:19]
                            self.pedestrian = self.tm.pedestrian_controller.random_spawn()
                            self.tm.pedestrian_controller.cross_road()
                            print("Pedestrian Spawned")
                    else:
                        if(self.pedestrian_wait_frames != 0):
                            self.pedestrian_wait_frames -= 1
                        else:
                            # self.tm.pedestrian_controller.cross_road()
                            ego_location = self.ego_vehicle.get_location()
                            pedestrian_location = self.pedestrian.get_location()
                            # print("Distance:", np.sqrt((ego_location.x - pedestrian_location.x)**2 + (ego_location.y - pedestrian_location.y)**2 + (ego_location.z - pedestrian_location.z)**2))
                            
                                
                    
                # ### Maintain Speed ###
                # for n,v in enumerate(self.tm.world.get_actors().filter('vehicle.*')):
                    
                #     nearest_waypoint = self.carla_handler.world_map.get_waypoint(v.get_location(), project_to_road=True)                    
                #     current_speed_limit = v.get_speed_limit()
                #     # current_speed = np.sqrt(v.get_velocity().x**2 + v.get_velocity().y**2 + v.get_velocity().z**2) * 3.6
                #     new_limit_percentage = 100 - (self.speed_limit * 100)/float(current_speed_limit)
                #     # print(n, ":", np.sqrt(v.get_velocity().x**2 + v.get_velocity().y**2 + v.get_velocity().z**2) * 3.6, nearest_waypoint.road_id, current_speed_limit, new_limit_percentage)
                #     self.tm.traffic_manager.vehicle_percentage_speed_difference(v, new_limit_percentage)
                #     # time.sleep(2)

                     

                ### Visualize requested waypoint ###
                # tracking_loc = carla.Location(x=tracking_pose.x, y=tracking_pose.y, z=self.ego_vehicle.get_location().z)
                # self.carla_handler.world.debug.draw_string(tracking_loc, 'O', draw_shadow=False,
                #                                     color=carla.Color(r=148, g=0, b=211), life_time=1,
                #                                     persistent_lines=True)
                

                    
                
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
        # nearest_waypoint = self.carla_handler.world_map.get_waypoint(self.ego_vehicle.get_location(), project_to_road=True)
        # self.carla_handler.world.debug.draw_string(self.ego_vehicle.get_location(), 'O', draw_shadow=False,
        #                                                     color=carla.Color(r=0, g=0, b=255), life_time=1,
        #                                                     persistent_lines=True)
        
        
        state_information = self.carla_handler.get_state_information_new(self.ego_vehicle, self.original_lane)
        current_lane_waypoints, left_lane_waypoints, right_lane_waypoints, front_vehicle, rear_vehicle, actors_in_current_lane, actors_in_left_lane, actors_in_right_lane = state_information
        
        
        pedestrians_on_current_road = self.carla_handler.get_pedestrian_information(self.ego_vehicle)
        
        ####
        
        ####

        # Current Lane
        if(reset_sim == True):
            self.lane_cur = self.getLanePoints(current_lane_waypoints)
            lane_cur = self.lane_cur
            # Left Lane
            self.lane_left = self.getLanePoints(left_lane_waypoints)
            lane_left = self.lane_left
            # Right Lane
            self.lane_right = self.getLanePoints(right_lane_waypoints)
            lane_right = self.lane_right
            
        else:
            lane_cur = self.lane_cur
            
            # Left Lane
            lane_left = self.lane_left
            
            # Right Lane
            lane_right = self.lane_right
            
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
        env_state.adjacent_lane_vehicles, _ = self.getClosest([self.getVehicleState(actor) for actor in actors_in_left_lane], vehicle_ego, self.max_num_vehicles)
        env_state.speed_limit = self.speed_limit

        # print(env_state.cur_vehicle_state.vehicle_location.theta)
        for idx in _:
            tmp_vehicle = actors_in_left_lane[idx]
            tmp_transform = tmp_vehicle.get_transform()
            tmp_bounding_box = tmp_vehicle.bounding_box
            tmp_bounding_box.location += tmp_transform.location
            self.carla_handler.world.debug.draw_box(tmp_bounding_box, tmp_transform.rotation, life_time=0.05, color=carla.Color(r=128, g=0, b=128))
        
        
        
        reward_info = RewardInfo()
        reward_info.time_elapsed = self.timestamp
        reward_info.new_run = self.first_run
        reward_info.collision = self.collision_marker
        reward_info.action_progress = self.action_progress
        reward_info.end_of_action = self.end_of_action
        reward_info.path_planner_terminate = self.path_planner_terminate
        env_state.reward = reward_info
        
        ## Pedestrian
        if (self.pedestrian is not None):
            env_state.nearest_pedestrian = self.getClosestPedestrian([self.getPedestrianState(actor) for actor in [self.pedestrian]], vehicle_ego, 1)[0]
            tmp_transform = self.pedestrian.get_transform()
            tmp_bounding_box = self.pedestrian.bounding_box
            tmp_bounding_box.location += tmp_transform.location
            self.carla_handler.world.debug.draw_box(tmp_bounding_box, tmp_transform.rotation, life_time=0.05, color=carla.Color(r=128, g=0, b=128))
        else:
            env_state.nearest_pedestrian = Pedestrian()
            env_state.nearest_pedestrian.exist = False

        if(reset_sim):
            self.last_call_reset = True
        else:
            self.last_call_reset = False

        
        return SimServiceResponse(env_state)

    def destroy_actors_and_sensors(self):


        if(self.collision_sensor is not None):
            self.collision_sensor.destroy()

        for actor in self.tm.world.get_actors().filter('vehicle.*'):
            actor.destroy()
            
        for actor in self.tm.world.get_actors().filter('walker.*'):
            actor.destroy()
            
        if(self.pedestrian is not None):
            self.tm.pedestrian_controller.destroy()
            self.pedestrian = None
            self.pedestrian_wait_frames = 10
                
        self.vehicles_list = []
        print("All actors destroyed..\n") 
        
    def collision_handler(self, event):
        # print("collision lol")
        self.collision_marker = 1

    def resetEnv(self):
        
        self.destroy_actors_and_sensors()
        self.timestamp = 0
        self.collision_marker = 0
        self.first_run = 1

        # time.sleep(1)
        synchronous_master = True

        try:
    
            self.ego_vehicle, self.vehicles_list, self.speed_limit = self.tm.reset()
            ## Handing over control
            del self.collision_sensor
            self.collision_sensor = self.carla_handler.world.spawn_actor(self.carla_handler.world.get_blueprint_library().find('sensor.other.collision'),
                                                    carla.Transform(), attach_to=self.ego_vehicle)

            self.collision_sensor.listen(lambda event: self.collision_handler(event))
            self.vehicle_controller = GRASPPIDController(self.ego_vehicle, args_lateral = {'K_P': 0.5, 'K_D': 0, 'K_I': 0, 'dt':self.simulation_sync_timestep}, args_longitudinal = {'K_P': 0.5, 'K_D': 0, 'K_I': 0, 'dt':self.simulation_sync_timestep})
            # time.sleep(1)
            self.original_lane = -3
                
        except rospy.ROSInterruptException:
            print("failed....")
            pass
        
    def getClosest(self, adjacent_lane_vehicles, ego_vehicle, n=5):
        ego_x = ego_vehicle.vehicle_location.x
        ego_y = ego_vehicle.vehicle_location.y
        
        distances = [((ego_x - adjacent_lane_vehicles[i].vehicle_location.x)**2 + (ego_y - adjacent_lane_vehicles[i].vehicle_location.y)**2) for i in range(len(adjacent_lane_vehicles))]
        sorted_idx = np.argsort(distances)[:n]
        
        return [adjacent_lane_vehicles[i] for i in sorted_idx], sorted_idx
    
    def getClosestPedestrian(self, pedestrians, ego_vehicle, n=1):
        ego_x = ego_vehicle.vehicle_location.x
        ego_y = ego_vehicle.vehicle_location.y
        
        distances = [((ego_x - pedestrians[i].pedestrian_location.x)**2 + (ego_y - pedestrians[i].pedestrian_location.y)**2) for i in range(len(pedestrians))]
        sorted_idx = np.argsort(distances)[:n]
        
        return [pedestrians[i] for i in sorted_idx]
                
                
    def initialize(self):
        # initialize node
        rospy.init_node(NODE_NAME, anonymous=True)

        # initialize publisher
        self.env_pub = rospy.Publisher(SIM_TOPIC_NAME, EnvironmentState, queue_size = 10)
        
        # initialize service
        self.planner_service = rospy.Service(SIM_SERVICE_NAME, SimService, self.pathRequest)

        # Start Client. Make sure Carla server is running before starting.

        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        print("Connection to CARLA server established!")

        # Create a CarlaHandler object. CarlaHandler provides some cutom built APIs for the Carla Server.
        self.carla_handler = CarlaHandler(client)
        self.client = client
        
        settings = self.carla_handler.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.simulation_sync_timestep
        self.carla_handler.world.apply_settings(settings)
        
        self.tm = CustomScenario(self.client, self.carla_handler)
        
        # Reset Environment
        self.resetEnv()
                
        # Initialize PID Controller
        self.vehicle_controller = GRASPPIDController(self.ego_vehicle, args_lateral = {'K_P': 0.5, 'K_D': 0, 'K_I': 0, 'dt':self.simulation_sync_timestep}, args_longitudinal = {'K_P': 1, 'K_D': 0.0, 'K_I': 0.0, 'dt':self.simulation_sync_timestep})


        state_information = self.carla_handler.get_state_information_new(self.ego_vehicle, self.original_lane)
        current_lane_waypoints, left_lane_waypoints, right_lane_waypoints, front_vehicle, rear_vehicle, actors_in_current_lane, actors_in_left_lane, actors_in_right_lane = state_information
        
        self.current_lane = current_lane_waypoints
        self.left_lane = left_lane_waypoints
        
        # self.carla_handler.draw_waypoints(current_lane_waypoints, life_time=100)
        # self.carla_handler.draw_waypoints(left_lane_waypoints, life_time=100, color=True)
        
        
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
            vehicle_front = VehicleState()#vehicle_ego
        else:
            vehicle_front = self.getVehicleState(front_vehicle)
        
        # Rear vehicle
        if(rear_vehicle == None):
            vehicle_rear = VehicleState()#vehicle_ego
        else:	
            vehicle_rear = self.getVehicleState(rear_vehicle)

            
        # Contruct enviroment state ROS message
        env_state = EnvironmentState()
        env_state.cur_vehicle_state = vehicle_ego
        env_state.front_vehicle_state = vehicle_front
        env_state.back_vehicle_state = vehicle_rear
        env_state.adjacent_lane_vehicles,_ = self.getClosest([self.getVehicleState(actor) for actor in actors_in_left_lane], vehicle_ego, self.max_num_vehicles) #TODO : Only considering left lane for now. Need to make this more general 
        env_state.current_lane = self.lane_cur
        env_state.next_lane = self.lane_left
        env_state.max_num_vehicles = self.max_num_vehicles
        env_state.speed_limit = 20
        # env_state.id = self.id
        
        ## Pedestrian
        if (self.pedestrian is not None):
            env_state.nearest_pedestrian = self.getClosestPedestrian([self.getPedestrianState(actor) for actor in [self.pedestrian]], vehicle_ego, 1)[0]
        else:
            env_state.nearest_pedestrian = Pedestrian()
            env_state.nearest_pedestrian.exist = False
            
        
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

    def spin(self):
        print("Start Ros Spin")	
        # spin
        rospy.spin()
  
    
if __name__ == '__main__':
	try:
		carla_manager = CarlaManager()
		carla_manager.initialize()
		print("Initialize Done.....")
		carla_manager.spin()
	except rospy.ROSInterruptException:
		pass
	