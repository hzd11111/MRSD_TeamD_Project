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

#######################################################################################

NODE_NAME = 'carla_bridge'
SIM_TOPIC_NAME = 'environment_state'
PATH_PLAN_TOPIC_NAME = 'path_plan'

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
  
    def getLanePoints(self, waypoints, flip=False):

        lane_cur = Lane()
        lane_points = []
        for i,waypoint in enumerate(waypoints):
            lane_point = LanePoint()
            lane_point.pose.y = waypoint.transform.location.y
            lane_point.pose.x = waypoint.transform.location.x
            lane_point.pose.theta = waypoint.transform.rotation.yaw * np.pi / 180 # CHECK : Changed this to radians.
            lane_point.width = 4 # TODO
            lane_points.append(lane_point)
        lane_cur.lane = lane_points

        return lane_cur
  
  
    def pathCallback(self, data):
        self.lock.acquire()
        print("Collision:", self.collision_marker)
        ### Check Sync ###
        if not data.id == self.id_waiting:
            self.lock.release()
            return
        
        ### Get requested pose and speed values ###
        tracking_pose = data.tracking_pose
        tracking_speed = data.tracking_speed
        reset_sim = data.reset_sim
        print("RESET:",reset_sim)
        
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
            self.timestamp += 0.05
            
        ### Extract State Information
        nearest_waypoint = self.carla_handler.world_map.get_waypoint(self.ego_vehicle.get_location(), project_to_road=True)
        state_information = self.carla_handler.get_state_information(self.ego_vehicle, self.original_lane)
        current_lane_waypoints, left_lane_waypoints, right_lane_waypoints, front_vehicle, rear_vehicle, actors_in_current_lane, actors_in_left_lane, actors_in_right_lane = state_information
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
        for actor in self.carla_handler.world.get_actors().filter('vehicle.*'):
            actor.destroy()
        for actor in self.carla_handler.world.get_actors().filter('sensor.*'):
            actor.destroy()
                
        self.vehicles_list = []
        print("All actors destroyed..\n") 
        
    def collision_handler(self, event):
        print("collision lol")
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

        


        try:

            traffic_manager = self.client.get_trafficmanager(8000)
            traffic_manager.set_global_distance_to_leading_vehicle(2.0)

            synchronous_master = False



            blueprints = self.carla_handler.world.get_blueprint_library().filter('vehicle.*')

            if True:
                blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
                blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
                blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
                blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
                blueprints = [x for x in blueprints if not x.id.endswith('t2')]

            num_vehicles = 15
            waypoints = self.carla_handler.world.get_map().generate_waypoints(distance=1)
            road_waypoints = []
            for waypoint in waypoints:
                if(waypoint.road_id == 40 and waypoint.lane_id == 4):
                    road_waypoints.append(waypoint)
            number_of_spawn_points = len(road_waypoints)
            print("get", len(road_waypoints), "on road 12 with lane id 0")
            if num_vehicles < number_of_spawn_points:
                print("randomize distance in between cars")
                random.shuffle(road_waypoints)
            elif num_vehicles > number_of_spawn_points:
                msg = 'requested %d vehicles, but could only find %d spawn points'
                # logging.warning(msg, num_vehicles, number_of_spawn_points)
                num_vehicles = number_of_spawn_points

            
            # @todo cannot import these directly.
            SpawnActor = carla.command.SpawnActor
            SetAutopilot = carla.command.SetAutopilot
            FutureActor = carla.command.FutureActor

            # --------------
            # Spawn vehicles
            # --------------
            batch = []
            for n, t in enumerate(road_waypoints):
                if n >= num_vehicles:
                    break
                blueprint = random.choice(blueprints)
                if blueprint.has_attribute('color'):
                    color = random.choice(blueprint.get_attribute('color').recommended_values)
                    blueprint.set_attribute('color', color)
                if blueprint.has_attribute('driver_id'):
                    driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                    blueprint.set_attribute('driver_id', driver_id)
                blueprint.set_attribute('role_name', 'autopilot')
                transform = t.transform
                transform.location.z += 0.1
                batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))

            for response in self.client.apply_batch_sync(batch, synchronous_master):
                if response.error:
                    # logging.error(response.error)
                    print("Response Error while applying batch!")
                else:
                    self.vehicles_list.append(response.actor_id)
                    print('created %s' % response.actor_id)
            my_vehicles = self.carla_handler.world.get_actors(self.vehicles_list)

            for n, v in enumerate(my_vehicles):
                # c = v.get_physics_control()
                # c.max_rpm = args.max_vehicles_speed * 133 
                # v.apply_physics_control(c)
                # if n == 0:
                #     print("vehicles' speed limit:", v.get_speed_limit())
                traffic_manager.auto_lane_change(v,False)
                
            # while True:
            #     if True:#args.sync and synchronous_master:
            #         self.carla_handler.world.tick()
            #     else:
            #         self.carla_handler.world.wait_for_tick()
                
            # Spawn ego vehicle on road 
            filtered_waypoints = self.carla_handler.filter_waypoints(self.carla_handler.get_waypoints(1), road_id=40, lane_id=5)
            spawn_point = filtered_waypoints[100].transform # Select random point from filtered waypoint list #TODO Initialization Scheme Design
            spawn_point.location.z = spawn_point.location.z + 0.1 # To avoid collision during spawn
            self.ego_vehicle, ego_vehicle_ID = self.carla_handler.spawn_vehicle(spawn_point=spawn_point)
            print("Ego spawned!")
                        
            self.collision_sensor = self.carla_handler.world.spawn_actor(self.carla_handler.world.get_blueprint_library().find('sensor.other.collision'),
                                                    carla.Transform(), attach_to=self.ego_vehicle)

            self.collision_sensor.listen(lambda event: self.collision_handler(event))
            self.vehicle_controller = GRASPPIDController(self.ego_vehicle, args_lateral = {'K_P': 0.15, 'K_D': 0.0, 'K_I': 0}, args_longitudinal = {'K_P': 0.15, 'K_D': 0.0, 'K_I': 0.0})
            time.sleep(1)
            self.original_lane = 5
            
            if True:
                settings = self.carla_handler.world.get_settings()
                traffic_manager.set_synchronous_mode(True)
                if not settings.synchronous_mode:
                    synchronous_master = True
                    settings.synchronous_mode = True
                    settings.fixed_delta_seconds = 0.05
                    self.carla_handler.world.apply_settings(settings)
                else:
                    synchronous_master = False
                
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

        # Start Client. Make sure Carla server is running before starting.

        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        print("Connection to CARLA server established!")

        # Create a CarlaHandler object. CarlaHandler provides some cutom built APIs for the Carla Server.
        self.carla_handler = CarlaHandler(client)
        self.client = client

        # Reset Environment
        self.resetEnv()
                
        # Initialize PID Controller
        self.vehicle_controller = GRASPPIDController(self.ego_vehicle, args_lateral = {'K_P': 0.15, 'K_D': 0.0, 'K_I': 0}, args_longitudinal = {'K_P': 0.15, 'K_D': 0.0, 'K_I': 0.0})


        state_information = self.carla_handler.get_state_information(self.ego_vehicle, self.original_lane)
        current_lane_waypoints, left_lane_waypoints, right_lane_waypoints, front_vehicle, rear_vehicle, actors_in_current_lane, actors_in_left_lane, actors_in_right_lane = state_information
        
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
	