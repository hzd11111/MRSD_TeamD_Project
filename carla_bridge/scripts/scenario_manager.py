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


class CustomScenario:
    def __init__(self, client, carla_handler):
        
        self.client = client
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.0)    
        self.world = self.client.get_world()
        self.carla_handler = carla_handler
        self.vehicles_list = []
        print("Scenario Manager Initialized...")
        
    def reset(self, warm_start=False, warm_start_duration=10):
                
        self.traffic_manager.set_synchronous_mode(True)

        self.vehicles_list = []
        synchronous_master = True
        
        blueprints = self.world.get_blueprint_library().filter('vehicle.*')

        if True:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
            blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('t2')]

        num_vehicles = 15
        waypoints = self.world.get_map().generate_waypoints(distance=1)
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
        my_vehicles = self.world.get_actors(self.vehicles_list)

        for n, v in enumerate(my_vehicles):
            # c = v.get_physics_control()
            # c.max_rpm = args.max_vehicles_speed * 133 
            # v.apply_physics_control(c)
            # if n == 0:
            #     print("vehicles' speed limit:", v.get_speed_limit())
            self.traffic_manager.auto_lane_change(v,False)
        
        
        ################# Spawn Ego ##########################
        ego_list = []
        filtered_waypoints = self.carla_handler.filter_waypoints(self.carla_handler.get_waypoints(1), road_id=40, lane_id=5)
        spawn_point = filtered_waypoints[100].transform # Select random point from filtered waypoint list #TODO Initialization Scheme Design
        spawn_point.location.z = spawn_point.location.z + 0.1 # To avoid collision during spawn
        vehicle_blueprint = self.carla_handler.blueprint_library.filter('model3')[0]
        ego_list.append(SpawnActor(vehicle_blueprint, spawn_point).then(SetAutopilot(FutureActor, True)))
        response = self.client.apply_batch_sync(ego_list, synchronous_master)
        ego_vehicle_ID = response[0].actor_id
        ego_vehicle = self.world.get_actors([ego_vehicle_ID])[0]
        # self.ego_vehicle, ego_vehicle_ID = self.carla_handler.spawn_vehicle(spawn_point=spawn_point)
        print("Ego spawned!")    
        
        
        print("Warming up....")
        
        warm_start_curr = 0
        while warm_start_curr < warm_start_duration:
            warm_start_curr += 0.1
            if synchronous_master:
                self.world.tick()
            else:
                self.world.wait_for_tick()        
        
        print("Bombs away....")
        return ego_vehicle
        