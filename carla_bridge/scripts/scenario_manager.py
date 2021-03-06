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
import copy
import random
import threading
from typing import Tuple

# sys.path.append("/home/cckai/Documents/CARLA_0.9.8/PythonAPI/carla/dist/carla-0.9.8-py3.5-linux-x86_64.egg")
import carla
import re

import agents.navigation.controller
import numpy as np

# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from carla_handler import CarlaHandler
# sys.path.insert(0, "/home/mayank/Mayank/GRASP_ws/src/MRSD_TeamD_Project/carla_bridge/scripts")
from grasp_controller import GRASPPIDController
# sys.path.insert(0, '/opt/ros/kinetic/lib/python2.7/dist-packages')

from std_msgs.msg import String
from grasp_path_planner.msg import LanePoint
from grasp_path_planner.msg import LanePoint
from grasp_path_planner.msg import Lane
from grasp_path_planner.msg import VehicleState
from grasp_path_planner.msg import RewardInfo
from grasp_path_planner.msg import EnvironmentState
from grasp_path_planner.msg import PathPlan

from dynamic_pedestrian import DynamicPedestrian

from os import path
sys.path.append(path.join(path.dirname(__file__), '../../grasp_path_planner/scripts/'))
from settings import *


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

class CustomScenario:
    def __init__(self, client, carla_handler) -> None:
        
        self.client = client
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.traffic_manager.set_global_distance_to_leading_vehicle(2)    

        self.world = self.client.get_world()
        self.carla_handler = carla_handler
        self.vehicles_list = []
        print("Scenario Manager Initialized...")
        
        if(CURRENT_SCENARIO == Scenario.PEDESTRIAN):
            self.pedestrian_controller = DynamicPedestrian(self.world)
        self.world.set_weather(find_weather_presets()[0][0])

        self.scenarios_town04 = [[[40],3,4,100,20], [[39],3,4,100,20], [[46],-2,-3,100,10], [[47],-2,-3,50,10]]
        self.scenarios_town04 = [[[52],0,1,0,20]]
        self.scenarios_town05 = [[[37], -2, -3, 320, 20]]#[[[21,22],-1,-2,0,10]], [[37], -2, -3, 0, 0]]
        self.scenarios_town03 = [[[8,7,6], 4, 5, 0, 0]]
        self.scenarios_town01 = [[[12], 0, -1, 0, 0]]
        self.ego_init_speed = EGO_INIT_SPEED
        
        self.pedestrian_mode = False
        self.pedestrian_spawn_dist_low = None
        self.pedestrian_spawn_dist_high = None
        if(CURRENT_SCENARIO == Scenario.PEDESTRIAN):
            self.pedestrian_mode = True
            self.pedestrian_spawn_dist_low = WALKER_SPAWN_DIST_MIN
            self.pedestrian_spawn_dist_high = WALKER_SPAWN_DIST_MAX

        # self.client.set_timeout(30)
        # self.client.load_world("Town05")
        # brak
        # self.client.set_timeout(2)
        # [0, 0, 0, 32.21003291083181, 156.01067181080413, -29.388363302418508, -0.004837898956386594, 0.0]
        # brak
        
        
    def reset(self, warm_start=False, warm_start_duration=5) -> Tuple[int, carla.ActorList, int]:
                
        self.target_speed = 15       
        if(CURRENT_MODE == Mode.TRAIN):
            self.spawn_roads, self.left_lane_id, self.curr_lane_id, self.ego_spawn_idx, self.vehicle_init_speed = self.scenarios_town04[np.random.randint(0,len(self.scenarios_town04))]
        else:
            self.spawn_roads, self.left_lane_id, self.curr_lane_id, self.ego_spawn_idx, self.vehicle_init_speed = ROAD_IDs, LEFT_LANE_ID, RIGHT_LANE_ID, EGO_SPAWN_IDX, NPC_INIT_SPEED      
        
        self.traffic_manager.set_synchronous_mode(True)
        synchronous_master = True

        self.vehicles_list = []
        
        if(self.pedestrian_mode == True):
            self.pedestrian_controller.road_id = self.spawn_roads[0]
            self.pedestrian_controller.lane_id = self.curr_lane_id
        
        blueprints = self.world.get_blueprint_library().filter('vehicle.*')

        if True:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
            blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('t2')]

        if(self.pedestrian_mode == True):
            num_vehicles = 0
        else:
            num_vehicles = np.random.randint(LOW_NUM_VEHICLES,HIGH_NUM_VEHICLES)
            
        waypoints = self.world.get_map().generate_waypoints(distance=np.random.randint(NPC_SPAWN_POINT_GAP_LOW,NPC_SPAWN_POINT_GAP_HIGH))
        road_waypoints = []
        for waypoint in waypoints:
            if(waypoint.road_id in self.spawn_roads and waypoint.lane_id  == self.left_lane_id):
                road_waypoints.append(waypoint)
        number_of_spawn_points = len(road_waypoints)

        if num_vehicles < number_of_spawn_points:
            random.shuffle(road_waypoints)
        elif num_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            num_vehicles = number_of_spawn_points

        
        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor
        ApplyVelocity = carla.command.ApplyVelocity

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
            yaw = transform.rotation.yaw *  np.pi / 180 
            batch.append(SpawnActor(blueprint, transform).then(ApplyVelocity(FutureActor, carla.Vector3D(self.vehicle_init_speed*np.cos(yaw), self.vehicle_init_speed*np.sin(yaw),0))).then(SetAutopilot(FutureActor, True)))

        for response in self.client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                # logging.error(response.error)
                # print("Response Error while applying batch!")
                dummy_var = 0
            else:
                self.vehicles_list.append(response.actor_id)
                # print('created %s' % response.actor_id)
                dummy_var = 0
        my_vehicles = self.world.get_actors(self.vehicles_list)

        for n, v in enumerate(my_vehicles):
            
                
                    
            # nearest_waypoint = self.carla_handler.world_map.get_waypoint(v.get_location(), project_to_road=True)                    
            # current_speed_limit = v.get_speed_limit()
            # # current_speed = np.sqrt(v.get_velocity().x**2 + v.get_velocity().y**2 + v.get_velocity().z**2) * 3.6
            # new_limit_percentage = 100 - (self.target_speed * 100)/float(current_speed_limit)
            # print(n, ":", np.sqrt(v.get_velocity().x**2 + v.get_velocity().y**2 + v.get_velocity().z**2) * 3.6, nearest_waypoint.road_id, current_speed_limit, new_limit_percentage)
            # self.traffic_manager.vehicle_percentage_speed_difference(v, new_limit_percentage)
            # # time.sleep(2)
            
            # self.traffic_manager.distance_to_leading_vehicle(v, -100)

            self.traffic_manager.auto_lane_change(v,False)
            self.traffic_manager.ignore_lights_percentage(v,100)
            self.traffic_manager.ignore_signs_percentage(v,100)
        
        
        ################# Spawn Ego ##########################
        ego_list = []
        filtered_waypoints = self.carla_handler.filter_waypoints(self.carla_handler.get_waypoints(1), road_id=self.spawn_roads[0], lane_id=self.curr_lane_id)

        spawn_point = filtered_waypoints[EGO_SPAWN_IDX].transform # Select random point from filtered waypoint list #TODO Initialization Scheme Design
        spawn_point.location.z = spawn_point.location.z + 0.1 # To avoid collision during spawn
        vehicle_blueprint = self.carla_handler.blueprint_library.filter(EGO_VEHICLE_MAKE)[0]
        
        yaw = spawn_point.rotation.yaw *  np.pi / 180
        
        ego_list.append(SpawnActor(vehicle_blueprint, spawn_point).then(ApplyVelocity(FutureActor, carla.Vector3D(self.ego_init_speed*np.cos(yaw), self.ego_init_speed*np.sin(yaw),0))).then(SetAutopilot(FutureActor, True)))
        response = self.client.apply_batch_sync(ego_list, synchronous_master)
        ego_vehicle_ID = response[0].actor_id
        ego_vehicle = self.world.get_actors([ego_vehicle_ID])[0]
        self.traffic_manager.ignore_lights_percentage(ego_vehicle, 100)
        
        if(not LIVES_MATTER):
            for v in my_vehicles:
                # self.traffic_manager.vehicle_percentage_speed_difference(v, 100)
                self.traffic_manager.collision_detection(v, ego_vehicle, False)

        
        
        print("Warming up....")
        warm_start_curr = 0
        while warm_start_curr < warm_start_duration:
            warm_start_curr += 0.1
            if synchronous_master:
                self.world.tick()
            else:
                self.world.wait_for_tick()                
        
        self.client.apply_batch_sync([SetAutopilot(ego_vehicle, False)], synchronous_master)
        
            
        print("Control handed to system....")
        return ego_vehicle, my_vehicles, self.target_speed