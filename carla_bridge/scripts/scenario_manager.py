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
import re

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


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

class CustomScenario:
    def __init__(self, client, carla_handler):
        
        self.client = client
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.0)    
        # self.traffic_manager.global_percentage_speed_difference()

        self.world = self.client.get_world()
        self.carla_handler = carla_handler
        self.vehicles_list = []
        print("Scenario Manager Initialized...")
        
        self.world.set_weather(find_weather_presets()[2][0])

        self.scenarios_town04 = [[[40],3,4,100,20], [[46],-2,-3,100,10], [[40],3,4,100,20], [[47],-2,-3,50,10], [[39],3,4,100,20], [[38],3,4,150,20]]
        # self.scenarios_town04 = [[[47],-2,-3,50,10]]
        self.scenarios_town05 = [[[21,22],-1,-2,0,10], [[37], -2, -3, 25, 0]]
        self.scenarios_town03 = [[[8,7,6], 4, 5, 0, 0]]
        
  
        # self.client.set_timeout(20)
        # self.client.load_world('Town04')
        # brak
        # self.client.set_timeout(2)
        # brak
        
        
    def reset(self, warm_start=False, warm_start_duration=10):
                
        self.target_speed = 15       
        self.spawn_roads, self.left_lane_id, self.curr_lane_id, self.ego_spawn_idx, self.vehicle_init_speed = self.scenarios_town04[np.random.randint(0,len(self.scenarios_town04))]
        # 
        
        
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

        num_vehicles = np.random.randint(7,15)
        waypoints = self.world.get_map().generate_waypoints(distance=np.random.randint(15,30))
        road_waypoints = []
        for waypoint in waypoints:
            if(waypoint.road_id in self.spawn_roads and waypoint.lane_id  == self.left_lane_id):
                road_waypoints.append(waypoint)
        number_of_spawn_points = len(road_waypoints)
        # print("get", len(road_waypoints), "on road 12 with lane id 0")
        if num_vehicles < number_of_spawn_points:
            # print("randomize distance in between cars")
            random.shuffle(road_waypoints)
        elif num_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            # logging.warning(msg, num_vehicles, number_of_spawn_points)
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
            # batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))

        for response in self.client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                # logging.error(response.error)
                print("Response Error while applying batch!")
            else:
                self.vehicles_list.append(response.actor_id)
                print('created %s' % response.actor_id)
        my_vehicles = self.world.get_actors(self.vehicles_list)

        for n, v in enumerate(my_vehicles):
            
                
                    
            # nearest_waypoint = self.carla_handler.world_map.get_waypoint(v.get_location(), project_to_road=True)                    
            # current_speed_limit = v.get_speed_limit()
            # # current_speed = np.sqrt(v.get_velocity().x**2 + v.get_velocity().y**2 + v.get_velocity().z**2) * 3.6
            # new_limit_percentage = 100 - (self.target_speed * 100)/float(current_speed_limit)
            # print(n, ":", np.sqrt(v.get_velocity().x**2 + v.get_velocity().y**2 + v.get_velocity().z**2) * 3.6, nearest_waypoint.road_id, current_speed_limit, new_limit_percentage)
            # self.traffic_manager.vehicle_percentage_speed_difference(v, new_limit_percentage)
            # # time.sleep(2)

            
            # self.traffic_manager.vehicle_percentage_speed_difference(v, np.random.randint(40,50))
            self.traffic_manager.auto_lane_change(v,False)
            self.traffic_manager.ignore_lights_percentage(v, 100)
        
        
        ################# Spawn Ego ##########################
        ego_list = []
        # ego_spawn_pt = np.random.randint(100,200)
        filtered_waypoints = self.carla_handler.filter_waypoints(self.carla_handler.get_waypoints(1), road_id=self.spawn_roads[0], lane_id=self.curr_lane_id)
        # filtered_waypoints_left = self.carla_handler.filter_waypoints(self.carla_handler.get_waypoints(1), road_id=self.spawn_road, lane_id=3)

        # self.carla_handler.draw_waypoints(filtered_waypoints_left, road_id=self.spawn_road, section_id=None, life_time=10.0)
        spawn_point = filtered_waypoints[self.ego_spawn_idx].transform # Select random point from filtered waypoint list #TODO Initialization Scheme Design
        spawn_point.location.z = spawn_point.location.z + 0.1 # To avoid collision during spawn
        vehicle_blueprint = self.carla_handler.blueprint_library.filter('model3')[0]
        
        yaw = spawn_point.rotation.yaw *  np.pi / 180
        
        ego_list.append(SpawnActor(vehicle_blueprint, spawn_point).then(ApplyVelocity(FutureActor, carla.Vector3D(self.vehicle_init_speed*np.cos(yaw), self.vehicle_init_speed*np.sin(yaw),0))).then(SetAutopilot(FutureActor, True)))
        # ego_list.append(SpawnActor(vehicle_blueprint, spawn_point).then(SetAutopilot(FutureActor, True)))
        response = self.client.apply_batch_sync(ego_list, synchronous_master)
        ego_vehicle_ID = response[0].actor_id
        ego_vehicle = self.world.get_actors([ego_vehicle_ID])[0]
        self.traffic_manager.ignore_lights_percentage(ego_vehicle, 100)
        # self.ego_vehicle, ego_vehicle_ID = self.carla_handler.spawn_vehicle(spawn_point=spawn_point)
        
        for n, v in enumerate(my_vehicles):
         
            # self.traffic_manager.vehicle_percentage_speed_difference(v, 100)
            self.traffic_manager.collision_detection(v, ego_vehicle, False)

        
        
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
        return ego_vehicle, my_vehicles, self.target_speed
        