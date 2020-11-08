"""
Author:Scott
Date: 10/20/20
Description: Generate lane switching scneario at selected road and fill non ego vehicles at other lanes
             Ego vehicles is garantueed to be at the first half section of the lane its spawned to perform lane switching
             Swithching_left/right could be tuned at town05_city config
"""
import sys
import time
import random
import subprocess
import carla
import numpy as np
import logging
from carla_handler import CarlaHandler
from collections import defaultdict
sys.path.append("../../global_route_planner/")
sys.path.append("../../grasp_path_planner/scripts/")
from global_planner import get_global_planner
from settings import *

#CONFIGURATIONS:
town05_city = { "road_ids": [6, 7, 45, 46],
                "distance_bwn_waypoints":1,
                "max_ego_line_non_ego_vehicles":4, # num of non ego vehicles in ego lane
                "min_ego_line_non_ego_vehicles":1, # num of non ego vehicles in ego lane
                "max_non_ego_line_non_ego_vehicles":4, # num of non ego vehicles in every other lane
                "min_non_ego_line_non_ego_vehicles":1, # num of non ego vehicles in every other lane
                "target_speed":1,
                "max_dist_bwn_veh":15,
                "min_dist_bwn_veh":3,
                "max_dist_bwn_veh_other_lane":15,
                "min_dist_bwn_veh_other_lane":3,
                "average_car_length":5,
                "swithching_left": False,
                "goal_distance_to_travel":30,
                "min_spawn_distance_from_EOL":30,                 
}

def filter_waypoints(waypoints, road_id, lane_id):
    """
        Get all the waypoints with specified road_id and lane_id
    """
    filtered_waypoints = []
    for wp in waypoints:
        if wp.road_id == road_id and wp.lane_id==lane_id:
            filtered_waypoints.append(wp) 
    return filtered_waypoints

def filter_non_ego_waypoints(waypoints, road_id, lane_id):
    """
        Get all the waypoints in specified road_id but with lane_id
    """
    non_ego_waypoints_lists = defaultdict(list)
    for wp in waypoints:
        if wp.road_id == road_id and wp.lane_id!=lane_id:
            non_ego_waypoints_lists[wp.lane_id].append(wp)
    for lane_id in non_ego_waypoints_lists.keys(): 
        if lane_id > 0:
            non_ego_waypoints_lists[lane_id].reverse()
    return non_ego_waypoints_lists

def get_lane_ids(waypoints, road_id, ego_lane_id):
    """
        Get all the lane_ids not on the lane with ego_vehicle with specified road_id and ego_lane_id
    """
    lane_ids = set()
    for wp in waypoints:
        if wp.road_id == road_id:
            lane_ids.add(wp.lane_id)
    lane_ids.discard(ego_lane_id)
    return lane_ids 

class LaneSwitchingScenario:

    def __init__(self, client, carla_handler):
        self.client = client
        self.world = client.get_world()
        self.carla_handler = carla_handler
        self.Map = self.world.get_map()
        self.spectator = self.world.get_spectator()

        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.vehicles_list = []
        self.ego_vehicle = []
        
        print("Scenario Manager Initialized...")
        self.config = town05_city

        # world and tm config settings
        self.synchronous_mode = True
        self.no_rendering_mode = False
        self.fixed_delta_seconds = 0.05
        self.hybrid_physics_mode = True
        self.auto_lane_change = False
        self.distance_to_leading_vehicle = 2
        self.target_speed = self.config["target_speed"]
        self.global_planner = get_global_planner(self.world, planner_resolution=1)


        # vehicle lists
        self.ego_vehicle = None
        self.non_ego_vehicle_list = None

    def configure_traffic_manager(self):
        tm = self.traffic_manager
        tm.set_synchronous_mode(self.synchronous_mode)
        tm.set_global_distance_to_leading_vehicle(self.distance_to_leading_vehicle)

    def tick(self):            
        if self.synchronous_mode:
            self.world.tick()
    
    def warm_start(self, warm_start_duration=5):
        warm_start_curr = 0
        while warm_start_curr < warm_start_duration:
            warm_start_curr += self.fixed_delta_seconds
            if self.synchronous_mode:
                self.world.tick()
            else:
                self.world.wait_for_tick()

    def spawn(self, distance_bwn_waypoints, switching_left = True):
        self.destroy_all_actors()
        left_turning_lane_ids = [2, -2]
        right_turning_lane_ids = [1, -1]
        
        road_id = random.choice(self.config["road_ids"])
        ego_line_id = random.choice(left_turning_lane_ids) if switching_left else random.choice(right_turning_lane_ids)
        
        all_waypoints = self.world.get_map().generate_waypoints(1)
        ego_waypoint_list = filter_waypoints(all_waypoints, road_id, ego_line_id)
        if ego_line_id > 0: ego_waypoint_list.reverse()
        
        non_ego_waypoints_lists = filter_non_ego_waypoints(all_waypoints, road_id, ego_line_id)
    

        self.current_lane_wps = ego_waypoint_list
        # Assume only line 1 and line 2 exist
        eog_line_id_positivity = 1 if ego_line_id > 0 else -1
        target_line_id = (3 - abs(ego_line_id)) * eog_line_id_positivity
        self.goal_waypoint = non_ego_waypoints_lists[target_line_id][-int(len(non_ego_waypoints_lists[target_line_id])/5)]
        ego_spawn_point, non_ego_spawn_points = self.get_ego_spawn_points(distance_bwn_waypoints, ego_waypoint_list)
        self.spawn_vehicles_on_selected_lane(ego_spawn_point, non_ego_spawn_points)
        
        for non_ego_waypoints_list in non_ego_waypoints_lists.values():
            self.spawn_vehicles_on_selected_lane(None, self.get_non_ego_spawn_points(non_ego_waypoints_list))

        # self.spawn_non_ego_lane_vehicles(road_id, ego_line_id)

    def get_ego_spawn_points(self, distance_bwn_waypoints, waypoint_list):
        
        config = self.config
        # select waypoints from the list with some randomness
        selected_waypoints = []
        ptr = 0
        non_ego_vehicle_num = np.random.randint(config["min_ego_line_non_ego_vehicles"], config["max_ego_line_non_ego_vehicles"])
        maximum_ego_vehicle_idx = 0
        print("Road Id: ", waypoint_list[0].road_id)
        print("Total waypoint from this road: ", len(waypoint_list))
        print("Remaining waypoint for Ego V: ", len(waypoint_list) - config["min_spawn_distance_from_EOL"])

        if((len(waypoint_list) - config["min_spawn_distance_from_EOL"]) < 0):
            raise Exception("Select Training road or setting config")
        
        for _ in range(non_ego_vehicle_num + 1):
            ptr += np.random.randint(config["min_dist_bwn_veh"], config["max_dist_bwn_veh"])
            if ptr >= (len(waypoint_list) - 1):
                print("This line will not fit the required vehicle num")
                break
            selected_waypoints.append(waypoint_list[ptr])
            if (ptr < len(waypoint_list) - config["min_spawn_distance_from_EOL"]):
                maximum_ego_vehicle_idx += 1
            ptr += config["average_car_length"]

        # convert waypoints to spawn points
        spawn_points_list = [item.transform for item in selected_waypoints]

        for point in spawn_points_list: point.location.z += 0.2 # prevent collision
        # select the position of the ego vehicle
        ego_veh_int = np.random.randint(0, max(1, maximum_ego_vehicle_idx))
        temp = spawn_points_list.pop(ego_veh_int)
        ego_spawn_point = (temp, temp.get_forward_vector())
        # non-ego vehicle spawn points
        non_ego_spawn_points = [(spawn_points_list[i], spawn_points_list[i].get_forward_vector()) \
                                for i in range(len(spawn_points_list))]
        # vectors = [x.get_forward_vector() for x in spawn_points_list[1:5]]
        return ego_spawn_point, non_ego_spawn_points

    def get_non_ego_spawn_points(self, waypoint_list):
        config = self.config
        # select waypoints from the list with some randomness
        selected_waypoints = []
        ptr = 0
        non_ego_vehicle_num = np.random.randint(config["min_non_ego_line_non_ego_vehicles"], config["max_non_ego_line_non_ego_vehicles"] + 1)
        for _ in range(non_ego_vehicle_num):
            ptr += np.random.randint(config["min_dist_bwn_veh_other_lane"], config["max_dist_bwn_veh_other_lane"])
            if ptr >= (len(waypoint_list) - 1): 
                print("This line will not fit the required vehicle num")
                break
            selected_waypoints.append(waypoint_list[ptr])
            ptr += config["average_car_length"]

        # convert waypoints to spawn points
        spawn_points_list = [item.transform for item in selected_waypoints]

        for point in spawn_points_list: point.location.z += 0.2 # prevent collision
        
        # non-ego vehicle spawn points
        non_ego_spawn_points = [(spawn_points_list[i], spawn_points_list[i].get_forward_vector()) \
                                for i in range(len(spawn_points_list))]
        return non_ego_spawn_points

    def spawn_vehicles_on_selected_lane(self, ego_spawn_point, non_ego_spawn_points):

        # Get vehicle blueprints, mustang for non-ego, tesla for ego :D
        self.non_ego_vehicle_blueprint = self.world.get_blueprint_library().filter('vehicle.mustang.mustang')[0]
        ego_vehicle_blueprint = self.world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
        
        # defining a few commands
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor
        ApplyVelocity = carla.command.ApplyVelocity

        # spawning non-ego vehicles
        batch = []
        for i in range(len(non_ego_spawn_points)): 
            batch.append(SpawnActor(self.non_ego_vehicle_blueprint, non_ego_spawn_points[i][0])
                .then(ApplyVelocity(FutureActor, non_ego_spawn_points[i][1] * self.target_speed))
                .then(SetAutopilot(FutureActor, True)))
        non_ego_vehicle_list = []
        for response in self.client.apply_batch_sync(batch, False):
            if response.error:
                print(response.error)
            else:
                non_ego_vehicle_list.append(self.world.get_actor(response.actor_id))
        
        ego_vehicle = None
        if ego_spawn_point is not None:
            # spawning ego vehicle
            ego_batch = []
            ego_batch.append(SpawnActor(ego_vehicle_blueprint, ego_spawn_point[0])
                    .then(ApplyVelocity(FutureActor, ego_spawn_point[1] * self.target_speed))
                    .then(SetAutopilot(FutureActor, True)))
            for response in self.client.apply_batch_sync(ego_batch, False):
                if response.error:
                    print(response.error)
                else:
                    ego_vehicle = self.world.get_actor(response.actor_id)
            ego_vehicle_spawn_waypoint = self.world.get_map().get_waypoint(
                ego_vehicle.get_location(), project_to_road=True
            )
            print ("Ego V spawned ", len(ego_vehicle_spawn_waypoint.next_until_lane_end(1)), "m away from EOL")
        
        self.tick()

        # changing a few vehicle params: disable lane change, ignore rules yolo
        all_actors = self.world.get_actors().filter("vehicle*")
        for actor in all_actors:
            self.traffic_manager.auto_lane_change(actor, self.auto_lane_change)
            self.traffic_manager.ignore_lights_percentage(actor, IGNORE_LIGHTS_PERCENTAGE)
            self.traffic_manager.ignore_signs_percentage(actor, IGNORE_SIGNS_PERCENTAGE)
            actor.set_simulate_physics(True)

        if ego_spawn_point is not None:
            self.ego_vehicle = ego_vehicle
        if self.non_ego_vehicle_list is None:
            self.non_ego_vehicle_list = non_ego_vehicle_list
        else: 
            self.non_ego_vehicle_list.extend(non_ego_vehicle_list)

        return ego_vehicle, non_ego_vehicle_list

    def destroy_all_actors(self):
        # destroy all old actors
        all_actors = self.world.get_actors().filter("vehicle*")
        self.client.apply_batch([carla.command.DestroyActor(x) for x in all_actors])
        self.tick()

    def disable_autopilot(self):
        SetAutopilot = carla.command.SetAutopilot
        self.client.apply_batch_sync(
            [SetAutopilot(self.ego_vehicle, False)], self.synchronous_mode)
        
    def get_global_path(self):
        '''For the lane following scenario, returns a global path, which is from
        the current ego_vehicle location to the end of the lane.'''
        start_location = self.current_lane_wps[0].transform.location
        # Lane tracking point is 5m from the intersection
        end_location = self.goal_waypoint.transform.location
        # end_location = self.current_lane_wps[-15].transform.location 

        route = self.global_planner.trace_route(start_location, end_location)
        global_path_wps = [route[i][0] for i in range(len(route))]

        return global_path_wps
    

    def reset(self, warm_start=True, warm_start_duration=2, switching_left=False):
        # reset camera view
        # self.spectator.set_transform(spectator_trans)

        # configure asynchronous mode
        # self.configure_world(self.no_rendering_mode,self.synchronous_mode,self.fixed_delta_seconds)
        self.configure_traffic_manager()
        # self.tick()

        # destroy any remaining actors
        self.destroy_all_actors()

        # spawn vehicles
        self.spawn(self.config["distance_bwn_waypoints"],switching_left)
        # self.tick()

        global_path = self.get_global_path()


        # warm start
        if warm_start: self.warm_start(warm_start_duration)
        
        

        # disable autopilot for ego vehicle
        self.disable_autopilot()

        return self.ego_vehicle, self.non_ego_vehicle_list, global_path



if __name__== "__main__":
    client = carla.Client("localhost", 2000)
    client.set_timeout(2.0)
    # carla_handler = CarlaHandler(client)
    carla_handler = None
    print("Connection to CARLA server established!")

    scenario = LaneSwitchingScenario(client, carla_handler)
    
    while True:
        st = time.time()
        scenario.reset()
        print(time.time()-st)
    





