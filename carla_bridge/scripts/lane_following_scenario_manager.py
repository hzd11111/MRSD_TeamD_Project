import sys
import time
import random
# from configparse import ConfigParser
import subprocess
import carla
import ipdb
import numpy as np

sys.path.append("../../grasp_path_planner/scripts/")
sys.path.append("../../global_route_planner/")

from carla_handler import CarlaHandler
from global_planner import get_global_planner, draw_waypoints

from settings import *




def filter_waypoints(waypoints, road_id, lane_id = None):

    filtered_waypoints = []
    for wp in waypoints:

        if lane_id == None and wp.road_id == road_id:
            filtered_waypoints.append(wp) 
        elif wp.road_id == road_id and wp.lane_id==lane_id:
            filtered_waypoints.append(wp) 

    return filtered_waypoints 


class LaneFollowingScenario:

    def __init__(self, client, carla_handler):
        self.client = client
        self.world = client.get_world()
        self.carla_handler = carla_handler
        self.Map = self.world.get_map()
        self.spectator = self.world.get_spectator()

        self.traffic_manager = self.client.get_trafficmanager(TM_PORT)
        self.global_planner = get_global_planner(self.world, planner_resolution=1)
        self.vehicles_list = []
        self.ego_vehicle = []
        print("Scenario Manager Initialized...")

        # configuration params imported from settings.py
        self.config = LANE_FOLLOWING_CONFIG
        self.map_spawn_locations = SPAWN_LOCS_VEC

        # current lane information
        self.dist_between_waypoints = 1
        self.all_waypoints = self.Map.generate_waypoints(self.dist_between_waypoints)
        self.current_road_and_lane = None
        self.current_lane_wps = []
        self.total_non_ego_vehicles = None
        self.goal_waypoint = None

        # world and tm config settings
        self.synchronous_mode = SYNCHRONOUS
        self.no_rendering_mode = NO_RENDER_MODE
        self.fixed_delta_seconds = FIXED_DELTA_SECONDS
        self.hybrid_physics_mode = HYBRID_PHYSICS_MODE
        self.auto_lane_change = AUTO_LANE_CHANGE
        self.distance_to_leading_vehicle = DISTANCE_TO_LEADING_VEHICLES
        self.target_speed = self.config["target_speed"]

        # vehicle lists
        self.ego_vehicle = None
        self.non_ego_vehicle_list = None

    def configure_traffic_manager(self):
        tm = self.traffic_manager
        tm.set_synchronous_mode(self.synchronous_mode)
        # tm.set_hybrid_physics_mode(True)
        # tm.set_hybrid_physics_radius(10.0)
        tm.set_global_distance_to_leading_vehicle(self.distance_to_leading_vehicle)

    def configure_world(self, no_rendering_mode=False, synchronous_mode=True, \
                                                    fixed_delta_seconds=0.03):
        self.world.apply_settings(carla.WorldSettings(no_rendering_mode=no_rendering_mode,
                                                        synchronous_mode=synchronous_mode,
                                                        fixed_delta_seconds=fixed_delta_seconds))
        self.synchronous_mode = synchronous_mode
        self.no_rendering_mode = no_rendering_mode
        self.fixed_delta_seconds = fixed_delta_seconds

    def tick(self):     
        if self.synchronous_mode:
            self.world.tick()
    
    def warm_start(self, warm_start_duration=1):
        warm_start_curr = 0
        while warm_start_curr < warm_start_duration:
            warm_start_curr += self.fixed_delta_seconds
            if self.synchronous_mode:
                self.world.tick()
            else:
                self.world.wait_for_tick()

    def get_spawn_points(self, distance_bwn_waypoints):
        try:
            # Get a list of waypoints 1 m apart on the desired lane
            config = self.config

            ind = np.random.choice(len(self.map_spawn_locations))
            spawn_location = carla.Location(*self.map_spawn_locations[ind])

            loc = carla.Location(spawn_location)
            first_waypoint = self.Map.get_waypoint(loc, project_to_road=True, \
                                                    lane_type=carla.LaneType.Driving)
            waypoint_list = first_waypoint.next_until_lane_end(distance_bwn_waypoints)
            waypoint_spawn_list = waypoint_list[:-config["min_dist_to_end_of_lane_from_first_veh"]]
            draw_waypoints(self.world, waypoint_list, life_time=1)

            self.current_road_and_lane = (first_waypoint.road_id, first_waypoint.lane_id) 
            self.current_lane_wps = waypoint_list
            # TODO: Fix the way we spawn things

            # select waypoints from the list with some randomness
            selected_waypoints = []
            ptr = 0
            total_vehicles_possible = 0
            self.total_non_ego_vehicles = np.random.randint(config["min_non_ego_veh"],
                                                        config["max_non_ego_veh"]+1)
            
            for _ in range(self.total_non_ego_vehicles+1):
                if ptr >= len(waypoint_spawn_list): break
                selected_waypoints.append(waypoint_spawn_list[ptr])
                total_vehicles_possible += 1
                
                ptr += np.random.randint(config["min_dist_bwn_veh"], 
                                            config["max_dist_bwn_veh"])
                ptr += config["average_car_length"]
            self.total_non_ego_vehicles = total_vehicles_possible - 1

            # convert waypoints to spawn points
            spawn_points_list = [item.transform for item in selected_waypoints]
            for point in spawn_points_list: point.location.z += 0.2

            # select the position of the ego vehicle
            ego_veh_int = np.random.randint(0, total_vehicles_possible)
            temp = spawn_points_list.pop(ego_veh_int)
            ego_spawn_point = (temp, temp.get_forward_vector())
            # self.goal_waypoint = waypoint_list[ptr + config["goal_distance_to_travel"]]
            self.goal_waypoint = waypoint_list[-STOP_LINE_DISTANCE_FOR_LANE_CHANGE_TERMINATE]
            draw_waypoints(self.world, [self.goal_waypoint], 1, [255, 255, 0])

            # non-ego vehicle spawn points
            non_ego_spawn_points = [(spawn_points_list[i], 
                                    spawn_points_list[i].get_forward_vector())
                                    for i in range(self.total_non_ego_vehicles)]
            # vectors = [x.get_forward_vector() for x in spawn_points_list[1:5]]
        
        except Exception as e:
            print(e)
            import ipdb; ipdb.set_trace()
        
        return ego_spawn_point, non_ego_spawn_points

    def spawn(self, ego_spawn_point, non_ego_spawn_points):
        self.destroy_all_actors()

        # Get vehicle blueprints, mustang for non-ego, tesla for ego :D
        non_ego_vehicle_blueprint = self.world.get_blueprint_library().filter(NON_EGO_VEHICLE_MODEL)[0]
        ego_vehicle_blueprint = self.world.get_blueprint_library().filter(EGO_VEHICLE_MODEL)[0]
        
        # defining a few commands
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor
        ApplyVelocity = carla.command.ApplyVelocity

        # spawning non-ego vehicles
        batch = []
        for i in range(self.total_non_ego_vehicles):
            batch.append(SpawnActor(non_ego_vehicle_blueprint, non_ego_spawn_points[i][0])
                .then(ApplyVelocity(FutureActor, non_ego_spawn_points[i][1] * self.target_speed))
                .then(SetAutopilot(FutureActor, True, TM_PORT)))
        non_ego_vehicle_list = []
        for response in self.client.apply_batch_sync(batch, False):
            if response.error:
                print(response.error)
            else:
                non_ego_vehicle_list.append(self.world.get_actor(response.actor_id))
        
        # spawning ego vehicle
        ego_batch = []
        ego_vehicle_blueprint.set_attribute('role_name', 'ego')
        ego_batch.append(SpawnActor(ego_vehicle_blueprint, ego_spawn_point[0])
                .then(ApplyVelocity(FutureActor, ego_spawn_point[1] * self.target_speed))
                .then(SetAutopilot(FutureActor, True, TM_PORT)))
        for response in self.client.apply_batch_sync(ego_batch, False):
            if response.error:
                print(response.error)
            else:
                ego_vehicle = self.world.get_actor(response.actor_id)
        
        self.tick()

        # changing a few vehicle params: disable lane change, ignore rules yolo
        all_actors = self.world.get_actors().filter("vehicle*")
        for actor in all_actors:
            self.traffic_manager.auto_lane_change(actor, self.auto_lane_change)
            self.traffic_manager.ignore_lights_percentage(actor, IGNORE_LIGHTS_PERCENTAGE)
            self.traffic_manager.ignore_signs_percentage(actor, IGNORE_SIGNS_PERCENTAGE)
            # actor.set_simulate_physics(ACTOR_SIMULATE_PHYSICS)

        self.ego_vehicle = ego_vehicle
        self.non_ego_vehicle_list = non_ego_vehicle_list

        return ego_vehicle, non_ego_vehicle_list

    def destroy_all_actors(self):
        # destroy all old actors
        all_actors = self.world.get_actors().filter("vehicle*")
        self.client.apply_batch([carla.command.DestroyActor(x) for x in all_actors])
        self.world.tick()
        
    def disable_autopilot(self):
        SetAutopilot = carla.command.SetAutopilot
        self.client.apply_batch_sync(
            [SetAutopilot(self.ego_vehicle, False)], self.synchronous_mode)

    def get_global_path(self):
        '''For the lane following scenario, returns a global path, which is from
        the current ego_vehicle location to the end of the lane.'''
        try:
            start_location = self.current_lane_wps[0].transform.location
            # Lane tracking point is 5m from the intersection
            end_location = self.goal_waypoint.transform.location
            
            # if len(self.current_lane_wps) < 100: #TODO: Check logic here for smaller roads
            #     dist = DISTANCE_TO_INTERSECTION_FOR_SCENARIO_CHANGE
            #     end_location = self.current_lane_wps[-dist].transform.location 

            route = self.global_planner.trace_route(start_location, end_location)
            global_path_wps = [route[i][0] for i in range(len(route))]
        except Exception as e:
            print(e)
            import ipdb; ipdb.set_trace()

        return global_path_wps
    
    def set_spectator_view(spectator_trans):
        self.spectator.set_transform(spectator_trans)

    def reset(self):

        # configure asynchronous mode
        self.configure_traffic_manager()

        # destroy any remaining actors
        self.destroy_all_actors()

        # spawn vehicles
        ego_spawn_point, non_ego_spawn_points = self.get_spawn_points( \
                                        self.config["distance_bwn_waypoints"])
        self.spawn(ego_spawn_point, non_ego_spawn_points)

        # get global planner for lane follow
        global_path = self.get_global_path()

        # warm start
        if self.config["warm_start"]: 
            self.warm_start(self.config["warm_start_duration"])

        # disable autopilot for ego vehicle
        self.disable_autopilot()

        self.tick()

        return (
            self.ego_vehicle,
            self.non_ego_vehicle_list,
            global_path
        )



if __name__== "__main__":
    client = carla.Client("localhost", 2000)
    client.set_timeout(2.0)
    # carla_handler = CarlaHandler(client)
    carla_handler = None
    print("Connection to CARLA server established!")

    scenario = LaneFollowingScenario(client, carla_handler)
    
    while True:
        st = time.time()
        scenario.reset()
        print(time.time()-st)
    


