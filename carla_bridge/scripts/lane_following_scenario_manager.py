import sys
import time
import random
# from configparse import ConfigParser
import subprocess
import carla
import numpy as np

from carla_handler import CarlaHandler

#CONFIGURATIONS:
town04_highway = {'X':-356, 'Y':30, 'Z':0.2, 
                  "distance_bwn_waypoints":1,
                  "total_non_ego_vehicles":4,
                  "max_vehicles_in_front":2,
                  "target_speed":15,
                  "max_dist_bwn_veh":20,
                  "min_dist_bwn_veh":3,
                  "average_car_length":5,
                 }
spectator_trans = carla.Transform(carla.Location(x=-360.875458, y=-18.428667, z=46.781178), \
                                carla.Rotation(pitch=-39.588058, yaw=41.867565, roll=0.000029))


class LaneFollowingScenario:

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

        self.config = town04_highway #change to change scenario parameters
        
        # world and tm config settings
        self.synchronous_mode = True
        self.no_rendering_mode = False
        self.fixed_delta_seconds = 0.05
        self.hybrid_physics_mode = True
        self.auto_lane_change = False
        self.distance_to_leading_vehicle = 2
        self.target_speed = self.config["target_speed"]

        # vehicle lists
        self.ego_vehicle = None
        self.non_ego_vehicle_list = None

        # TODO: Reset camera angle
        # TODO: Throw an error for wrong town
        # TODO:
        # TODO: 

    def configure_traffic_manager(self):
        tm = self.traffic_manager
        tm.set_synchronous_mode(self.synchronous_mode)
        tm.set_hybrid_physics_mode(self.hybrid_physics_mode)
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
    
    def warm_start(self, warm_start_duration=5):
        warm_start_curr = 0
        while warm_start_curr < warm_start_duration:
            warm_start_curr += self.fixed_delta_seconds
            if self.synchronous_mode:
                self.world.tick()
            else:
                self.world.wait_for_tick()

    def get_spawn_points(self, distance_bwn_waypoints):
        
        # Get a list of waypoints 1 m apart on the desired lane
        config = self.config
        loc = carla.Location(x=config['X'], y=config['Y'], z=config['Z'])
        first_waypoint = self.Map.get_waypoint(loc, project_to_road=True, \
                                                lane_type=carla.LaneType.Driving)
        waypoint_list = first_waypoint.next_until_lane_end(distance_bwn_waypoints)

        # select waypoints from the list with some randomness
        selected_waypoints = []
        ptr = 0
        for _ in range(config["total_non_ego_vehicles"]+1):
            ptr += np.random.randint(config["min_dist_bwn_veh"], config["max_dist_bwn_veh"])
            selected_waypoints.append(waypoint_list[ptr])
            ptr += config["average_car_length"]

        # convert waypoints to spawn points
        spawn_points_list = [item.transform for item in selected_waypoints]
        for point in spawn_points_list: point.location.z += 0.2 # prevent collision

        # select the position of the ego vehicle
        ego_veh_int = np.random.randint(1, config["total_non_ego_vehicles"])
        temp = spawn_points_list.pop(ego_veh_int)
        ego_spawn_point = (temp, temp.get_forward_vector())
        
        # non-ego vehicle spawn points
        non_ego_spawn_points = [(spawn_points_list[i], spawn_points_list[i].get_forward_vector()) \
                                for i in range(config["total_non_ego_vehicles"])]
        # vectors = [x.get_forward_vector() for x in spawn_points_list[1:5]]
        

        return ego_spawn_point, non_ego_spawn_points

    def spawn(self, ego_spawn_point, non_ego_spawn_points):
        self.destroy_all_actors()

        # Get vehicle blueprints, mustang for non-ego, tesla for ego :D
        non_ego_vehicle_blueprint = self.world.get_blueprint_library().filter('vehicle.mustang.mustang')[0]
        ego_vehicle_blueprint = self.world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
        
        # defining a few commands
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor
        ApplyVelocity = carla.command.ApplyVelocity

        # spawning non-ego vehicles
        batch = []
        for i in range(self.config["total_non_ego_vehicles"]):
            batch.append(SpawnActor(non_ego_vehicle_blueprint, non_ego_spawn_points[i][0])
                .then(ApplyVelocity(FutureActor, non_ego_spawn_points[i][1] * self.target_speed))
                .then(SetAutopilot(FutureActor, True, 8000)))
        non_ego_vehicle_list = []
        for response in self.client.apply_batch_sync(batch, False):
            if response.error:
                print(response.error)
            else:
                non_ego_vehicle_list.append(self.world.get_actor(response.actor_id))
        
        # spawning ego vehicle
        ego_batch = []
        ego_batch.append(SpawnActor(ego_vehicle_blueprint, ego_spawn_point[0])
                .then(ApplyVelocity(FutureActor, ego_spawn_point[1] * self.target_speed))
                .then(SetAutopilot(FutureActor, True, 8000)))
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
            self.traffic_manager.ignore_lights_percentage(actor, 100)
            self.traffic_manager.ignore_signs_percentage(actor, 100)
            actor.set_simulate_physics(False)

        self.ego_vehicle = ego_vehicle
        self.non_ego_vehicle_list = non_ego_vehicle_list

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

    def reset(self, warm_start=True, warm_start_duration=2):
        # reset camera view
        self.spectator.set_transform(spectator_trans)

        # configure asynchronous mode
        self.configure_world(self.no_rendering_mode,self.synchronous_mode,self.fixed_delta_seconds)
        self.configure_traffic_manager()
        self.tick()

        # destroy any remaining actors
        self.destroy_all_actors()

        # spawn vehicles
        ego_spawn_point, non_ego_spawn_points = self.get_spawn_points( \
                                                    self.config["distance_bwn_waypoints"])
        self.spawn(ego_spawn_point, non_ego_spawn_points)
        self.tick()

        # warm start
        if warm_start: self.warm_start(warm_start_duration)

        # disable autopilot for ego vehicle
        self.disable_autopilot()

        return self.ego_vehicle, self.non_ego_vehicle_list, self.target_speed



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
    


