import sys
import random
import time

import carla

sys.path.append("../../global_route_planner")
from global_planner import get_global_planner


class P2PScenario:
    def __init__(self, client) -> None:

        self.client = client

        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.traffic_manager.set_global_distance_to_leading_vehicle(2)

        self.world = self.client.get_world()
        self.vehicles_list = []
        print("Intersection Manager Initialized...")

        # Global Planner
        self.global_planner = get_global_planner(self.world, 1)

    def reset(self, warm_start_duration=5, num_vehicles=50):
        self.vehicles_list = []

        synchronous_master = True
        self.traffic_manager.set_synchronous_mode(True)

        blueprints = self.world.get_blueprint_library().filter("vehicle.*")

        blueprints = [
            x for x in blueprints if int(x.get_attribute("number_of_wheels")) == 4
        ]
        blueprints = [x for x in blueprints if not x.id.endswith("isetta")]
        blueprints = [x for x in blueprints if not x.id.endswith("carlacola")]
        blueprints = [x for x in blueprints if not x.id.endswith("t2")]
        blueprints = [x for x in blueprints if not x.id.endswith("police")]

        ego_blueprints = [x for x in blueprints if x.id.endswith("model3")]

        spawn_points = self.world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if num_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif num_vehicles > number_of_spawn_points:
            num_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        ego_batch = []

        # Ego Vehicle
        for n, transform in enumerate(spawn_points):
            if n >= 1:
                break
            blueprint = random.choice(ego_blueprints)
            if blueprint.has_attribute("color"):
                color = random.choice(
                    blueprint.get_attribute("color").recommended_values
                )
                blueprint.set_attribute("color", color)
            if blueprint.has_attribute("driver_id"):
                driver_id = random.choice(
                    blueprint.get_attribute("driver_id").recommended_values
                )
                blueprint.set_attribute("driver_id", driver_id)
            blueprint.set_attribute("role_name", "autopilot")
            # transform = t.transform
            transform.location.z += 2.0
            ego_batch.append(
                SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True))
            )

        for n, transform in enumerate(spawn_points):
            if n == 0:
                continue
            if n >= num_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute("color"):
                color = random.choice(
                    blueprint.get_attribute("color").recommended_values
                )
                blueprint.set_attribute("color", color)
            if blueprint.has_attribute("driver_id"):
                driver_id = random.choice(
                    blueprint.get_attribute("driver_id").recommended_values
                )
                blueprint.set_attribute("driver_id", driver_id)
            blueprint.set_attribute("role_name", "autopilot")
            # transform = t.transform
            transform.location.z += 2.0
            batch.append(
                SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True))
            )

        ego_vehicle_id = None
        for response in self.client.apply_batch_sync(ego_batch, synchronous_master):
            if response.error:
                print("Response Error while applying ego batch!")
            else:
                # self.vehicles_list.append(response.actor_id)
                ego_vehicle_id = response.actor_id

        for response in self.client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                print("Response Error while applying batch!")
            else:
                self.vehicles_list.append(response.actor_id)

        my_vehicles = self.world.get_actors(self.vehicles_list)
        ego_vehicle = self.world.get_actors([ego_vehicle_id])[0]

        for n, v in enumerate(my_vehicles):

            self.traffic_manager.auto_lane_change(v, False)

        warm_start_curr = 0
        while warm_start_curr < warm_start_duration:
            warm_start_curr += 0.1
            if synchronous_master:
                self.world.tick()
            else:
                self.world.wait_for_tick()

        self.client.apply_batch_sync(
            [SetAutopilot(ego_vehicle, False)], synchronous_master
        )

        ego_vehicle_location = ego_vehicle.get_transform().location

        end_point_location = random.choice(
            self.world.get_map().get_spawn_points()
        ).location

        route = self.global_planner.trace_route(
            ego_vehicle_location, end_point_location
        )
        global_path_wps = [route[i][0] for i in range(len(route))]

        print("Control handed to system....")

        return ego_vehicle, my_vehicles, global_path_wps, route