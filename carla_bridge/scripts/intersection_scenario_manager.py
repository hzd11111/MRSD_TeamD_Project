import sys
import random
import time

import carla

sys.path.append("../../carla_bridge/scripts/cartesian_to_frenet")
from topology_extraction import (
    get_opendrive_tree,
    get_junction_topology,
    get_junction_roads_topology,
)
from utils import get_intersection_topology, get_full_lanes


class IntersectionScenario:
    def __init__(self, client) -> None:

        self.client = client
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.traffic_manager.set_global_distance_to_leading_vehicle(2)

        self.world = self.client.get_world()
        self.vehicles_list = []
        print("Intersection Manager Initialized...")

        ### Get the topology
        tree = get_opendrive_tree(self.world)
        self.junction_topology = get_junction_topology(tree)
        self.road_topology = get_junction_roads_topology(tree)

    def reset(
        self,
        warm_start_duration=5,
        num_vehicles=10,
        junction_id=None,
        follow_traffic_rules=True,
    ):

        self.vehicles_list = []
        self.walkers_list = []

        if junction_id is None:
            junction_id, _ = random.choice(
                [53]  # , 905, 599, 965]
            )  # random.choice(list(self.junction_topology.items()))
        print("generating scenario at junction id: ", junction_id)

        incoming_road_lane_id_set = set([])
        outgoing_road_lane_id_set = set([])

        incoming_road_lane_id_to_outgoing_lane_id_dict = {}
        road_id_set = set([])

        for idx in range(len(self.junction_topology[junction_id])):
            intersection_road_id, _ = self.junction_topology[junction_id][idx][1]
            road_1_id, road_2_id, lane_connections = self.road_topology[
                intersection_road_id
            ]
            road_id_set.add(road_1_id)
            road_id_set.add(road_2_id)

            for intersection_connection_lanes in lane_connections:

                direction = intersection_connection_lanes[0]
                lane_1_id = intersection_connection_lanes[1]
                lane_2_id = intersection_connection_lanes[-1]

                used_intersection_lane_IDs = []
                for i in range(2, len(intersection_connection_lanes) - 1):
                    if intersection_connection_lanes[i] in used_intersection_lane_IDs:
                        continue
                    used_intersection_lane_IDs.append(intersection_connection_lanes[i])

                if direction == "forward":
                    incoming_road_lane_id_set.add((road_1_id, lane_1_id))
                    outgoing_road_lane_id_set.add((road_2_id, lane_2_id))
                    if (
                        road_1_id,
                        lane_1_id,
                    ) not in incoming_road_lane_id_to_outgoing_lane_id_dict:
                        incoming_road_lane_id_to_outgoing_lane_id_dict[
                            (road_1_id, lane_1_id)
                        ] = [
                            (
                                road_2_id,
                                lane_2_id,
                                intersection_road_id,
                                tuple(used_intersection_lane_IDs),
                            )
                        ]
                    else:
                        incoming_road_lane_id_to_outgoing_lane_id_dict[
                            (road_1_id, lane_1_id)
                        ].append(
                            (
                                road_2_id,
                                lane_2_id,
                                intersection_road_id,
                                tuple(used_intersection_lane_IDs),
                            )
                        )
                else:
                    incoming_road_lane_id_set.add((road_2_id, lane_2_id))
                    outgoing_road_lane_id_set.add((road_1_id, lane_1_id))
                    if (
                        road_2_id,
                        lane_2_id,
                    ) not in incoming_road_lane_id_to_outgoing_lane_id_dict:
                        incoming_road_lane_id_to_outgoing_lane_id_dict[
                            (road_2_id, lane_2_id)
                        ] = [
                            (
                                road_1_id,
                                lane_1_id,
                                intersection_road_id,
                                tuple(used_intersection_lane_IDs),
                            )
                        ]
                    else:
                        incoming_road_lane_id_to_outgoing_lane_id_dict[
                            (road_2_id, lane_2_id)
                        ].append(
                            (
                                road_1_id,
                                lane_1_id,
                                intersection_road_id,
                                tuple(used_intersection_lane_IDs),
                            )
                        )

        for key in incoming_road_lane_id_to_outgoing_lane_id_dict.keys():
            incoming_road_lane_id_to_outgoing_lane_id_dict[key] = list(
                set(incoming_road_lane_id_to_outgoing_lane_id_dict[key])
            )

        synchronous_master = True

        # settings = self.world.get_settings()
        self.traffic_manager.set_synchronous_mode(True)
        # if not settings.synchronous_mode:
        #     synchronous_master = True
        #     settings.synchronous_mode = True
        #     settings.fixed_delta_seconds = 0.05
        #     self.world.apply_settings(settings)
        # else:
        #     synchronous_master = False

        blueprints = self.world.get_blueprint_library().filter("vehicle.*")

        blueprints = [
            x for x in blueprints if int(x.get_attribute("number_of_wheels")) == 4
        ]
        blueprints = [x for x in blueprints if not x.id.endswith("isetta")]
        blueprints = [x for x in blueprints if not x.id.endswith("carlacola")]
        blueprints = [x for x in blueprints if not x.id.endswith("cybertruck")]
        blueprints = [x for x in blueprints if not x.id.endswith("t2")]

        waypoints = self.world.get_map().generate_waypoints(distance=4)
        road_waypoints = []
        for waypoint in waypoints:
            if waypoint.road_id in road_id_set:
                road_waypoints.append(waypoint)
        number_of_spawn_points = len(road_waypoints)

        ego_road_waypoints = []
        for waypoint in waypoints:
            if (waypoint.road_id, waypoint.lane_id) in incoming_road_lane_id_set:
                ego_road_waypoints.append(waypoint)
        random.shuffle(ego_road_waypoints)

        if num_vehicles < number_of_spawn_points:
            random.shuffle(road_waypoints)
        elif num_vehicles > number_of_spawn_points:
            num_vehicles = number_of_spawn_points

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        ego_batch = []

        # Ego Vehicle
        for n, t in enumerate(ego_road_waypoints):
            if n >= 1:
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
            transform = t.transform
            transform.location.z += 2.0
            ego_batch.append(
                SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True))
                # SpawnActor(blueprint, transform)
            )

        for n, t in enumerate(road_waypoints):
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
            transform = t.transform
            transform.location.z += 2.0
            batch.append(
                SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True))
                # SpawnActor(blueprint, transform)
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
        waypoints = self.world.get_map().generate_waypoints(distance=1)

        ego_key = (ego_road_waypoints[0].road_id, ego_road_waypoints[0].lane_id)

        intersection_topology, road_lane_to_orientation = get_intersection_topology(
            waypoints,
            incoming_road_lane_id_set,
            outgoing_road_lane_id_set,
            junction_id,
            ego_key,
        )

        # Adding ego key to parallel lane same dir list
        intersection_topology[2].append(ego_key)
        intersection_topology = get_full_lanes(
            intersection_topology[0],
            intersection_topology[1],
            intersection_topology[2],
            intersection_topology[3],
            incoming_road_lane_id_to_outgoing_lane_id_dict,
        )

        # for n, v in enumerate(my_vehicles):

        #     self.traffic_manager.auto_lane_change(v, False)
        #     if follow_traffic_rules is not True:
        #         print("breaking traffic rules")
        #         self.traffic_manager.auto_lane_change(v, True)
        #         self.traffic_manager.ignore_lights_percentage(v, 100)
        #         self.traffic_manager.distance_to_leading_vehicle(v, 1)

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

        print("Control handed to system....")

        connecting_elem = incoming_road_lane_id_to_outgoing_lane_id_dict[ego_key][1]

        connection_key = (connecting_elem[2], connecting_elem[3][0])
        next_road_key = (connecting_elem[0], connecting_elem[1])

        ego_wps = [
            wp
            for wp in waypoints
            if wp.road_id == ego_key[0] and wp.lane_id == ego_key[1]
        ]
        connection_wps = [
            wp
            for wp in waypoints
            if wp.road_id == connection_key[0] and wp.lane_id == connection_key[1]
        ]
        next_wps = [
            wp
            for wp in waypoints
            if wp.road_id == next_road_key[0] and wp.lane_id == next_road_key[1]
        ]

        global_path_wps = ego_wps + connection_wps + next_wps

        return (
            ego_vehicle,
            my_vehicles,
            incoming_road_lane_id_to_outgoing_lane_id_dict,
            intersection_topology,
            ego_key,
            global_path_wps,
            road_lane_to_orientation,
        )