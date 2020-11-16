import sys
import random
import time
import numpy as np

import carla
from agents.navigation.local_planner import RoadOption

sys.path.append("../../global_route_planner")
from global_planner import get_global_planner, draw_waypoints, destroy_all_actors

sys.path.append("../../carla_bridge/scripts/cartesian_to_frenet")
from topology_extraction import (
    get_opendrive_tree,
    get_junction_topology,
    get_junction_roads_topology,
)
from utils import get_intersection_topology, get_full_lanes
from options import Scenario, GlobalPathAction
from settings import TEST_ROUTE, CURRENT_SCENARIO

# if(TEST_ROUTE == 9 and CURRENT_SCENARIO == Scenario.P2P):
#     np.random.seed(22)
#     random.seed(22)
    
# if(TEST_ROUTE == 5 and CURRENT_SCENARIO == Scenario.P2P):
#     print("Starting route 5....")
#     np.random.seed(0)
#     random.seed(0)
    
# if(TEST_ROUTE == 7 and CURRENT_SCENARIO == Scenario.P2P):
#     print("Starting route 7....")
#     np.random.seed(0)
#     random.seed(0)
    
# if(TEST_ROUTE == 8 and CURRENT_SCENARIO == Scenario.P2P):
#     print("Starting route 8....")
#     np.random.seed(0)
#     random.seed(0)
    
    
'''
LF - Lane Follow, RT/LT - Right/Left Turn, LCR/LCL - Lane chane right/left
GS - Go straight
Description of routes:Lane Following is involved in all of them
0. Left turn at an intersection
1. Go straight intersection
2. Right turn intersection
3. Lane change and right turn
4. Left Lane Change
5. Lane Fol - Go Str - Lane Fol - Left Turn - Lane Follow
6. Lane Ch Right - Lane Fol - Right Turn - Left Change Lane - Left Turn
7. (Bug free lanes) LF - GS - LCL - LF - LT - RT - LF - RT
8. (Bug free lanes) LCR - LF - RT - LF - GS - LCL - LT - LF
9. (GS - LT - RLC - RT - LLC)
10. LF - LT - LF - GS - LCR
11. RT - LF - LT - LF
12. RT - LF - RT
'''
route_start_locations = [(-47.5,-18.8,0.06) ,(-66.5,-91.5,0), (-66.5,-95,0), (-131.7,-70.3,0), \
                        (-125.1,-17.9,0), (-47, 52, 0), (-47.5,-13.8,0), (96,37,0), (29,-130,0), (-220,3.08,0),(-91.22312927246094, 151.28395080566406, 0.055450439453125), (24.58, 68.39, 0),(26.79, -106.88, 0)]
route_end_locations = [(-92.1,-91.5,0), (-167.1,-91.6,0), (-120.9,-120.970520,0), (-128.6,-18.8,0), \
                        (-121.2,-69.9,0), (-95,-91.5,0), (34,-124,0), (-72,140,0), (-128,-49,-0), (-74.20005798339844, -88.00556182861328, 0.0),(-163.62777709960938, 84.37947082519531, 0.0), (-81.98, 143.73, 0), (-73.24, -138.42, 0)]

class P2PScenario:
    def __init__(self, client) -> None:

        self.client = client

        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.traffic_manager.set_global_distance_to_leading_vehicle(2)
        
        self.world = self.client.get_world()
        self.vehicles_list = []
        print("Intersection Manager Initialized...")

        # Global Planner
        self.global_planner = get_global_planner(self.world, 2)
        
        
        
        # Intersection information pre-computation
        ## Get the topology
        tree = get_opendrive_tree(self.world)
        self.junction_topology = get_junction_topology(tree)
        self.road_topology = get_junction_roads_topology(tree)
        self.global_planner = get_global_planner(self.world, planner_resolution=1)

        ## Get map waypoints
        self.waypoints_finer = self.world.get_map().generate_waypoints(distance=1)
        
        

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
        blueprints = [x for x in blueprints if not x.id.endswith("cybertruck")]

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

        route, global_path_wps = self.get_random_route()
        # draw_waypoints(self.world, global_path_wps, life_time=100)

        spawn_waypoint = global_path_wps[0]
        #spawn_waypoint = spawn_waypoint.previous(1)[0]
        # import pdb; pdb.set_trace()
        # Ego Vehicle
        for n, transform in enumerate([spawn_waypoint.transform]):
            if n >= 1:
                break
            blueprint = random.choice(ego_blueprints)
            if blueprint.has_attribute("color"):
                color = '255,0,0'
                # color = random.choice(
                #     blueprint.get_attribute("color").recommended_values
                # )
                blueprint.set_attribute("color", color)
            if blueprint.has_attribute("driver_id"):
                driver_id = random.choice(
                    blueprint.get_attribute("driver_id").recommended_values
                )
                blueprint.set_attribute("driver_id", driver_id)
            blueprint.set_attribute('role_name', 'ego')
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
                color = '0,0,0'
                # color = random.choice(
                #     blueprint.get_attribute("color").recommended_values
                # )
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
        print("Ego vehicle id: ------------------------------", ego_vehicle_id)
        for response in self.client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                print("Response Error while applying batch!")
            else:
                self.vehicles_list.append(response.actor_id)

        my_vehicles = self.world.get_actors(self.vehicles_list)
        ego_vehicle = self.world.get_actors([ego_vehicle_id])[0]

        for n, v in enumerate(my_vehicles):

            self.traffic_manager.auto_lane_change(v, False)
            self.traffic_manager.ignore_lights_percentage(v, 50)


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

        
        ### Pre-computation for intersections
        global_path_actions = self.get_global_path_actions(route)[:-2] # -2 to end the path slightly before the final point.
        
        ## Get the list of intersections that the global path will take in sequence. Also get the planned high level actions for each of these intersections.
        list_of_intersection_ids_to_pass_in_ordered_sequence = []
        list_of_high_level_actions_to_take_in_ordered_sequence = []
        first_wp_for_each_junction = []
        
        
        
        for global_path_point in global_path_wps[:-2]:
            if(global_path_point.is_junction and global_path_point.get_junction().id not in list_of_intersection_ids_to_pass_in_ordered_sequence):
                list_of_intersection_ids_to_pass_in_ordered_sequence.append(global_path_point.get_junction().id)
                first_wp_for_each_junction.append(global_path_point)
                
        for global_path_action in global_path_actions:
            if(global_path_action == GlobalPathAction.GO_STRAIGHT or global_path_action == GlobalPathAction.RIGHT_TURN or global_path_action == GlobalPathAction.LEFT_TURN):
                list_of_high_level_actions_to_take_in_ordered_sequence.append(global_path_action)
                
        
        ## Find the road_id and lane_id on the lane through which the ego_vehicle will enter each intersection.
        road_and_lane_ids_for_incoming_roads_in_global_path_for_each_intersection = []
        for first_wp_in_junction in first_wp_for_each_junction:
            if first_wp_in_junction.next(2)[0].is_junction:
                if first_wp_in_junction.previous(2)[0].is_junction:
                    print("lol")
                previous_wp = first_wp_in_junction.previous(2)[0]
                road_id = previous_wp.road_id
                lane_id = previous_wp.lane_id
                road_and_lane_ids_for_incoming_roads_in_global_path_for_each_intersection.append((road_id, lane_id))
            else:
                if first_wp_in_junction.next(2)[0].is_junction:
                    print("lol-2")    
                next_wp = first_wp_in_junction.next(2)[0]
                road_id = next_wp.road_id
                lane_id = next_wp.lane_id            
                road_and_lane_ids_for_incoming_roads_in_global_path_for_each_intersection.append((road_id, lane_id))
                

        ## Get intersection topologies for each intersection accoring to the global path
        
        
        ### Precompute Stuff

        intersection_topologies = []
        road_lane_to_orientation_for_each_intersection = []
        incoming_road_lane_id_to_outgoing_lane_id_dict_for_each_intersection = []
        list_of_intersection_waypoints_for_each_intersection = []

        for i,junction in enumerate(list_of_intersection_ids_to_pass_in_ordered_sequence):

            (
                road_id_set,
                incoming_road_lane_id_set,
                outgoing_road_lane_id_set,
                incoming_road_lane_id_to_outgoing_lane_id_dict,
            ) = self.get_road_sets(junction)
            
            # Get intersection path waypoints
            current_scenario_setups = self.get_scenario_setups(self.waypoints_finer, junction, incoming_road_lane_id_set, outgoing_road_lane_id_set, incoming_road_lane_id_to_outgoing_lane_id_dict)[list_of_high_level_actions_to_take_in_ordered_sequence[i]]
            
            current_road_lane_id = road_and_lane_ids_for_incoming_roads_in_global_path_for_each_intersection[i]
            for setup in current_scenario_setups:
                if(setup[0] == current_road_lane_id):
                    current_setup = setup
                        
            incoming_road_lane = current_setup[0]
            outgoing_road_lane = current_setup[1]
            incoming_lane_waypoints = [
                wp
                for wp in self.waypoints_finer
                if wp.road_id == incoming_road_lane[0]
                and wp.lane_id == incoming_road_lane[1]
            ]
            incoming_lane_orientation = current_setup[2]
            if incoming_lane_orientation == 1:
                incoming_lane_waypoints = incoming_lane_waypoints[::-1]

            outgoing_lane_waypoints = [
                wp
                for wp in self.waypoints_finer
                if wp.road_id == outgoing_road_lane[0]
                and wp.lane_id == outgoing_road_lane[1]
            ]
            outgoing_lane_orientation = current_setup[3]
            if outgoing_lane_orientation == 1:
                outgoing_lane_waypoints = outgoing_lane_waypoints[::-1]

            start_location = incoming_lane_waypoints[-1].transform.location  # Start of lane
            end_location = outgoing_lane_waypoints[
                min(len(outgoing_lane_waypoints)-1,5)
            ].transform.location  # 10 m after end of intersection

            route = self.global_planner.trace_route(start_location, end_location)
            global_path_wps_for_intersection = [route[i][0] for i in range(len(route))]
            list_of_intersection_waypoints_for_each_intersection.append(global_path_wps_for_intersection)
            #################################################################################
            
            
            
            intersection_topology, road_lane_to_orientation = get_intersection_topology(
                self.waypoints_finer,
                incoming_road_lane_id_set,
                outgoing_road_lane_id_set,
                junction,
                road_and_lane_ids_for_incoming_roads_in_global_path_for_each_intersection[i],
            )      
            # Adding ego key to parallel lane same dir list
            intersection_topology[2].append(road_and_lane_ids_for_incoming_roads_in_global_path_for_each_intersection[i])
            intersection_topology = get_full_lanes(
                intersection_topology[0],
                intersection_topology[1],
                intersection_topology[2],
                intersection_topology[3],
                incoming_road_lane_id_to_outgoing_lane_id_dict,
            )
            
            intersection_topologies.append(intersection_topology)
            road_lane_to_orientation_for_each_intersection.append(road_lane_to_orientation)
            incoming_road_lane_id_to_outgoing_lane_id_dict_for_each_intersection.append(incoming_road_lane_id_to_outgoing_lane_id_dict)

        

        
        print("Control handed to system....")     
        

        return ego_vehicle, my_vehicles, global_path_wps, route, global_path_actions, intersection_topologies, incoming_road_lane_id_to_outgoing_lane_id_dict_for_each_intersection, road_lane_to_orientation_for_each_intersection, road_and_lane_ids_for_incoming_roads_in_global_path_for_each_intersection, list_of_intersection_waypoints_for_each_intersection, list_of_intersection_ids_to_pass_in_ordered_sequence

    def get_random_route(self):
        
        random_ind = np.random.randint(0, len(route_start_locations))
        random_ind = TEST_ROUTE  # TODO: REMOVE THIS LINE.
        start_location = carla.Location(*route_start_locations[random_ind])
        end_location = carla.Location(*route_end_locations[random_ind])

        grp = self.global_planner
        # Generate the route using the global route planner object.
        route = grp.trace_route(start_location, end_location)
        global_path_wps = [route[i][0] for i in range(len(route))]
        draw_waypoints(self.world, global_path_wps, life_time=30)

        return route, global_path_wps

    def destroy_all_actors(self):
        world = self.client.get_world()
        vehicles_list = world.get_actors().filter("vehicle*")
        for vehicle in vehicles_list:
            vehicle.destroy()
            
            
    def get_scenario_setups(
        self,
        waypoints,
        junction_id,
        incoming_road_lane_id_set,
        outgoing_road_lane_id_set,
        incoming_road_lane_id_to_outgoing_lane_id_dict,
        ):
        scenario_to_setup = {}
        scenario_to_setup[GlobalPathAction.LEFT_TURN] = []
        scenario_to_setup[GlobalPathAction.RIGHT_TURN] = []
        scenario_to_setup[GlobalPathAction.GO_STRAIGHT] = []

        for incoming_road_lane_id in incoming_road_lane_id_set:
            intersection_topology, road_lane_to_orientation = get_intersection_topology(
                waypoints,
                incoming_road_lane_id_set,
                outgoing_road_lane_id_set,
                junction_id,
                incoming_road_lane_id,
            )

            intersection_topology[2].append(incoming_road_lane_id)
            intersection_topology = get_full_lanes(
                intersection_topology[0],
                intersection_topology[1],
                intersection_topology[2],
                intersection_topology[3],
                incoming_road_lane_id_to_outgoing_lane_id_dict,
            )

            (
                full_intersecting_left,
                full_intersecting_right,
                full_parallel_same_dir,
                full_parallel_opposite_dir,
            ) = intersection_topology

            current_road_lane_connections = (
                incoming_road_lane_id_to_outgoing_lane_id_dict[incoming_road_lane_id]
            )

            for connection in current_road_lane_connections:
                outgoing_road_id = connection[0]
                outgoing_lane_id = connection[1]

                # Turn Left Scenario
                for full_lane in full_intersecting_left:
                    if full_lane[-1] == (outgoing_road_id, outgoing_lane_id):
                        scenario_to_setup[GlobalPathAction.LEFT_TURN].append(
                            [
                                incoming_road_lane_id,
                                full_lane[-1],
                                road_lane_to_orientation[incoming_road_lane_id][-1],
                                road_lane_to_orientation[full_lane[-1]][-1],
                            ]
                        )
                for full_lane in full_intersecting_right:
                    if full_lane[-1] == (outgoing_road_id, outgoing_lane_id):
                        scenario_to_setup[GlobalPathAction.RIGHT_TURN].append(
                            [
                                incoming_road_lane_id,
                                full_lane[-1],
                                road_lane_to_orientation[incoming_road_lane_id][-1],
                                road_lane_to_orientation[full_lane[-1]][-1],
                            ]
                        )
                for full_lane in full_parallel_same_dir:
                    if full_lane[-1] == (outgoing_road_id, outgoing_lane_id):
                        scenario_to_setup[GlobalPathAction.GO_STRAIGHT].append(
                            [
                                incoming_road_lane_id,
                                full_lane[-1],
                                road_lane_to_orientation[incoming_road_lane_id][-1],
                                road_lane_to_orientation[full_lane[-1]][-1],
                            ]
                        )

        return scenario_to_setup
    
    def get_road_sets(self, junction_id):

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

        return (
            road_id_set,
            incoming_road_lane_id_set,
            outgoing_road_lane_id_set,
            incoming_road_lane_id_to_outgoing_lane_id_dict,
        )
        
    def get_global_path_actions(self, route):

        global_path_actions = [route[i][1] for i in range(len(route))]

        for i in range(len(global_path_actions) - 1):
            if global_path_actions[i] == RoadOption.LANEFOLLOW:
                continue
            ct = 0
            curr_action = global_path_actions[i]
            while global_path_actions[i + ct + 1] == curr_action:
                ct += 1
                if i + ct + 1 >= len(global_path_actions):
                    break

            for j in range(i + 1, i + ct + 1):
                global_path_actions[j] = RoadOption.LANEFOLLOW

        for i in range(len(global_path_actions)):
            if global_path_actions[i] == RoadOption.LANEFOLLOW:
                global_path_actions[i] = GlobalPathAction.NO_ACTION
            elif global_path_actions[i] == RoadOption.LEFT:
                global_path_actions[i] = GlobalPathAction.LEFT_TURN
            elif global_path_actions[i] == RoadOption.RIGHT:
                global_path_actions[i] = GlobalPathAction.RIGHT_TURN
            elif global_path_actions[i] == RoadOption.STRAIGHT:
                global_path_actions[i] = GlobalPathAction.GO_STRAIGHT
            elif global_path_actions[i] == RoadOption.CHANGELANELEFT:
                global_path_actions[i] = GlobalPathAction.SWITCH_LANE_LEFT
            elif global_path_actions[i] == RoadOption.CHANGELANERIGHT:
                global_path_actions[i] = GlobalPathAction.SWITCH_LANE_RIGHT

        return global_path_actions
            

if __name__ == "__main__":
    
    client = carla.Client("localhost", 2000)
    world = client.get_world()
    client.set_timeout(10.0)

    p2p_tm = P2PScenario(client)
    p2p_tm.destroy_all_actors()
    ego_vehicle, my_vehicles, global_path_wps, route = p2p_tm.reset()
    draw_waypoints(world, global_path_wps, life_time=30)
    for i in range(100): world.tick(); time.sleep(0.1)
    import ipdb; ipdb.set_trace()