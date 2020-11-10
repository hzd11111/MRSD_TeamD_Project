#!/usr/bin/env python

""" Implementation of the CarlaHandler class. CarlaHandler class provides some custom built APIs for Carla. """

__author__ = "Mayank Singal"
__maintainer__ = "Mayank Singal"
__email__ = "mayanksi@andrew.cmu.edu"
__version__ = "0.1"


from builtins import isinstance
from builtins import list
from os import get_inheritable
import random
import time
import math
import sys
from collections import defaultdict
from typing import Text


import numpy as np
import carla

from utils import get_matrix, create_bb_points
from enum import Enum
import re

sys.path.append("../../carla_utils/utils")
from functional_utility import Pose2D, Frenet
from utility import LanePoint
from actors import Actor, Vehicle, Pedestrian
from options import StopLineStatus


class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.
    """

    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


def find_weather_presets():
    rgx = re.compile(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)")
    name = lambda x: " ".join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match("[A-Z].+", x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


class CarlaHandler:
    def __init__(self, client):

        self.client = client  # TODO: Is this needed?
        self.world = client.get_world()
        self.world_map = self.world.get_map()
        self.all_waypoints = self.get_waypoints()
        self.blueprint_library = self.world.get_blueprint_library()
        self.actor_dict = {}

        self.world.set_weather(find_weather_presets()[2][0])

        print("Handler Initialized!\n")

    def __del__(self):
        self.destroy_actors()
        print("Handler destroyed..\n")

    def destroy_actors(self):
        for actor in self.world.get_actors():
            if actor.id in self.actor_dict:
                actor.destroy()
        print("All actors destroyed..\n")

    def get_waypoints(self, distance=10):

        return self.world_map.generate_waypoints(distance=distance)

    def filter_waypoints(self, waypoints, road_id=None, lane_id=None):

        filtered_waypoints = []
        for waypoint in waypoints:
            if lane_id == None:
                if waypoint.road_id == road_id:
                    filtered_waypoints.append(waypoint)
            else:
                if waypoint.road_id == road_id and waypoint.lane_id == lane_id:
                    filtered_waypoints.append(waypoint)

        return filtered_waypoints

    def draw_waypoints(
        self, waypoints, road_id=None, section_id=None, life_time=50.0, 
        color=False, text="O"):
        if color:
            b = 255
        else:
            b = 0

        for waypoint in waypoints:

            if waypoint.road_id == road_id or road_id == None:
                self.world.debug.draw_string(
                    waypoint.transform.location,
                    text,
                    draw_shadow=False,
                    color=carla.Color(r=0, g=255, b=b),
                    life_time=life_time,
                    persistent_lines=True,
                )
            if waypoint.section_id == section_id:
                self.world.debug.draw_string(
                    waypoint.transform.location,
                    "O",
                    draw_shadow=False,
                    color=carla.Color(r=0, g=255, b=b),
                    life_time=life_time,
                    persistent_lines=True,
                )

    def _retrieve_options(self, list_waypoints, current_waypoint):
        """
        Compute the type of connection between the current active waypoint and the multiple waypoints present in
        list_waypoints. The result is encoded as a list of RoadOption enums.
        :param list_waypoints: list with the possible target waypoints in case of multiple options
        :param current_waypoint: current active waypoint
        :return: list of RoadOption enums representing the type of connection from the active waypoint to each
                candidate in list_waypoints
        """
        options = []
        for next_waypoint in list_waypoints:
            # this is needed because something we are linking to
            # the beggining of an intersection, therefore the
            # variation in angle is small
            next_next_waypoint = next_waypoint.next(3.0)[0]
            link = self._compute_connection(current_waypoint, next_next_waypoint)
            options.append(link)

        return options

    def _compute_connection(self, current_waypoint, next_waypoint, threshold=10):
        """
        Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
        (next_waypoint).
        :param current_waypoint: active waypoint
        :param next_waypoint: target waypoint
        :return: the type of topological connection encoded as a RoadOption enum:
                RoadOption.STRAIGHT
                RoadOption.LEFT
                RoadOption.RIGHT
        """
        n = next_waypoint.transform.rotation.yaw
        n = n % 360.0

        c = current_waypoint.transform.rotation.yaw
        c = c % 360.0

        diff_angle = (n - c) % 180.0
        if diff_angle < threshold or diff_angle > (180 - threshold):
            return RoadOption.STRAIGHT
        elif diff_angle > 90.0:
            return RoadOption.LEFT
        else:
            return RoadOption.RIGHT

    def convert_global_transform_to_actor_frame(self, actor=None, transform=None):

        if actor == None or transform == None:
            print("Input is None. Please Check")
            return None
        else:

            actor_to_world_transform = actor.get_transform()
            R_actor_to_world = get_matrix(actor_to_world_transform)
            R_world_to_actor = np.linalg.inv(R_actor_to_world)

            transform_coords = np.zeros((4, 1))
            transform_coords[0] = transform.location.x
            transform_coords[1] = transform.location.y
            transform_coords[2] = transform.location.z
            transform_coords[3] = 1

            transform_position_as_seen_from_actor = np.dot(
                R_world_to_actor, transform_coords
            )

            return transform_position_as_seen_from_actor

    def get_pedestrian_information(self, ego_vehicle=None):
        pedestrian_list = []

        ego_vehicle_location = ego_vehicle.get_location()
        nearest_waypoint = self.world_map.get_waypoint(
            ego_vehicle_location, project_to_road=True
        )

        # Get current road and lane IDs
        current_road_ID = nearest_waypoint.road_id

        for actor in self.world.get_actors().filter("walker.*"):
            actor_nearest_waypoint = self.world_map.get_waypoint(
                actor.get_location(), project_to_road=True
            )
            if actor_nearest_waypoint.road_id == current_road_ID:
                pedestrian_list.append(actor)

        return pedestrian_list

    def get_next_waypoints(self, last_waypoint, rev=False, k=100):

        if last_waypoint == None:
            return []

        sampling_radius = 1
        full_waypoints = []
        for i in range(k):
            if rev == False:
                next_waypoints = last_waypoint.next(sampling_radius)
            else:
                next_waypoints = last_waypoint.previous(sampling_radius)

            if len(next_waypoints) == 0:
                break
            elif len(next_waypoints) == 1:
                # only one option available ==> lanefollowing
                next_waypoint = next_waypoints[0]
                road_option = RoadOption.LANEFOLLOW
            else:
                # random choice between the possible options
                road_options_list = self._retrieve_options(
                    next_waypoints, last_waypoint
                )
                road_option = random.choice(road_options_list)

                if RoadOption.STRAIGHT in road_options_list:
                    next_waypoint = next_waypoints[
                        road_options_list.index(RoadOption.STRAIGHT)
                    ]
                else:
                    next_waypoint = next_waypoints[road_options_list.index(road_option)]
            full_waypoints.append(next_waypoint)
            # curr_waypoint = next_waypoints[-1]

            last_waypoint = next_waypoint

        return full_waypoints

    def waypoint_to_pose2D(self, waypoint):

        x = waypoint.transform.location.x
        y = waypoint.transform.location.y
        theta = waypoint.transform.rotation.yaw * np.pi / 180

        return Pose2D(x=x, y=y, theta=theta)

    def get_direction_information(
        self,
        road_id,
        lane_id,
        ego_info,
        in_out_dict,
        same_dir,
        intersecting_left,
        intersecting_right,
        parallel_same_dir,
        parallel_opposite_dir,
        ):
        right_turning_lane = False
        left_turning_lane = False
        right_most_turning_lane = False
        left_to_the_current = False
        right_next_to_the_current = False
        lanes_to_right = intersecting_right if same_dir else intersecting_left
        lanes_to_left = intersecting_left if same_dir else intersecting_right

        for connection in in_out_dict[(road_id, lane_id)]:
            # print((connection[0], connection[1]))
            for right_lanes in lanes_to_right:
                if (connection[0], connection[1]) in right_lanes:
                    right_turning_lane = True

            for left_lanes in lanes_to_left:
                if (connection[0], connection[1]) in left_lanes:
                    left_turning_lane = True

        # determine if the lane is right_most
        lane_group = parallel_same_dir if same_dir else parallel_opposite_dir
        maximum_id_magnitude = 0
        for lane in lane_group:
            maximum_id_magnitude = max(abs(lane[0][1]), maximum_id_magnitude)
        right_most_turning_lane = maximum_id_magnitude == abs(lane_id)

        ego_lane = ego_info[1]
        ego_road = [
            parallel_same_dir[0][0][0],
            parallel_same_dir[0][1][0],
            parallel_same_dir[0][2][0],
        ]
        if road_id in ego_road:
            if not same_dir:
                left_to_the_current = (
                    True  # which could means right_to_the_current = False
                )
                # since this function should only be applied on (road_id, lane_id) in in_out_dict.keys()
                # update all the fields in connecting lanes if needed
                # update_field_in_group(road_id, lane_id, parallel_opposite_dir, left_to_the_current, True)
                if abs(ego_lane) == 1 and lane_id * ego_lane == -1:
                    right_next_to_the_current = True
                    # update_field_in_group(road_id, lane_id, parallel_opposite_dir, right_next_to_the_current, True)

            else:
                if abs(lane_id) < abs(ego_lane):
                    left_to_the_current = (
                        True  # on the same road but with opposite direction
                    )
                    # update_field_in_group(road_id, lane_id, parallel_opposite_dir, left_to_the_current, True)
                if abs(ego_lane - lane_id) == 1:
                    right_next_to_the_current = True
                    # update_field_in_group(road_id, lane_id, parallel_opposite_dir, right_next_to_the_current, True)

        return (
            left_turning_lane,
            right_turning_lane,
            left_to_the_current,
            right_next_to_the_current,
        )

    def get_positional_booleans(
        self, road_lane_list, ego_info, in_out_dict, intersection_topology, same_dir
        ):

        road_lane = None

        for key in in_out_dict:
            if key in road_lane_list:
                road_lane = key
                break

        (
            intersecting_left,
            intersecting_right,
            parallel_same_dir,
            parallel_opposite_dir,
        ) = intersection_topology
        return self.get_direction_information(
            road_lane[0],
            road_lane[1],
            ego_info,
            in_out_dict,
            same_dir,
            intersecting_left,
            intersecting_right,
            parallel_same_dir,
            parallel_opposite_dir,
        )

    def get_lane_waypoints(self, road_lane_collection, road_lane_to_orientation):

        incoming_road = road_lane_collection[0][0]
        incoming_lane = road_lane_collection[0][1]

        connecting_road = road_lane_collection[1][0]
        connecting_lane = road_lane_collection[1][1]

        outgoing_road = road_lane_collection[2][0]
        outgoing_lane = road_lane_collection[2][1]

        incoming_waypoints = self.filter_waypoints(
            self.all_waypoints, incoming_road, incoming_lane
        )
        connecting_waypoints = self.filter_waypoints(
            self.all_waypoints, connecting_road, connecting_lane
        )
        outgoing_waypoints = self.filter_waypoints(
            self.all_waypoints, outgoing_road, outgoing_lane
        )
        if (
            road_lane_to_orientation[(incoming_road, incoming_lane)][-1] == 0
        ):  # 0 : Starting from near junction
            incoming_waypoints = incoming_waypoints[::-1]

        if (
            road_lane_to_orientation[(outgoing_road, outgoing_lane)][-1] == 1
        ):  # 1 : Ending at junction
            outgoing_waypoints = outgoing_waypoints[::-1]

        first_connecting_waypoint = connecting_waypoints[0]
        last_connecting_waypoint = connecting_waypoints[-1]

        last_incoming_waypoint = incoming_waypoints[-1]

        dist1 = first_connecting_waypoint.transform.location.distance(
            last_incoming_waypoint.transform.location
        )
        dist2 = last_connecting_waypoint.transform.location.distance(
            last_incoming_waypoint.transform.location
        )

        if dist1 > dist2:
            connecting_waypoints = connecting_waypoints[::-1]

        return (
            incoming_waypoints + connecting_waypoints[1:-1] + outgoing_waypoints,
            len(incoming_waypoints),
            len(connecting_waypoints[1:-1]),
        )

    def get_lane_info(
        self,
        all_vehicles,
        lane_list,
        ego_road_lane_ID_pair=None,
        road_lane_to_orientation=None,
        ):

        full_info = []
        ego_lane_info = []

        ### Misc Info
        allocated_actor_ids = []

        road_lane_to_vehicle_id = defaultdict(lambda: [])

        for vehicle in all_vehicles:
            vehicle_nearest_waypoint = self.world_map.get_waypoint(
                vehicle.get_location(), project_to_road=True
            )
            # if(vehicle_nearest_waypoint.is_junction):
            #     continue
            key = (vehicle_nearest_waypoint.road_id, vehicle_nearest_waypoint.lane_id)
            if key in road_lane_to_vehicle_id:
                road_lane_to_vehicle_id[key].append(vehicle.id)
            else:
                road_lane_to_vehicle_id[key] = [vehicle.id]

        for elem in lane_list:

            this_connection_road_lanes = []

            if (
                len(elem) == 2
            ):  ## This is for 3 way intersections. TODO: Enable get_lane_waypoints for these.
                this_connection_waypoints = self.filter_waypoints(
                    self.all_waypoints, elem[0], elem[1]
                )
                this_connection_actors = road_lane_to_vehicle_id[(elem[0], elem[1])]
                this_connection_road_lanes.append(tuple(elem))

            else:

                ### Add road and lane_ids
                this_connection_road_lanes.append((elem[0][0], elem[0][1]))
                this_connection_road_lanes.append((elem[1][0], elem[1][1]))
                this_connection_road_lanes.append((elem[2][0], elem[2][1]))

                ### Get Waypoints
                (
                    this_connection_waypoints,
                    length_incoming_section,
                    length_connecting_section,
                ) = self.get_lane_waypoints(elem, road_lane_to_orientation)
                this_connection_actors = []
                ### Get actors

                this_connection_actors.extend(
                    road_lane_to_vehicle_id[(elem[0][0], elem[0][1])]
                )
                this_connection_actors.extend(
                    road_lane_to_vehicle_id[(elem[1][0], elem[1][1])]
                )
                this_connection_actors.extend(
                    road_lane_to_vehicle_id[(elem[2][0], elem[2][1])]
                )

                ### Convert actors to custom objects
                this_connection_actors = [
                    Vehicle(self.world, vehicle_id)
                    for vehicle_id in this_connection_actors
                ]

                ## Keep a list of all allocated actor ids. Unused actor ids will be sent in an additional misc_lane.
                for actor in this_connection_actors:
                    allocated_actor_ids.append(actor.actor_id)

                ### Convert waypoints to custom objects
                this_connection_waypoints = [
                    LanePoint(global_pose=self.waypoint_to_pose2D(wp))
                    for wp in this_connection_waypoints
                ]

                ### Adding lane start and stop line information
                this_connection_waypoints[
                    length_incoming_section - 3
                ].stop_line = StopLineStatus.STRAIGHT_STOP

                this_connection_waypoints[
                    length_incoming_section + length_connecting_section - 1
                ].lane_start = True

            if ego_road_lane_ID_pair is not None and ego_road_lane_ID_pair in elem:
                ego_lane_info.append(
                    [
                        this_connection_actors,
                        this_connection_waypoints,
                        this_connection_road_lanes,
                    ]
                )
            else:
                full_info.append(
                    [
                        this_connection_actors,
                        this_connection_waypoints,
                        this_connection_road_lanes,
                    ]
                )

        return full_info, ego_lane_info, allocated_actor_ids

    def get_lane_info_only_actors(self, all_vehicles, lane_list, ego_road_lane_ID_pair):

        full_info = []
        ego_lane_info = []

        ### Misc Info
        allocated_actor_ids = []

        road_lane_to_vehicle_id = defaultdict(lambda: [])

        for vehicle in all_vehicles:
            vehicle_nearest_waypoint = self.world_map.get_waypoint(
                vehicle.get_location(), project_to_road=True
            )
            # if(vehicle_nearest_waypoint.is_junction):
            #     continue
            key = (vehicle_nearest_waypoint.road_id, vehicle_nearest_waypoint.lane_id)
            if key in road_lane_to_vehicle_id:
                road_lane_to_vehicle_id[key].append(vehicle.id)
            else:
                road_lane_to_vehicle_id[key] = [vehicle.id]

        for elem in lane_list:
            this_connection_road_lanes = []

            if (
                len(elem) == 2
            ):  ## This is for 3 way intersections. TODO: Enable get_lane_waypoints for these.
                this_connection_actors = road_lane_to_vehicle_id[(elem[0], elem[1])]
                this_connection_road_lanes.append(tuple(elem))

            else:
                ### Add road and lane_ids
                this_connection_road_lanes.append((elem[0][0], elem[0][1]))
                this_connection_road_lanes.append((elem[1][0], elem[1][1]))
                this_connection_road_lanes.append((elem[2][0], elem[2][1]))

                this_connection_actors = []
                ### Get actors

                this_connection_actors.extend(
                    road_lane_to_vehicle_id[(elem[0][0], elem[0][1])]
                )
                this_connection_actors.extend(
                    road_lane_to_vehicle_id[(elem[1][0], elem[1][1])]
                )
                this_connection_actors.extend(
                    road_lane_to_vehicle_id[(elem[2][0], elem[2][1])]
                )

                ### Convert actors to custom objects
                this_connection_actors = [
                    Vehicle(self.world, vehicle_id)
                    for vehicle_id in this_connection_actors
                ]

                ## Keep a list of all allocated actor ids. Unused actor ids will be sent in an additional misc_lane.
                for actor in this_connection_actors:
                    allocated_actor_ids.append(actor.actor_id)

            if ego_road_lane_ID_pair is not None and ego_road_lane_ID_pair in elem:
                ego_lane_info.append(
                    [this_connection_actors, [], this_connection_road_lanes]
                )
            else:
                full_info.append(
                    [this_connection_actors, [], this_connection_road_lanes]
                )

        return full_info, ego_lane_info, allocated_actor_ids
    
    def get_intersection_vehicles(self, all_vehicles, intersection_id, ego_vehicle_id):
        intersection_vehicles = []
        for vehicle in all_vehicles:
            vehicle_nearest_waypoint = self.world_map.get_waypoint(
                vehicle.get_location(), project_to_road=True
            )
            if(vehicle_nearest_waypoint.is_junction and vehicle_nearest_waypoint.get_junction().id == intersection_id and (vehicle.id != ego_vehicle_id)):
                intersection_vehicles.append(Vehicle(self.world, vehicle.id))
                
        return intersection_vehicles

    def get_state_information_intersection(
        self,
        ego_vehicle=None,
        all_vehicles=None,
        ego_road_lane_ID_pair=None,
        intersection_topology=None,
        road_lane_to_orientation=None,
        only_actors=False,
        intersection_id=53,
    ):

        (
            intersecting_left,
            intersecting_right,
            parallel_same_dir,
            parallel_opposite_dir,
        ) = intersection_topology

        if only_actors == False:
            (intersecting_left_info, _, allocated_left,) = self.get_lane_info(
                all_vehicles, intersecting_left, None, road_lane_to_orientation
            )
            (intersecting_right_info, _, allocated_right,) = self.get_lane_info(
                all_vehicles, intersecting_right, None, road_lane_to_orientation
            )
            (
                parallel_same_dir_info,
                ego_lane_info,
                allocated_parallel_same,
            ) = self.get_lane_info(
                all_vehicles,
                parallel_same_dir,
                ego_road_lane_ID_pair,
                road_lane_to_orientation,
            )
            (
                parallel_opposite_dir_info,
                _,
                allocated_parallel_opposite,
            ) = self.get_lane_info(
                all_vehicles, parallel_opposite_dir, None, road_lane_to_orientation
            )

            all_allocated_vehicle_ids = (
                allocated_left
                + allocated_right
                + allocated_parallel_same
                + allocated_parallel_opposite
            )
            unallocated_actor_ids = [
                vehicle.id
                for vehicle in all_vehicles
                if vehicle.id not in all_allocated_vehicle_ids
            ]
            # all_misc_vehicles = [
            #     Vehicle(self.world, vehicle_id) for vehicle_id in unallocated_actor_ids
            # ]
            all_misc_vehicles = self.get_intersection_vehicles(all_vehicles, intersection_id, ego_vehicle.id)
            all_misc_vehicles_ids = [v.actor_id for v in all_misc_vehicles]
            
            for i in range(len(intersecting_left_info)):
                intersecting_left_info[i][0] = [elem for elem in intersecting_left_info[i][0] if elem.actor_id not in all_misc_vehicles_ids]
            for i in range(len(intersecting_right_info)):
                intersecting_right_info[i][0] = [elem for elem in intersecting_right_info[i][0] if elem.actor_id not in all_misc_vehicles_ids]
            for i in range(len(parallel_same_dir_info)):
                parallel_same_dir_info[i][0] = [elem for elem in parallel_same_dir_info[i][0] if elem.actor_id not in all_misc_vehicles_ids]
            for i in range(len(parallel_opposite_dir_info)):
                parallel_opposite_dir_info[i][0] = [elem for elem in parallel_opposite_dir_info[i][0] if elem.actor_id not in all_misc_vehicles_ids]
            for i in range(len(ego_lane_info)):
                parallel_opposite_dir_info[i][0] = [elem for elem in parallel_opposite_dir_info[i][0] if elem.actor_id not in all_misc_vehicles_ids]

            return (
                intersecting_left_info,
                intersecting_right_info,
                parallel_same_dir_info,
                parallel_opposite_dir_info,
                ego_lane_info,
                all_misc_vehicles,
            )
        else:
            (
                intersecting_left_info,
                _,
                allocated_left,
            ) = self.get_lane_info_only_actors(all_vehicles, intersecting_left, None)
            (
                intersecting_right_info,
                _,
                allocated_right,
            ) = self.get_lane_info_only_actors(all_vehicles, intersecting_right, None)
            (
                parallel_same_dir_info,
                ego_lane_info,
                allocated_parallel_same,
            ) = self.get_lane_info_only_actors(
                all_vehicles, parallel_same_dir, ego_road_lane_ID_pair
            )
            (
                parallel_opposite_dir_info,
                _,
                allocated_parallel_opposite,
            ) = self.get_lane_info_only_actors(
                all_vehicles, parallel_opposite_dir, None
            )

            all_allocated_vehicle_ids = (
                allocated_left
                + allocated_right
                + allocated_parallel_same
                + allocated_parallel_opposite
            )
            unallocated_actor_ids = [
                vehicle.id
                for vehicle in all_vehicles
                if vehicle.id not in all_allocated_vehicle_ids
            ]
            
            # all_misc_vehicles = [
            #     Vehicle(self.world, vehicle_id) for vehicle_id in unallocated_actor_ids
            # ]
            all_misc_vehicles = self.get_intersection_vehicles(all_vehicles, intersection_id, ego_vehicle.id)
            all_misc_vehicles_ids = [v.actor_id for v in all_misc_vehicles]
            
            for i in range(len(intersecting_left_info)):
                intersecting_left_info[i][0] = [elem for elem in intersecting_left_info[i][0] if elem.actor_id not in all_misc_vehicles_ids]
            for i in range(len(intersecting_right_info)):
                intersecting_right_info[i][0] = [elem for elem in intersecting_right_info[i][0] if elem.actor_id not in all_misc_vehicles_ids]
            for i in range(len(parallel_same_dir_info)):
                parallel_same_dir_info[i][0] = [elem for elem in parallel_same_dir_info[i][0] if elem.actor_id not in all_misc_vehicles_ids]
            for i in range(len(parallel_opposite_dir_info)):
                parallel_opposite_dir_info[i][0] = [elem for elem in parallel_opposite_dir_info[i][0] if elem.actor_id not in all_misc_vehicles_ids]
            for i in range(len(ego_lane_info)):
                parallel_opposite_dir_info[i][0] = [elem for elem in parallel_opposite_dir_info[i][0] if elem.actor_id not in all_misc_vehicles_ids]


            return (
                intersecting_left_info,
                intersecting_right_info,
                parallel_same_dir_info,
                parallel_opposite_dir_info,
                ego_lane_info,
                all_misc_vehicles,
            )

    def get_state_information_lane_follow(self, ego_vehicle=None):
        '''
        Given an ego_vehicle, returns state information:
        '''
        if ego_vehicle == None:
            print("No ego vehicle specified..")
            return None

        # get actors in current, right, and left lane.
        (
            actors_in_current_lane,
            actors_in_left_lane,
            actors_in_right_lane,
        ) = self.sort_parallel_actors_by_lane(ego_vehicle, as_Vehicle=True)

        # also get actors directly in front and rear. 
        # (
        #     front_vehicle,
        #     rear_vehicle
        # ) = self.get_closest_front_and_back_actors(ego_vehicle,
        #         actors_in_current_lane=actors_in_current_lane, as_Vehicle=True)
        
        # find the lane width
        lane_distance = self.get_nearest_waypoint(ego_vehicle).lane_width

        # current lane waypoints
        (
            current_lane_waypoints,
            left_lane_waypoints,
            right_lane_waypoints
        ) = self.get_current_adjacent_lane_waypoints(ego_vehicle, as_LanePoint=True)

        return (
            current_lane_waypoints,
            left_lane_waypoints,
            right_lane_waypoints,
            None,
            None,
            actors_in_current_lane,
            actors_in_left_lane,
            actors_in_right_lane,
            lane_distance,
        )

    '''
    Utility functions
    '''

    def get_current_adjacent_lane_waypoints(self, vehicle, as_LanePoint=False):

        '''Given a vehicle, returns the lane waypoints on current lane, right 
        lane, and left lane.
        If as_LanePoint, returns LanePoint objects rather than waypoints. 
        '''

        # Get ego vehicle location and nearest waypoint for reference.
        vehicle_location = vehicle.get_location()
        nearest_waypoint = self.world_map.get_waypoint(
            vehicle_location, project_to_road=True
        )
        # else:
        #     nearest_waypoint = self.spawn_ego_point
        left_waypoint = nearest_waypoint.get_left_lane()
        right_waypoint = nearest_waypoint.get_right_lane()

        current_lane_waypoints = []
        left_lane_waypoints = []
        right_lane_waypoints = []
        
        # filter current_lane, right_lane, left lane waypoints
        if(nearest_waypoint.lane_type == carla.LaneType.Driving):
            current_lane_waypoints = self.filter_waypoints(
                self.all_waypoints, 
                nearest_waypoint.road_id, 
                nearest_waypoint.lane_id
            )
            
        if(left_waypoint is not None and left_waypoint.lane_type == carla.LaneType.Driving):
            left_lane_waypoints = self.filter_waypoints(
                self.all_waypoints,
                left_waypoint.road_id,
                left_waypoint.lane_id,
            )
        if(right_waypoint is not None and right_waypoint.lane_type == carla.LaneType.Driving):
            right_lane_waypoints = self.filter_waypoints(
                self.all_waypoints,
                right_waypoint.road_id,
                right_waypoint.lane_id,
            )
        
            
        if(nearest_waypoint.lane_id > 0):
            current_lane_waypoints.reverse()
        
        if(left_waypoint is not None and left_waypoint.lane_id > 0):
            left_lane_waypoints.reverse()
            
        if(right_waypoint is not None and right_waypoint.lane_id > 0):
            right_lane_waypoints.reverse()
        

        if as_LanePoint == True:
            current_lane_waypoints = [
                LanePoint(global_pose=self.waypoint_to_pose2D(wp))
                for wp in current_lane_waypoints
            ]
            left_lane_waypoints = [
                LanePoint(global_pose=self.waypoint_to_pose2D(wp))
                for wp in left_lane_waypoints
            ]
            right_lane_waypoints = [
                LanePoint(global_pose=self.waypoint_to_pose2D(wp))
                for wp in right_lane_waypoints
            ]

        return current_lane_waypoints, left_lane_waypoints, right_lane_waypoints

    def get_nearest_waypoint(self, actor):
        return self.world_map.get_waypoint(
                actor.get_location(), project_to_road=True
            )
        
    def get_full_lane_waypoints(self, waypoint):
        if(waypoint is None):
            return []
        if(waypoint.lane_type != carla.LaneType.Driving):
            return []
        
        waypoints =  waypoint.previous_until_lane_start(1) + [waypoint] + waypoint.next_until_lane_end(1)
        return waypoints
    
    def get_lane_ids(self, waypoints):
        '''Given a list of waypoints, returns unique lane ids'''
        
        if not isinstance(waypoints, list):
            waypoints = [waypoints]
        
        return list(set([wp.lane_id for wp in waypoints]))

    def sort_parallel_actors_by_lane(self, ego_vehicle, as_Vehicle=False):
        '''Given an ego_vehicle, returns three lists of vehicles in the current,
        right and left lane actors.
        If as_Vehicle, returns Vehicle objects, else returns carla.Actor'''

        # Get waypoints in current, right and left lane
        (
        current_lane_waypoints, 
        left_lane_waypoints, 
        right_lane_waypoints 
        ) = self.get_current_adjacent_lane_waypoints(ego_vehicle)

        # get lane ids for current, left and right lanes
        left_lane_ids = self.get_lane_ids(left_lane_waypoints)
        current_lane_ids = self.get_lane_ids(current_lane_waypoints)
        right_lane_ids = self.get_lane_ids(right_lane_waypoints)

        # containers for actors in current, left and right lanes
        actors_in_current_lane = []
        actors_in_left_lane = []
        actors_in_right_lane = []

        # get all the vehicles in the world
        all_vehicles = self.world.get_actors().filter("vehicle.*")
        
        current_road_id = [self.get_nearest_waypoint(ego_vehicle).road_id]
        # loop over all actors to get vehicles in the same, right of left lane
        # as ego vehicles

        ######### Fix the carla lane_ids bug
        road_ids_with_road_bug = [(5,49),  (17,47), (16,25), (18,40), (26,29) ,(50,52)]
        road_ids_with_lane_bug = [(26,29) ,(50,52)]

        for pair in road_ids_with_road_bug:
            if current_road_id[0] in pair:
                for elem in pair:
                    if elem == current_road_id[0]: continue
                    current_road_id.append(elem)
                break

        for pair in road_ids_with_lane_bug: # loop over all buggy pairs
            if current_road_id[0] in pair:  # check if current road_id is buggy
                # for buggy lanes, also add the negative of lane_id
                left_lane_ids.append(left_lane_ids[0] * -1)
                right_lane_ids.append(right_lane_ids[0] * -1)
                break
        ########

        for actor in all_vehicles:

            # skip the ego vehicle for the following calculations
            if actor.id == ego_vehicle.id:
                continue
            
            # get waypoint closest to the actor
            actor_nearest_waypoint = self.get_nearest_waypoint(actor)
            # append the actor to the correct lane list based on their lane id
            if actor_nearest_waypoint.lane_id in left_lane_ids and actor_nearest_waypoint.road_id in current_road_id:
                actors_in_left_lane.append(actor)
            elif actor_nearest_waypoint.lane_id in right_lane_ids and actor_nearest_waypoint.road_id in current_road_id:
                actors_in_right_lane.append(actor)
            elif actor_nearest_waypoint.lane_id in current_lane_ids and actor_nearest_waypoint.road_id in current_road_id:
                actors_in_current_lane.append(actor)

        # return as Vehicle object list, instead of carla.Vehicle object list
        if as_Vehicle:
            actors_in_current_lane = [Vehicle(self.world, vehicle.id) for \
                                        vehicle in actors_in_current_lane]
            actors_in_left_lane = [Vehicle(self.world, vehicle.id) for \
                                    vehicle in actors_in_left_lane]
            actors_in_right_lane = [Vehicle(self.world, vehicle.id) for \
                                    vehicle in actors_in_right_lane]

        return actors_in_current_lane, actors_in_left_lane, \
                actors_in_right_lane

    def get_closest_front_and_back_actors(self, ego_vehicle, 
                    actors_in_current_lane=None, as_Vehicle=False):

        if actors_in_current_lane is None:
            actors_in_current_lane= sort_parallel_actors_by_lane(ego_vehicle)
        
        # weird mayank logic
        closest_distance_front = (10000000000)  # TODO Formalize this
        closest_distance_rear = (-10000000000)  # TODO Formalize this

        # container for actors right in front and back
        front_vehicle = None
        rear_vehicle = None

        # loop over all actors to find the closest in front and back
        for actor in actors_in_current_lane:

            # convert actors in Vehicle class 
            if isinstance(actor, Vehicle):
                actor = self.world.get_actor(actor.actor_id)
            # skip the ego vehicle for the following calculations
            if actor.id == ego_vehicle.id:
                continue
            
            # get waypoint closest to the actor
            actor_nearest_waypoint = self.get_nearest_waypoint(actor)
                
            # get the current actors location in ego vehicle frame
            actor_loc_in_ego = (
                self.convert_global_transform_to_actor_frame(
                    actor=ego_vehicle, transform=actor.get_transform()))

            actor_x_in_ego = actor_loc_in_ego[0][0]
            
            # check if actor is directly in front or back of ego
            if (actor_x_in_ego > 0.0) and \
                (actor_x_in_ego < closest_distance_front):
                front_vehicle = actor
                closest_distance_front = actor_x_in_ego
            elif (actor_x_in_ego < 0.0) and \
                (actor_x_in_ego > closest_distance_rear):
                rear_vehicle = actor
                closest_distance_rear = actor_x_in_ego
        
        if as_Vehicle:
            vehicle_dummy = Vehicle(
                actor_id=-1, speed=-1, location_global=Pose2D(1000, 1000, 0))

            # Front vehicle
            if front_vehicle == None:
                front_vehicle = vehicle_dummy
            else:
                front_vehicle = Vehicle(self.world, front_vehicle.id)

            # Rear vehicle
            if rear_vehicle == None:
                rear_vehicle = vehicle_dummy
            else:
                rear_vehicle = Vehicle(self.world, rear_vehicle.id)
        
        return front_vehicle, rear_vehicle
    
    def carlavehicle_to_Vehicle_class(self, carla_vehicle_list):
        '''Returns a Vehicle class for carla.Vehicle list'''
        
        vehicle_list = []

        for vehicle in carla_vehicle_list:
            vehicle_list.append(Vehicle(self.world, self.vehicle.id))
        
        return vehicle_list
    
    def location_to_Pose2D(self, location):
        '''For a given location, projects a waypoint on the road and returns a
        Pose2D object for it'''

        waypoint = self.world_map.get_waypoint()

    def get_distance_to_lane_end(self, vehicle):
        carla_vehicle = self.world.get_actor(vehicle.actor_id)
        nearest_waypoint = self.get_nearest_waypoint(carla_vehicle)

        waypoints_to_end_of_lane = nearest_waypoint.next_until_lane_end(1)

        return len(waypoints_to_end_of_lane)

    def get_distance_to_lane_end_2(self, vehicle):
        nearest_waypoint = self.get_nearest_waypoint(vehicle)

        waypoints_to_end_of_lane = nearest_waypoint.next_until_lane_end(1)

        return len(waypoints_to_end_of_lane)