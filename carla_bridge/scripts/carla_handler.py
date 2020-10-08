#!/usr/bin/env python

""" Implementation of the CarlaHandler class. CarlaHandler class provides some custom built APIs for Carla. """

__author__ = "Mayank Singal"
__maintainer__ = "Mayank Singal"
__email__ = "mayanksi@andrew.cmu.edu"
__version__ = "0.1"


import random
import time
import math

import numpy as np
import carla

from utils import get_matrix, create_bb_points
from enum import Enum
import re


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

    def get_spawn_points(self):

        return self.world_map.get_spawn_points()

    def spawn_vehicle(self, vehicle_type="model3", spawn_point=None):

        if spawn_point == None:

            spawn_point = random.choice(self.get_spawn_points())

        vehicle_blueprint = self.blueprint_library.filter(vehicle_type)[0]

        vehicle = self.world.spawn_actor(vehicle_blueprint, spawn_point)

        self.actor_dict[vehicle.id] = vehicle

        print("Vehicle spawned at", spawn_point, "with ID:", vehicle.id, "\n")

        return vehicle, vehicle.id

    def get_waypoints(self, distance=1):

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
        self, waypoints, road_id=None, section_id=None, life_time=50.0, color=False
    ):
        if color:
            b = 255
        else:
            b = 0

        for waypoint in waypoints:

            if waypoint.road_id == road_id or road_id == None:
                self.world.debug.draw_string(
                    waypoint.transform.location,
                    "O",
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

    def draw_arrow(self, waypoints, road_id=None, section_id=None, life_time=50.0):

        for i, waypoint in enumerate(waypoints):
            if i == len(waypoints) - 1:
                continue
            trans = waypoints[i + 1].transform
            # yaw_in_rad = math.radians(trans.rotation.yaw)
            yaw_in_rad = math.radians(
                np.arctan(waypoint.transform.location.y - trans.location.y)
                / (waypoint.transform.location.x - trans.location.x)
            )
            # pitch_in_rad = math.radians(trans.rotation.pitch)
            p1 = carla.Location(
                x=trans.location.x + math.cos(yaw_in_rad),
                y=trans.location.y + math.sin(yaw_in_rad),
                z=trans.location.z,
            )

            if road_id == None or waypoint.road_id == road_id:
                self.world.debug.draw_arrow(
                    waypoint.transform.location,
                    p1,
                    thickness=0.01,
                    arrow_size=0.05,
                    color=carla.Color(r=0, g=255, b=0),
                    life_time=life_time,
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

    def get_next_waypoints(self, last_waypoint, ego_speed, rev=False, k=100):

        if last_waypoint == None:
            return []

        sampling_radius = 1  # ego_speed * 1 / 3.6
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

    def get_state_information_new(
        self,
        ego_vehicle=None,
        original_lane_ID=None,
    ):

        if ego_vehicle == None:
            print("No ego vehicle specified..")
            return None
        else:
            # Get ego vehicle location and nearest waypoint for reference.
            ego_vehicle_location = ego_vehicle.get_location()
            nearest_waypoint = self.world_map.get_waypoint(
                ego_vehicle_location, project_to_road=True
            )

            ego_speed = (
                np.sqrt(
                    ego_vehicle.get_velocity().x ** 2
                    + ego_vehicle.get_velocity().y ** 2
                    + ego_vehicle.get_velocity().z ** 2
                )
                * 3.6
            )

            current_lane_waypoints = self.get_next_waypoints(
                nearest_waypoint, ego_speed, k=300
            )[::-1]
            left_lane_waypoints = self.get_next_waypoints(
                nearest_waypoint.get_left_lane(), ego_speed, k=300
            )[
                ::-1
            ]  # +
            right_lane_waypoints = self.get_next_waypoints(
                nearest_waypoint.get_right_lane(), ego_speed, k=300
            )[
                ::-1
            ]  # +

            # self.draw_waypoints(current_lane_waypoints, life_time=5)
            # self.draw_waypoints(left_lane_waypoints, life_time=5, color=True)

            left_lane_ids = list(set([wp.lane_id for wp in left_lane_waypoints]))
            current_lane_ids = list(set([wp.lane_id for wp in current_lane_waypoints]))
            right_lane_ids = list(set([wp.lane_id for wp in right_lane_waypoints]))

            # Containers for actors in current, left and right lanes
            actors_in_current_lane = []
            actors_in_left_lane = []
            actors_in_right_lane = []

            # Containers for leading and rear vehicle in current lane
            front_vehicle = None
            rear_vehicle = None

            closest_distance_front = (
                10000000000  # TODO Change this to more formal value
            )
            closest_distance_rear = (
                -10000000000
            )  # TODO Change this to more formal value

            for actor in self.world.get_actors().filter("vehicle.*"):

                # For all actors that are not ego vehicle
                if actor.id != ego_vehicle.id:
                    actor_nearest_waypoint = self.world_map.get_waypoint(
                        actor.get_location(), project_to_road=True
                    )
                    if actor_nearest_waypoint.lane_id in left_lane_ids:
                        actors_in_left_lane.append(actor)
                    elif actor_nearest_waypoint.lane_id in right_lane_ids:
                        actors_in_right_lane.append(actor)
                    else:

                        actors_in_current_lane.append(actor)

                        curr_actor_location_in_ego_vehicle_frame = (
                            self.convert_global_transform_to_actor_frame(
                                actor=ego_vehicle, transform=actor.get_transform()
                            )
                        )

                        if (
                            curr_actor_location_in_ego_vehicle_frame[0][0] > 0.0
                            and curr_actor_location_in_ego_vehicle_frame[0][0]
                            < closest_distance_front
                        ):
                            front_vehicle = actor
                            closest_distance_front = (
                                curr_actor_location_in_ego_vehicle_frame[0][0]
                            )
                        elif (
                            curr_actor_location_in_ego_vehicle_frame[0][0] < 0.0
                            and curr_actor_location_in_ego_vehicle_frame[0][0]
                            > closest_distance_rear
                        ):
                            rear_vehicle = actor
                            closest_distance_rear = (
                                curr_actor_location_in_ego_vehicle_frame[0][0]
                            )

        return (
            current_lane_waypoints,
            left_lane_waypoints,
            right_lane_waypoints,
            front_vehicle,
            rear_vehicle,
            actors_in_current_lane,
            actors_in_left_lane,
            actors_in_right_lane,
        )