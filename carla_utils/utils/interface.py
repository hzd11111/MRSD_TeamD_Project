import random
import sys
from typing import Optional, List
from configparser import ConfigParser

config = ConfigParser()
config.read("config.ini")
CARLA_PATH = config.get("main", "CARLA_PATH")
# Enable import of 'carla'
sys.path.append(CARLA_PATH + "PythonAPI/carla/dist/carla-0.9.9-py3.6-linux-x86_64.egg")

import carla


class ActorExtractor(object):
    __slots__ = ["world"]

    def __init__(self, world: carla.libcarla.World):
        self.world = world

    def get_actors(self):
        """
        Get a list of all actors in the CARLA world. Actors are a superset of vehicles, pedestrians, traffic lights etc.
        """
        return self.world.get_actors()


class VehicleExtractor(ActorExtractor):
    def __init__(self, world: carla.libcarla.World):
        ActorExtractor.__init__(self, world)

    def get_vehicles(self):
        """
        Get a list of all vehicles in the CARLA world.
        """
        return self.world.get_actors().filter("vehicle.*")


class PedestrianExtractor(ActorExtractor):
    def __init__(self, world: carla.libcarla.World):
        ActorExtractor.__init__(self, world)

    def get_pedestrians(self):
        """
        Get a list of all pedestrians in the CARLA world.
        """
        return self.world.get_actors().filter("walker.*")


class ActorFilter(object):
    __slots__ = ["world", "world_map"]

    def __init__(self, world: carla.libcarla.World):
        self.world = world
        self.world_map = self.world.get_map()

    def filter_by_road(self, road_id: int, actor_list):
        """
        Filter a list of actors by 'road_id'.
        """
        return [
            actor
            for actor in actor_list
            if self.world_map.get_waypoint(
                actor.get_location(), project_to_road=True
            ).road_id
            == road_id
        ]

    def filter_by_lane(self, lane_id: int, actor_list):
        """
        Filter a list of actors by 'lane_id'. This should ideally be used only after actors have been filtered using 'road_id'.
        """
        return [
            actor
            for actor in actor_list
            if self.world_map.get_waypoint(
                actor.get_location(), project_to_road=True
            ).lane_id
            == lane_id
        ]

    def filter_by_road_and_lane_id(self, road_id: int, lane_id: int, actor_list):
        """
        Filter a list of actors by both 'road_id' and 'lane_id'.
        """
        return [
            actor
            for actor in actor_list
            if self.world_map.get_waypoint(
                actor.get_location(), project_to_road=True
            ).lane_id
            == lane_id
            and self.world_map.get_waypoint(
                actor.get_location(), project_to_road=True
            ).road_id
            == road_id
        ]
