#!/usr/bin/env python

""" Implementation of the CarlaHandler class. CarlaHandler class provides some custom built APIs for Carla. """

__author__ = "Mayank Singal"
__maintainer__ = "Mayank Singal"
__email__ = "mayanksi@andrew.cmu.edu"
__version__ = "0.1"


import random
import time

import numpy as np
import cv2
import carla

from utils import get_matrix, create_bb_points


class CarlaHandler:

	def __init__(self, client):

		self.client = client  # TODO: Is this needed?
		self.world = client.get_world()
		self.world_map = self.world.get_map()
		self.blueprint_library = self.world.get_blueprint_library()
		self.actor_dict = {}

		print("Handler Initialized!\n")

	def __del__(self):
		print("Handler destroyed..\n")

	def destroy_actors(self):
		for actor in self.world.get_actors():
			actor.destroy()
		print("All actors destroyed..\n") 


	def get_spawn_points(self):

		return self.world_map.get_spawn_points()

	def spawn_vehicle(self, vehicle_type = 'model3', spawn_point=None):

		if(spawn_point == None):

			spawn_point = random.choice(self.get_spawn_points())

		
		vehicle_blueprint = self.blueprint_library.filter(vehicle_type)[0]

		vehicle = self.world.spawn_actor(vehicle_blueprint, spawn_point)

		self.actor_dict[vehicle.id] = vehicle

		print("Vehicle spawned at", spawn_point, "with ID:", vehicle.id, "\n")

		return vehicle, vehicle.id


	def get_waypoints(self, distance=1.0):

		return self.world_map.generate_waypoints(distance=distance)

	def filter_waypoints(self, waypoints, road_id=None):

		filtered_waypoints = []
		for waypoint in waypoints:
			if(waypoint.road_id == road_id):
				filtered_waypoints.append(waypoint)

		return filtered_waypoints

	def draw_waypoints(self, waypoints, road_id=None, section_id=None, life_time=50.0):

		for waypoint in waypoints:
		
			if(waypoint.road_id == road_id):
				self.world.debug.draw_string(waypoint.transform.location, 'O', draw_shadow=False,
			                               color=carla.Color(r=0, g=255, b=0), life_time=life_time,
			                               persistent_lines=True)


	def move_vehicle(self, vehicle_id=None, throttle=None, steer=None):

		if(vehicle_id==None or throttle==None or steer==None):
			print("Invalid vechicle motion parameters.")
		else:
			if(self.actor_dict[vehicle_id]==None):
				print("Actor with given ID does not exist")
			else:
				vehicle = self.actor_dict[vehicle_id]
				vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))


	def convert_global_transform_to_actor_frame(self, actor=None, transform=None):

		if(actor == None or transform == None):
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

			transform_position_as_seen_from_actor = np.dot(R_world_to_actor, transform_coords)
			
			return transform_position_as_seen_from_actor




 









