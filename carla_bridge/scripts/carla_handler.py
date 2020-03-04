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
import cv2
import carla

from utils import get_matrix, create_bb_points


class CarlaHandler:

	def __init__(self, client):

		self.client = client
		self.world = client.get_world()
		self.world_map = self.world.get_map()
		self.all_waypoints = self.get_waypoints()
		self.blueprint_library = self.world.get_blueprint_library()
		self.actor_dict = {}
		settings = self.world.get_settings()
		settings.synchronous_mode = True
		settings.fixed_delta_seconds = 0.05
		self.world.apply_settings(settings)

		print("Handler Initialized!\n")

	def __del__(self):
		self.destroy_actors()
		settings = self.world.get_settings()
		settings.synchronous_mode = False
		self.world.apply_settings(settings)
		print("Handler destroyed..\n")


	def destroy_actors(self):
		for actor in self.world.get_actors():
			if actor.id in self.actor_dict:
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

	def filter_waypoints(self, waypoints, road_id=None, lane_id=None):

		filtered_waypoints = []
		for waypoint in waypoints:
			if(lane_id == None):
				if(waypoint.road_id == road_id):
					filtered_waypoints.append(waypoint)
			else:
				if(waypoint.road_id == road_id and waypoint.lane_id == lane_id):
					filtered_waypoints.append(waypoint)

		return filtered_waypoints

	def draw_waypoints(self, waypoints, road_id=None, section_id=None, life_time=50.0):

		for waypoint in waypoints:
		
			if(road_id == None or waypoint.road_id == road_id):
				self.world.debug.draw_string(waypoint.transform.location, 'O', draw_shadow=False,
			                               color=carla.Color(r=0, g=255, b=0), life_time=life_time,
			                               persistent_lines=True)

	def draw_arrow(self, waypoints, road_id=None, section_id=None, life_time=50.0):

		for i,waypoint in enumerate(waypoints):
			if(i == len(waypoints)-1):
				continue
			trans = waypoints[i+1].transform

			yaw_in_rad = math.radians(np.arctan(waypoint.transform.location.y - trans.location.y)/(waypoint.transform.location.x - trans.location.x)) 
			
			p1 = carla.Location(
			x=trans.location.x + math.cos(yaw_in_rad),
			y=trans.location.y + math.sin(yaw_in_rad),
			z=trans.location.z)

			if(road_id == None or waypoint.road_id == road_id):
				self.world.debug.draw_arrow(waypoint.transform.location, p1, thickness = 0.01, arrow_size=0.05,
			                               color=carla.Color(r=0, g=255, b=0), life_time=life_time)


	def move_vehicle(self, vehicle_id=None, control=None):

		if(vehicle_id==None or control==None):
			print("Invalid vechicle motion parameters.")
		else:
			if(self.actor_dict[vehicle_id]==None):
				print("Actor with given ID does not exist")
			else:
				vehicle = self.actor_dict[vehicle_id]
				vehicle.apply_control(control)


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


	def get_state_information(self, ego_vehicle=None):

		# Check for valid inputs
		if(ego_vehicle==None):
			print("No ego vehicle specified..")
			return None
		else:
			# Get ego vehicle location and nearest waypoint for reference.
			ego_vehicle_location = ego_vehicle.get_location()
			nearest_waypoint = self.world_map.get_waypoint(ego_vehicle_location, project_to_road=True)

			# Get current road and lane IDs
			current_road_ID = nearest_waypoint.road_id
			#print("Spawn Road ID Inside Handler:", current_road_ID)
			current_lane_ID = nearest_waypoint.lane_id

			# Get IDs of left and right lanes
			left_lane_ID = nearest_waypoint.get_left_lane().lane_id
			right_lane_ID = nearest_waypoint.get_right_lane().lane_id

			# Finding waypoints in current, left and right lanes
			current_lane_waypoints = self.filter_waypoints(self.all_waypoints, road_id=current_road_ID, lane_id=current_lane_ID)
			left_lane_waypoints = self.filter_waypoints(self.all_waypoints, road_id=current_road_ID, lane_id=left_lane_ID)
			right_lane_waypoints = self.filter_waypoints(self.all_waypoints, road_id=current_road_ID, lane_id=right_lane_ID)

			# Containers for leading and rear vehicle in current lane
			front_vehicle = None
			rear_vehicle = None

			closest_distance_front = 10000000000 #TODO Change this to more formal value
			closest_distance_rear = -10000000000 #TODO Change this to more formal value

			# Containers for actors in current, left and right lanes
			actors_in_current_lane = []
			actors_in_left_lane = []
			actors_in_right_lane = []

			# Fill containers defined above
			for actor in self.world.get_actors().filter('vehicle.*'):


				# For all actors that are not ego vehicle
				if(actor.id != ego_vehicle.id):

					# Find nearest waypoint on the map
					actor_nearest_waypoint = self.world_map.get_waypoint(actor.get_location(), project_to_road=True)

					# If actor is on the same road as the ego vehicle
					if(actor_nearest_waypoint.road_id == current_road_ID):
						
						#print(actor_nearest_waypoint.road_id, actor_nearest_waypoint.lane_id, "OLA")
						# If actor is on the same lane as the ego vehicle: Add to relevant container, and find if it's the leading or trailing vehicle
						if(actor_nearest_waypoint.lane_id == current_lane_ID):
							actors_in_current_lane.append(actor)
							
							curr_actor_location_in_ego_vehicle_frame = self.convert_global_transform_to_actor_frame(actor=ego_vehicle, transform=actor.get_transform())
							
							if(curr_actor_location_in_ego_vehicle_frame[0][0] > 0.0 and curr_actor_location_in_ego_vehicle_frame[0][0] < closest_distance_front):
								front_vehicle = actor
								closest_distance_front = curr_actor_location_in_ego_vehicle_frame[0][0]
							elif(curr_actor_location_in_ego_vehicle_frame[0][0] < 0.0 and curr_actor_location_in_ego_vehicle_frame[0][0] > closest_distance_rear):
								rear_vehicle = actor
								closest_distance_rear = curr_actor_location_in_ego_vehicle_frame[0][0]
							
						# Add to relevant container
						elif(actor_nearest_waypoint.lane_id == left_lane_ID):
							actors_in_left_lane.append(actor)
						# Add to relevant container
						elif(actor_nearest_waypoint.lane_id == right_lane_ID):
							actors_in_right_lane.append(actor)

			return current_lane_waypoints, left_lane_waypoints, right_lane_waypoints, front_vehicle, rear_vehicle, actors_in_current_lane, actors_in_left_lane, actors_in_right_lane








			

			
















 









