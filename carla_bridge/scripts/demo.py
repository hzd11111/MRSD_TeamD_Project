import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import random
import time
import numpy as np
import cv2
import carla
import glob
import os

from carla_handler import CarlaHandler
from utils import get_matrix, create_bb_points


### Start Client. Make sure Carla server is running before starting.

client = carla.Client('localhost', 2000)
client.set_timeout(2.0)


################################################################################################################################

### Create a CarlaHandler object. CarlaHandler acts as an API for the Carla Server.

carla_handler_1 = CarlaHandler(client)
#carla_handler_1.destroy_actors()


################################################################################################################################

### Example code to draw waypoints
carla_handler_1.draw_waypoints(carla_handler_1.get_waypoints(), road_id=22)


################################################################################################################################

### Example code to spawn a vehicle using pre-defined spawn points

# # Get spawn point
# spawn_points = carla_handler_1.get_spawn_points()

# # Select a spawn point
# spawn_point = spawn_points[0] # Can also select a random spawn point

# # Spawn Vehicle
# carla_handler_1.spawn_vehicle(spawn_point=spawn_point)

################################################################################################################################

### Example code to spawn a vehicle using manually selected/defined waypoints, and printing vehicle state information.

filtered_waypoints = carla_handler_1.filter_waypoints(carla_handler_1.get_waypoints(), road_id=12)
spawn_point = filtered_waypoints[100].transform

# Increase Z to avoid collisions during spawn
spawn_point.location.z = spawn_point.location.z + 2
#spawn_point.rotation.yaw += 4

# Spawn vehicle. (Remember to Keep track of ID for future use)
ego_vehicle, ego_vehicle_ID = carla_handler_1.spawn_vehicle(spawn_point=spawn_point)

# Get ego vehicle state. Can be used to get information about any actor given ID.
#ego_vehicle = carla_handler_1.actor_dict[ego_vehicle_ID]
ego_vehicle_transform = ego_vehicle.get_transform()
print("Transform:", ego_vehicle_transform, "\n")
ego_vehicle_velocity = ego_vehicle.get_velocity()
print("Velocity:", ego_vehicle_velocity, "\n")
ego_vehicle_bounding_box = ego_vehicle.bounding_box
print("Bounding Box:", ego_vehicle_bounding_box, "\n")


################################################################################################################################

### Example code to obtain current road + current lane/ right lane/ left lane information.

# Get ego vehicle location
ego_vehicle_location = ego_vehicle.get_location()

# Get nearest waypoint
nearest_waypoint = carla_handler_1.world_map.get_waypoint(ego_vehicle_location, project_to_road=True)
print("Nearest Waypoint:", nearest_waypoint, "\n")

# Get current road ID and current lane ID (Road and lane on which ego vehicle is currently operating). Lane IDs follow opendrive standards.
current_road_ID = nearest_waypoint.road_id
current_lane_ID = nearest_waypoint.lane_id
print("Current Road ID:", current_road_ID)
print("Current Lane ID:", current_lane_ID, "\n")

# Get left lane ID and right lane ID
left_lane_ID = nearest_waypoint.get_left_lane().lane_id
right_lane_ID = nearest_waypoint.get_right_lane().lane_id
print("Lane to the left of current lane ID:", left_lane_ID)
print("Lane to the right of current lane ID:", right_lane_ID, "\n")

# Get nearest waypoint on the left lane
left_lane_nearest_waypoint = nearest_waypoint.get_left_lane()
right_lane_nearest_waypoint = nearest_waypoint.get_right_lane()
print("Closest waypoint in the left lane:", left_lane_nearest_waypoint)
print("Closest waypoint in the right lane:", right_lane_nearest_waypoint, "\n")


################################################################################################################################

# Example code to test throttle and steer vehicles
#carla_handler_1.move_vehicle(ego_vehicle_ID, 1.0, 0.0)

################################################################################################################################

### Example code to get IDs of all actors on current road (Current and other lane)


# while(True):
# # Get all actors
# 	all_actors = carla_handler_1.world.get_actors()

# 	actors_in_current_lane = []
# 	actors_in_other_lane = []
# 	for actor in all_actors.filter('vehicle.*'):
# 		actor_nearest_waypoint = carla_handler_1.world_map.get_waypoint(actor.get_location(), project_to_road=True)
# 		if(actor_nearest_waypoint.road_id == current_road_ID):
			
# 			if(actor_nearest_waypoint.lane_id == current_lane_ID):
# 				actors_in_current_lane.append(actor.id)
# 				tmp_vehicle = actor
# 				tmp_transform = tmp_vehicle.get_transform()
# 				tmp_bounding_box = tmp_vehicle.bounding_box
# 				tmp_bounding_box.location += tmp_transform.location
# 				carla_handler_1.world.debug.draw_box(tmp_bounding_box, tmp_transform.rotation, life_time=0.05)
# 			else:
# 				actors_in_other_lane.append(actor.id)
# 				# Draw bounding boxes



# 	print("Actors in current lane:", actors_in_current_lane)
# 	print("Actors in other lane:", actors_in_other_lane, "\n")

# 	time.sleep(1)


################################################################################################################################

### Example code to find vehicle in front and vehicle behind the ego vehicle

## Spawn some test vehicles.
spawn_point = filtered_waypoints[150].transform
spawn_point.location.z = spawn_point.location.z + 0.75
vehicle_front, vehicle_front_id = carla_handler_1.spawn_vehicle(spawn_point=spawn_point)

spawn_point = filtered_waypoints[200].transform
spawn_point.location.z = spawn_point.location.z + 0.75
vehicle_front_2, vehicle_front_2_id = carla_handler_1.spawn_vehicle(spawn_point=spawn_point)

spawn_point = filtered_waypoints[50].transform
spawn_point.location.z = spawn_point.location.z + 0.75
vehicle_back, vehicle_back_id = carla_handler_1.spawn_vehicle(spawn_point=spawn_point)

spawn_point = filtered_waypoints[0].transform
spawn_point.location.z = spawn_point.location.z + 0.75
vehicle_back_2, vehicle_back_2_id = carla_handler_1.spawn_vehicle(spawn_point=spawn_point)

spawn_point = filtered_waypoints[75].transform
spawn_point.location.z = spawn_point.location.z + 0.75
spawn_point.rotation.yaw += 180
vehicle_front, vehicle_front_id = carla_handler_1.spawn_vehicle(spawn_point=spawn_point)

spawn_point = filtered_waypoints[175].transform
spawn_point.location.z = spawn_point.location.z + 0.75
spawn_point.rotation.yaw += 180
vehicle_front_2, vehicle_front_2_id = carla_handler_1.spawn_vehicle(spawn_point=spawn_point)

spawn_point = filtered_waypoints[25].transform
spawn_point.location.z = spawn_point.location.z + 0.75
spawn_point.rotation.yaw += 180
vehicle_back, vehicle_back_id = carla_handler_1.spawn_vehicle(spawn_point=spawn_point)


#time.sleep(3)
#ego_vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))

# Get rotation matrix from : ego vehicle location and rotation vectors (i.e ego vehicle transform)
# ego_vehicle_to_world_transform = ego_vehicle.get_transform()
# R_ego_vehicle_to_world = get_matrix(ego_vehicle_to_world_transform)
# R_world_to_ego_vehicle = np.linalg.inv(R_ego_vehicle_to_world)


while(True):
	# Get all actors
	all_actors = carla_handler_1.world.get_actors()

	vehicle_in_front = None
	vehicle_in_rear = None

	actors_in_current_lane = []
	actors_in_other_lane = []
	for actor in all_actors.filter('vehicle.*'):

		if(actor.id != ego_vehicle.id):

			tmp_transform = actor.get_transform()
			tmp_bounding_box = actor.bounding_box
			tmp_bounding_box.location += tmp_transform.location

			actor_nearest_waypoint = carla_handler_1.world_map.get_waypoint(actor.get_location(), project_to_road=True)
			#carla_handler_1.world.debug.draw_box(tmp_bounding_box, tmp_transform.rotation, life_time=0.5)
			
			if(actor_nearest_waypoint.road_id == current_road_ID):
				
				if(actor_nearest_waypoint.lane_id == nearest_waypoint.lane_id):
					actors_in_current_lane.append(actor.id)
					tmp_vehicle = actor
					tmp_transform = tmp_vehicle.get_transform()
					tmp_bounding_box = tmp_vehicle.bounding_box
					tmp_bounding_box.location += tmp_transform.location
					#carla_handler_1.world.debug.draw_box(tmp_bounding_box, tmp_transform.rotation, life_time=0.05)
					curr_actor_location_in_ego_vehicle_frame = carla_handler_1.convert_global_transform_to_actor_frame(actor=ego_vehicle, transform=actor.get_transform())
					if(curr_actor_location_in_ego_vehicle_frame[0][0] >= 0.0):
						
						tmp_transform = actor.get_transform()
						tmp_bounding_box = actor.bounding_box
						tmp_bounding_box.location += tmp_transform.location
						carla_handler_1.world.debug.draw_box(tmp_bounding_box, tmp_transform.rotation, life_time=1, color=carla.Color(255,0,255))
					else:

						tmp_transform = actor.get_transform()
						tmp_bounding_box = actor.bounding_box
						tmp_bounding_box.location += tmp_transform.location
						carla_handler_1.world.debug.draw_box(tmp_bounding_box, tmp_transform.rotation, life_time=1, color=carla.Color(0,255,255))




				else:
					
					actors_in_other_lane.append(actor.id)
					tmp_transform = actor.get_transform()
					tmp_bounding_box = actor.bounding_box
					tmp_bounding_box.location += tmp_transform.location
					carla_handler_1.world.debug.draw_box(tmp_bounding_box, tmp_transform.rotation, life_time=1)
					# Draw bounding boxes

	time.sleep(2)



# transform_vehicle_front = vehicle_front.get_transform()
# vehicle_front_coords = np.zeros((4, 1))
# vehicle_front_coords[0] = transform_vehicle_front.location.x
# vehicle_front_coords[1] = transform_vehicle_front.location.y
# vehicle_front_coords[2] = transform_vehicle_front.location.z
# vehicle_front_coords[3] = 1

# front_vehicle_as_seen_from_ego_vehicle = np.dot(R_world_to_ego_vehicle, vehicle_front_coords)
# print(front_vehicle_as_seen_from_ego_vehicle, "\n")


# transform_vehicle_front_2 = vehicle_front_2.get_transform()
# vehicle_front_coords_2 = np.zeros((4, 1))
# vehicle_front_coords_2[0] = transform_vehicle_front_2.location.x
# vehicle_front_coords_2[1] = transform_vehicle_front_2.location.y
# vehicle_front_coords_2[2] = transform_vehicle_front_2.location.z
# vehicle_front_coords_2[3] = 1

# front_vehicle_2_as_seen_from_ego_vehicle = np.dot(R_world_to_ego_vehicle, vehicle_front_coords_2)
# print(front_vehicle_2_as_seen_from_ego_vehicle)


# bb_points_vehicle_front = create_bb_points(carla_handler_1.actor_dict[vehicle_front])
# bb_front_vehicle_as_seen_from_ego_vehicle = np.dot(R_world_to_ego_vehicle, np.transpose(bb_points_vehicle_front))
# print(bb_front_vehicle_as_seen_from_ego_vehicle)




















