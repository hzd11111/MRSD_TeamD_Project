import glob
import os
import sys
import csv
import carla
import random
import time
import numpy as np


class DynamicPedestrian():
    def __init__(self, world, road_id=None, lane_id=None, distance=None, ego_veh_loc=None):
        self.location = None
        self.rotation = None
        self.world    = world
        self.actor    = None
        self.id       = None
        self.walker_blueprint = None
        self.spawn_location = None
        self.direction = None
        self.max_spawn_attempts = 5
        

        # Lane information
        self.road_id = road_id
        self.lane_id = lane_id
        self.waypoints_list = None#self.get_waypoints_list()
        self.closest_waypoint = None

        # Relative Distance
        self.distance = distance
        self.ego_veh_loc = ego_veh_loc
        self.offset = None
        self.z_spawn_height = None
        self.lane_width = None
        
        # Pedestrian motion parameters
        self.direction = None
        self.speed = None
        self.max_speed = 5
        

    def set_waypoints_list(self):
        '''
        Gets a list of road waypoints from CARLA map for the particular road and lane id
        Input: road id and lane id
        Output: list of waypoints on the specified road at every 0.5 meter 
        '''
        road_id = self.road_id
        lane_id = self.lane_id
        world_map = self.world.get_map()
        waypoint_interval = 0.5
        first_waypoint = world_map.get_waypoint_xodr(road_id, lane_id, 0)
        self.waypoints_list = first_waypoint.next_until_lane_end(waypoint_interval)

        self.lane_width = first_waypoint.lane_width

        # return waypoints_list

    def random_spawn(self):
        spawn_attempts = 0
        # choose walker type
        self.walker_blueprint = self.world.get_blueprint_library().filter('walker.pedestrian.*')[5]
        
        while spawn_attempts < self.max_spawn_attempts:
            try:
                # choose a random waypoint
                waypoint_ind          = random.randint(1,len(self.waypoints_list))
                self.closest_waypoint = self.waypoints_list[waypoint_ind]
                prev_waypoint         = self.waypoints_list[waypoint_ind - 1]
                self.road_vector      = self.closest_waypoint.transform.location - prev_waypoint.transform.location
                
                # specify spawn location closest to the random waypoint
                perp_vector = -1 * np.array([self.road_vector.y, self.road_vector.x, 0])  # vec perpedicular to the road
                perp_vector /= np.linalg.norm(perp_vector) # normalize the vec
                wp_loc = self.closest_waypoint.transform.location
                self.offset = prev_waypoint.lane_width/4 * 2.5
                loc   = np.array([wp_loc.x, wp_loc.y, wp_loc.z]) - perp_vector*self.offset
                self.z_spawn_height = self.closest_waypoint.transform.location.z + 1

                self.spawn_location = carla.Location(loc[0], loc[1], self.z_spawn_height)
                
                # spawn the pedestrian
                location = self.spawn_location
                rotation = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
                transform = carla.Transform(location, rotation)
                pedestrian = self.world.spawn_actor(self.walker_blueprint, transform)
                
                self.location = location #[location.x, location.y, location.z]
                self.rotation = rotation #[rotation.pitch, rotation.yaw, rotation.roll]
                self.actor    = pedestrian
                self.id       = pedestrian.id
                self.direction= carla.Vector3D(x=perp_vector[0],y=perp_vector[1], z=perp_vector[2])

                return pedestrian
            except:
                spawn_attempts += 1
                print("Pedestrian Spawn failed " + str(spawn_attempts) + " number of times. Retrying...")
        
        # print("Pedestrian Spawn Error, likely collision at all attempted spawn locations")

    def cross_road(self):
        self.speed = random.random() * self.max_speed
        walk  = carla.WalkerControl(self.direction, speed=self.speed, jump=False)

        self.actor.apply_control(walk)

    def destroy(self):
        if(self.actor is not None):
            self.actor.destroy()
            print('Pedestrian destroyed...')

if __name__ == '__main__':

    # connect to client
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    world = client.get_world()
    world.set_weather(getattr(carla.WeatherParameters, 'ClearNoon'))
    
    road_id = 19
    lane_id = -1

    pedestrian = DynamicPedestrian(world, road_id, lane_id)
    pedestrian.random_spawn()
    pedestrian.cross_road()
    input("press any key to destroy pedestrian")

    pedestrian.destroy()