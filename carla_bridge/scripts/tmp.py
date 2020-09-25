import sys
from typing import List
sys.path.append("/home/grasp/carla/PythonAPI/carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg")
sys.path.append("/home/grasp/carla/PythonAPI/carla")

import carla



def draw_waypoints(waypoints, world, road_id=None, life_time=50.0, lane_id=None):

    for waypoint in waypoints:
    
        if(waypoint.road_id == road_id and waypoint.lane_id == lane_id):
            world.debug.draw_string(waypoint.transform.location, 'O', draw_shadow=False,
                                        color=carla.Color(r=0, g=255, b=0), life_time=life_time,
                                        persistent_lines=True)

def get_client() -> carla.libcarla.Client:
    """
    Get a CARLA client object. The client object enables interaction with the CARLA server.
    """
    client = carla.Client("localhost", 2000)
    client.set_timeout(2.0)
    print("Connection to CARLA server established!")
    return client



if __name__ == "__main__":

    # Get client to interact with CARLA server
    client = get_client()

    # Get current CARLA world
    world = client.get_world()

    waypoints = world.get_map().generate_waypoints(distance=1)

    draw_waypoints(waypoints, world, 37, 20, -3)