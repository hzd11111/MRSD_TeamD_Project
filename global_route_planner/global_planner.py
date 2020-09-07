import random
import sys
from typing import Optional, List
from configparser import ConfigParser

config = ConfigParser()
config.read("config.ini")
CARLA_PATH = config.get("main", "CARLA_PATH")

# Enable import of 'carla'
sys.path.append(CARLA_PATH + "PythonAPI/carla/dist/carla-0.9.9-py3.6-linux-x86_64.egg")
# Enable import of 'agents' and it's submodules
sys.path.insert(0, CARLA_PATH + "PythonAPI/carla/")

import carla
import agents
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.navigation.global_route_planner import GlobalRoutePlanner


def destroy_all_actors(world: carla.libcarla.World) -> None:
    """
    Destroys all actors in the given CARLA world.
    """
    for actor in world.get_actors():
        actor.destroy()


def draw_waypoints(
    world: carla.libcarla.World,
    waypoints: List[carla.libcarla.Waypoint],
    life_time: float = 30.0,  # Seconds
) -> None:
    """
    Draws a list of waypoints in the given CARLA world.
    """
    for waypoint in waypoints:
        world.debug.draw_string(
            waypoint.transform.location,
            "O",
            draw_shadow=False,
            color=carla.Color(r=0, g=255, b=0),
            life_time=life_time,
            persistent_lines=True,
        )


def get_client() -> carla.libcarla.Client:
    """
    Get a CARLA client object. The client object enables interaction with the CARLA server.
    """
    client = carla.Client("localhost", 2000)
    client.set_timeout(2.0)
    print("Connection to CARLA server established!")
    return client


def get_global_planner(
    world: carla.libcarla.World, planner_resolution: float
) -> agents.navigation.global_route_planner.GlobalRoutePlanner:
    """
    Get a global route planner object. The planner object can be used to retrieve point to point routes and route topologies.
    """
    world_map = world.get_map()
    dao = GlobalRoutePlannerDAO(world_map, planner_resolution)
    grp = GlobalRoutePlanner(dao)
    grp.setup()
    return grp


def spawn_vehicle(
    world: carla.libcarla.World,
    vehicle_type: str = "model3",
    spawn_point: Optional[carla.libcarla.Transform] = None,
) -> carla.libcarla.Vehicle:
    """
    Spawns a vehicle at a given spawn point. Default car model is 'model3'.
    If no spawn point is provided, randomly selects the spawn point from the set of pre-assigned spawn points in the map.
    """

    if spawn_point is None:
        spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle_blueprint = world.get_blueprint_library().filter(vehicle_type)[0]
    vehicle = world.spawn_actor(vehicle_blueprint, spawn_point)
    return vehicle


if __name__ == "__main__":

    # Get client to interact with CARLA server
    client = get_client()

    # Get current CARLA world
    world = client.get_world()

    # Destroy all actors
    # destroy_all_actors(world)

    # Get global route planner
    grp = get_global_planner(world=world, planner_resolution=2.0)

    # Spawn two vehicles, these signify the start and end points of our global path.
    spawned_vehicle_start = spawn_vehicle(world=world)
    spawned_vehicle_end = spawn_vehicle(world=world)
    start_location = (
        world.get_map()
        .get_waypoint(spawned_vehicle_start.get_location(), project_to_road=True)
        .transform.location
    )
    end_location = (
        world.get_map()
        .get_waypoint(spawned_vehicle_end.get_location(), project_to_road=True)
        .transform.location
    )

    # Generate the route using the global route planner object.
    route = grp.trace_route(start_location, end_location)
    route_waypoints = [route[i][0] for i in range(len(route))]

    # Draw the generated waypoints on the CARLA world.
    draw_waypoints(world, route_waypoints)

    # Print the (waypoint, topology) tuples
    for waypoint_topology_tuple in route:
        print(waypoint_topology_tuple)