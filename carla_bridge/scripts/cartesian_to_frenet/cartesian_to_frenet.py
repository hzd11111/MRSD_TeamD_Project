import sys
from typing import List, Tuple
from configparser import ConfigParser

config = ConfigParser()
config.read("config.ini")
CARLA_PATH = config.get("main", "CARLA_PATH")

# Enable import of 'carla'
sys.path.append(CARLA_PATH + "PythonAPI/carla/dist/carla-0.9.9-py3.6-linux-x86_64.egg")
# Enable import of 'agents' and it's submodules
sys.path.insert(0, CARLA_PATH + "PythonAPI/carla/")
# Enable import of utilities from GlobalPathPlanner
sys.path.insert(0, "../../../global_route_planner/")

import numpy as np
import carla
from agents.navigation.local_planner import LocalPlanner
from shapely.geometry import LineString, Point

from global_planner import get_client, spawn_vehicle, draw_waypoints


def get_frenet_from_cartesian(
    linestring: LineString,
    cartesian_point: Point,
    cartesian_heading: float = None,
    resolution: float = 0.01,
) -> Tuple[float, float, float, Point]:

    """
    Converts a point (x,y,heading) in the cartesian frame to it's frenet frame representation.

    :param linestring:
        A shapely 'LineString' (Polyline) that represents the centre line of the lane being used as
        the reference for the frenet frame.
    :param cartesian_point:
        A shapely 'Point' of the point in cartesian frame that has to be represented in the frenet frame.
    :param cartesian_heading:
        The heading associated with the cartesian point, in degrees.
    :param resolution:
        The resolution used to calculate the heading direction of local sections of the lane LineString.

    :returns:
        Tuple of (s, d, frenet_heading, projection of cartesian_point on linestring)
    """

    # Get s and d. d is not associated with a direction here.
    s = linestring.project(cartesian_point)
    d = linestring.distance(cartesian_point)

    # Find closest point (to the input point) on the lane LineString.
    closest_point_on_linestring = linestring.interpolate(s)

    # Find heading direction of the lane LineString by using a small section of the LineString,
    # around closest_point_on_linestring computed above.
    local_section_of_spline_start = linestring.interpolate(max(0, s - resolution))
    local_section_of_spline_end = linestring.interpolate(
        min(linestring.length, s + resolution)
    )
    local_section_heading_in_cartesian_coordinates = np.degrees(
        np.arctan2(
            local_section_of_spline_end.y - local_section_of_spline_start.y,
            local_section_of_spline_end.x - local_section_of_spline_start.x,
        )
    )

    # Find heading in frenet frame.
    heading_relative_to_local_section_heading = (
        cartesian_heading - local_section_heading_in_cartesian_coordinates
    )

    # Assign a direction (+ or -) to the distance d.
    heading_of_line_joining_input_points_and_its_closest_point_on_linestring = (
        np.degrees(
            np.arctan2(
                cartesian_point.y - closest_point_on_linestring.y,
                cartesian_point.x - closest_point_on_linestring.x,
            )
        )
    )
    relative_heading = (
        heading_of_line_joining_input_points_and_its_closest_point_on_linestring
        - local_section_heading_in_cartesian_coordinates
    )
    if relative_heading < 0 or relative_heading > 180:
        d = -1 * d

    return s, d, heading_relative_to_local_section_heading, closest_point_on_linestring


def get_cartesian_from_frenet(
    linestring: LineString,
    frenet_point: List[float],
    frenet_heading: float,
    resolution: float = 0.01,
) -> Tuple[float, float, float]:

    """
    Converts a point (s,d,frenet_heading) in the frenet frame to it's cartesian_frame representation.

    :param linestring:
        A shapely 'LineString' (Polyline) that represents the centre line of the lane being used as
        the reference for the frenet frame.
    :param frenet_point:
        A list of the form [s,d].
    :param frenet_heading:
        The heading associated with the frenet point, in degrees, in the frenet frame.
    :param resolution:
        The resolution used to calculate the heading direction of local sections of the lane LineString.

    :returns:
        Tuple of (x, y, heading), in the cartesian frame.
    """

    s = frenet_point[0]
    d = frenet_point[1]

    # Find point on the lane LineString at a runlength of s.
    point_on_linestring = linestring.interpolate(s)

    # Find heading direction of the lane LineString by using a small section of the LineString,
    # around point_on_linestring computed above.
    local_section_of_spline_start = linestring.interpolate(max(0, s - resolution))
    local_section_of_spline_end = linestring.interpolate(
        min(linestring.length, s + resolution)
    )
    local_section_heading_in_cartesian_coordinates = np.degrees(
        np.arctan2(
            local_section_of_spline_end.y - local_section_of_spline_start.y,
            local_section_of_spline_end.x - local_section_of_spline_start.x,
        )
    )

    # Get the cartesian point at offset by d, along a direction perpendicular to the heading of the local section of the lane LineString.
    angle_to_extend = (
        (local_section_heading_in_cartesian_coordinates + 90) * np.pi / 180
    )
    cartesian_point = [
        point_on_linestring.x + d * np.cos(angle_to_extend),
        point_on_linestring.y + d * np.sin(angle_to_extend),
    ]

    # Get the heading in cartesian frame.
    cartesian_heading = local_section_heading_in_cartesian_coordinates + frenet_heading

    return cartesian_point[0], cartesian_point[1], cartesian_heading


def get_path_linestring(waypoints: List[carla.libcarla.Waypoint]) -> LineString:

    coordinates = [
        [waypoint[0].transform.location.x, waypoint[0].transform.location.y]
        for waypoint in waypoints
    ]
    shapely_linestring = LineString(coordinates)
    return shapely_linestring


if __name__ == "__main__":

    # Get client to interact with CARLA server
    client = get_client()

    # Get current CARLA world
    world = client.get_world()

    # Spawn a vehicle at a random spawn point
    vehicle = spawn_vehicle(world=world)

    # Get a local planner for the vehicle
    opt_dict = {"target_speed": 20}
    local_planner = LocalPlanner(vehicle, opt_dict=opt_dict)

    # Get waypoins of the lane in which the vehicle has been spawned
    # local_planner._compute_next_waypoints(k=20)
    # current_path_waypoints = [item[0] for item in local_planner._waypoints_queue]dw

    # draw_waypoints(world, current_path_waypoints)

    # path_linestring = get_path_linestring(current_path_waypoints)

    # vehicle.destroy()
    # flag = False
    # while True:
    #     control = local_planner.run_step(debug=True)
    #     vehicle.apply_control(control)

    #         current_path_linestring = get_path_linestring(
    #             list(local_planner._waypoint_buffer)
    #         )
    #         # draw_waypoints(
    #         #     world, [item[0] for item in list(local_planner._waypoint_buffer)]
    #         # )

    #         # Find s, d (Frenet Coordinates)
    #         vehicle_location = vehicle.get_location()
    #         vehicle_location_point = Point(vehicle_location.x, vehicle_location.y)
    #         length_along_linestring = current_path_linestring.project(
    #             vehicle_location_point
    #         )

    #         print("s:", length_along_linestring)
    #         print("d:", current_path_linestring.distance(vehicle_location_point))
