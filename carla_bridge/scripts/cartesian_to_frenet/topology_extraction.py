import sys
import copy
from typing import Dict, Any, List, Tuple
from configparser import ConfigParser
import xml.etree.ElementTree as ET


import carla

from cartesian_to_frenet import get_frenet_from_cartesian, get_path_linestring


def get_opendrive_tree(world: carla.libcarla.World) -> ET.ElementTree:
    """
    Parses the OpenDrive XML string for the current CARLA World and returns it as an ET.ElementTree.
    """

    xodr = world.get_map().to_opendrive()
    tree = ET.ElementTree(ET.fromstring(xodr))

    return tree


def get_junction_topology(
    tree: ET.ElementTree,
) -> Dict[int, List[List[Tuple[int, int]]]]:
    """
    Processes the OpenDrive XML Tree to get the topology for all junctions in the CARLA Map.
    The output dictionary has the following structure:

    Dict[junction_id] = [[(Road1, Road1_lane), (Road2, Road2_lane)], .....].

    junction_id is the integer representing the ID of the junction.
    (Road1, Road1_lane) represents an incoming road + lane combination. Incoming means that the road flows into the intersection.
    (Road2, Road2_lane) represents a road + lane combination within the intersection. Think of this as a lane that lies completely within the junction's bounding box.

    """

    junction_topology = {}

    all_junctions = tree.findall("junction")

    for junction in all_junctions:

        ID = int(junction.get("id"))
        connections = []

        all_connections = junction.findall("connection")
        for connection in all_connections:
            incomingRoad = int(connection.get("incomingRoad"))
            connectingRoad = int(connection.get("connectingRoad"))
            contactPoint = connection.get("contactPoint")
            laneLinks = connection.findall("laneLink")

            for laneLink in laneLinks:
                from_lane_id = int(laneLink.get("from"))
                to_lane_id = int(laneLink.get("to"))
                connections.append(
                    [
                        (incomingRoad, from_lane_id),
                        (connectingRoad, to_lane_id),
                        contactPoint,
                    ]
                )

        junction_topology[ID] = connections

    return junction_topology


def get_junction_roads_topology(
    tree: ET.ElementTree,
) -> Dict[int, Tuple[int, int, List[List[int]]]]:

    """
    Processes the OpenDrive XML Tree to get the topology for all roads that are connected to junctions in the CARLA Map.
    The output dictionary has the following structure:

    Dict[Road_ID] = (Predecessor_Road_ID, Successor_Road_ID, [LaneID_1, LaneID_2, ....], .....)

    Road_ID is the integer road ID for a connecting road within the intersection. This is a road that lies completely within the junction
            and connects two lanes on different roads.
    Predecessor_Road_ID is the road ID for the road that is the predecessor of the road represented by Road_ID. The 'Road_ID' road forms a connection between this road and the successor road.
    Successor_Road_ID is the road ID for the road that is the successor of the road represented by Road_ID. The 'Road_ID' road forms a connection between this road and the predecessor road.
    """

    road_data = {}

    all_roads = tree.findall("road")
    for road in all_roads:

        if road.get("junction") == "-1":
            continue

        ID = int(road.get("id"))
        link = road.find("link")
        predecessor_road = link.find("predecessor")
        successor_road = link.find("successor")

        p_ID = int(predecessor_road.get("elementId"))
        s_ID = int(successor_road.get("elementId"))

        lanes = road.findall(".//lane")

        local_connections = []
        direct_connections = []

        for lane in lanes:
            if lane.get("type") != "driving":
                continue

            lane_ID = int(lane.get("id"))
            lane_link = lane.find("link")

            if lane_link is not None:
                pred_lane_ID = int(lane_link.find("predecessor").get("id"))
                succ_lane_ID = int(lane_link.find("successor").get("id"))
                direction = lane.find("userData").find("vectorLane").get("travelDir")

                local_connections.append(
                    [direction, pred_lane_ID, lane_ID, succ_lane_ID]
                )

            else:
                direct_connections.append(["forward", lane_ID, lane_ID, lane_ID])

        connections = []
        for item in copy.deepcopy(local_connections):
            flag = 0
            for connection in connections:

                if connection[-1] == item[2] and connection[-2] == item[1]:
                    flag = 1
                    connection.append(item[3])
            if flag == 0:
                connections.append(item)

        road_data[ID] = (p_ID, s_ID, connections + direct_connections)

    return road_data