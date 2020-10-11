#!/usr/bin/env python

# author: scott
# date 10.08.2020
# TODO: MAX SPEED LIMIT IMPLEMENTATION 
# TODO: MAYBE randomization distancing with range input
# Needs Carla 0.9.8 to run 

"""Spawn NPCs into the simulation"""

import glob
import os
import sys
import time
import random
from configparser import ConfigParser
config = ConfigParser()
config.read("config.ini")

import carla
from topology_extraction import get_opendrive_tree, get_junction_topology, get_junction_roads_topology
from topology_extraction import get_client, draw_waypoints, spawn_vehicle

import argparse
import logging
import random


def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=30,
        type=int,
        help='number of vehicles (default: 50)')
    argparser.add_argument(
        '-s', '--max-vehicles-speed',
        metavar='S',
        default=30,
        type=int,
        help='max speed of vehicles (default: 30)')
    argparser.add_argument(
        '-d', '--min-vehicles-distance',
        metavar='D',
        default=2.0,
        type=float,
        help='min distance between vehicles (default: 2.0)')
    argparser.add_argument(
        '-w', '--number-of-walkers',
        metavar='W',
        default=50,
        type=int,
        help='number of walkers (default: 50)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        default=True,
        help='avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--follow-traffic-rules',
        action='store_true',
        default=True,
        help='force scenario vechiles to follow the traffic rules')
    argparser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.*',
        help='vehicles filter (default: "vehicle.*")')
    argparser.add_argument(
        '--filterw',
        metavar='PATTERN',
        default='walker.pedestrian.*',
        help='pedestrians filter (default: "walker.pedestrian.*")')
    argparser.add_argument(
        '-tm_p', '--tm_port',
        metavar='P',
        default=8000,
        type=int,
        help='port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    args = argparser.parse_args()
    
    #Get the Road/Lane Ids to generate Intersection
    world = get_client().get_world()
    tree = get_opendrive_tree(world)
    junction_topology = get_junction_topology(tree)
    road_topology = get_junction_roads_topology(tree)
    print("available junction choices: " ,junction_topology.keys())
    junction_id, _ = random.choice(list(junction_topology.items()))
    print("generating scenario at junction id: " , junction_id)


    road_id_set = set([])

    for idx in range(len(junction_topology[junction_id])):
        intersection_road_id, _ = junction_topology[junction_id][idx][1]

        road_1_id, road_2_id, _ = road_topology[intersection_road_id]
        road_id_set.add(road_1_id)
        road_id_set.add(road_2_id)


    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    vehicles_list = []
    walkers_list = []
    all_id = []
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    ego_vehicle = None
    try:

        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(args.min_vehicles_distance)
        world = client.get_world()

        synchronous_master = False

        if args.sync:
            settings = world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
                world.apply_settings(settings)
            else:
                synchronous_master = False

        blueprints = world.get_blueprint_library().filter(args.filterv)
        blueprintsWalkers = world.get_blueprint_library().filter(args.filterw)

        if args.safe:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
            blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('t2')]

        waypoints = world.get_map().generate_waypoints(distance=1)
        road_waypoints = []
        for waypoint in waypoints:
            # if(waypoint.road_id <=2 and waypoint.lane_id == 2):
            if(waypoint.road_id in road_id_set):
                road_waypoints.append(waypoint)
        number_of_spawn_points = len(road_waypoints)
        print("get", len(road_waypoints), "on road 0 with lane id 2")
        if args.number_of_vehicles < number_of_spawn_points:
            print("randomize distance in between cars")
            random.shuffle(road_waypoints)
        elif args.number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, args.number_of_vehicles, number_of_spawn_points)
            args.number_of_vehicles = number_of_spawn_points

        
        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        for n, t in enumerate(road_waypoints):
            if n >= args.number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            transform = t.transform
            transform.location.z += 2.0
            batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)
                print('created %s' % response.actor_id)
        my_vehicles = world.get_actors(vehicles_list)
        print(len(my_vehicles))
        for n, v in enumerate(my_vehicles):
            # c = v.get_physics_control()
            # c.max_rpm = args.max_vehicles_speed * 133 
            # v.apply_physics_control(c)
            # if n == 0:
            #     print("vehicles' speed limit:", v.get_speed_limit())
            traffic_manager.auto_lane_change(v,False)
            if args.follow_traffic_rules is not True:
                print("breaking traffic rules") 
                traffic_manager.auto_lane_change(v,True)
                traffic_manager.ignore_lights_percentage(v,100)
                traffic_manager.distance_to_leading_vehicle(v,1)
            if ego_vehicle is not None:
                traffic_manager.collision_detection(v, ego_vehicle, True)

        # -------------
        # Spawn Walkers
        # -------------
        # some settings
        percentagePedestriansRunning = 0.0      # how many pedestrians will run
        percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(args.number_of_walkers):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id
        # 4. we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
        all_actors = world.get_actors(all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        if not args.sync or not synchronous_master:
            world.wait_for_tick()
            print("wait for tick")
        else:
            world.tick()
            print("client tick")


        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(vehicles_list), len(walkers_list)))

        # example of how to use parameters
        traffic_manager.global_percentage_speed_difference(30.0)

        while True:
            if args.sync and synchronous_master:
                world.tick()
            else:
                world.wait_for_tick()


    finally:

        if args.sync and synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()

        print('\ndestroying %d walkers' % len(walkers_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

        time.sleep(0.5)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
