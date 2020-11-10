import carla
import pickle
import sys

sys.path.append("../../carla_utils/utils")
sys.path.append("../../grasp_path_planner/scripts/")
from settings import STOP_LINE_DISTANCE_FOR_LANE_CHANGE_TERMINATE
from actors import Vehicle
from options import TrafficLightStatus

TOWN_O5_LIGHTS = 'traffic_lights/town05_traffic_light_x_V2.pkl'
LIGHTS_DICT = TOWN_O5_LIGHTS

class TrafficLightManager():
    def __init__(self, client, town="Town05"):
        
        self.client = client
        self.world = self.client.get_world()
        self.Map = self.world.get_map()
        
        # a tick is needed to update the actors in the world
        self.world.tick()

        if town is "Town05":
            self.lane_id_to_tl_x_loc_dict = pickle.load(open(LIGHTS_DICT, "rb"))
        else:
            raise NotImplementedError

        self.all_traffic_lights = self.world.get_actors().filter("traffic.traffic_light*")
        self.road_to_tl_actor_dict = self.get_road_to_tl_actorid_dict()

    def get_road_to_tl_actorid_dict(self):
        dic = {}
        
        for tl in self.all_traffic_lights:
            x = tl.get_location().x
            y = tl.get_location().y
            for key, value in self.lane_id_to_tl_x_loc_dict.items():
                if (x + y) == value:
                    dic[key] = tl.id
        
        return dic
    
    def get_actor_to_traffic_light(self, actor: carla.Vehicle):
        # extract the actor's current road_id, and lane_id
        location = actor.get_location()
        nearest_waypoint = self.Map.get_waypoint(location, project_to_road=True)
        road_id = nearest_waypoint.road_id
        lane_id = nearest_waypoint.lane_id
        
        # get the traffic light from the stored dictionary
        tl = self.get_traffic_light_from_road(road_id=road_id, lane_id=lane_id)

        return tl, nearest_waypoint

    def set_actor_traffic_light_state(self, actor:Vehicle, is_ego=False):
        '''Set the traffic_light_status parameter for a Vehicle object'''
        # create a carla actor object from the Vehicle class
        carla_actor = self.world.get_actor(actor.actor_id)
        
        # get the traffic light state
        tl, nearest_waypoint = self.get_actor_to_traffic_light(carla_actor)
        
        if tl is None:
            # print("tl is none for some reason")
            actor.traffic_light_status = TrafficLightStatus.GREEN
            # actor.traffic_light_stop_distance = -1
        else:
            state = str(tl.get_state())

            # get the traffic manager 
            if state == 'Red':
                actor.traffic_light_status = TrafficLightStatus.RED
            elif state == 'Yellow':
                actor.traffic_light_status = TrafficLightStatus.YELLOW
            elif state == 'Green':
                actor.traffic_light_status = TrafficLightStatus.GREEN
        
        if is_ego and tl is not None:
            dist = len(nearest_waypoint.next_until_lane_end(1)) - STOP_LINE_DISTANCE_FOR_LANE_CHANGE_TERMINATE
            dist = max(0, dist)
            actor.traffic_light_stop_distance = dist
            

    def set_traffic_light_for_vehicle(self, vehicle, state=TrafficLightStatus.GREEN):
        carla_actor = self.world.get_actor(vehicle.actor_id)
        # tl_state = carla_actor.get_traffic_light_state() 
        
        nearest_waypoint = self.Map.get_waypoint(carla_actor.get_location())
        road_id = nearest_waypoint.road_id
        lane_id = nearest_waypoint.lane_id

        tl = self.get_traffic_light_from_road(road_id, lane_id, raise_exception=False)
        if (tl == None):
            return
        if(state == TrafficLightStatus.GREEN):
            tl.set_state(carla.TrafficLightState.Green)
        if(state == TrafficLightStatus.RED):
            tl.set_state(carla.TrafficLightState.Red)
        if(state == TrafficLightStatus.YELLOW):
            tl.set_state(carla.TrafficLightState.Yellow)

    def get_traffic_light_from_road(self, road_id, lane_id, raise_exception=False):
        '''
        Inputs:
            road_id (int): CARLA road_id
            lane_id (int): CARLA lane_id
            raise_exception (bool): flag to raise an exception is road_id, lane_id has
                                    no traffic light
        Returns:
            carla.Actor obj of the traffic light
        '''
        actor_id = self.road_to_tl_actor_dict.get((road_id, lane_id))

        if actor_id is not None:
            return self.world.get_actor(actor_id)
        else:
            if raise_exception:
                raise Exception("No traffic light found for this road_id," \
                            "lane_id combo", road_id, lane_id)
            return None

if __name__ == "__main__":
    import string

    '''Testing this script with manual_control.py'''
    # Updating traffic light status for each vehicle takes 0.08 ms

    # connect to client, get world
    client = carla.Client('127.0.0.1', 2000)
    world = client.get_world()
    Map = world.get_map()
    tlm = TrafficLightManager(client)
    waypoint_list = Map.generate_waypoints(3)
    lf = 100 #lifetime of draw points

    # # get ego vehicle spawned by manual control
    # actor = world.get_actors().filter("vehicle*")[0]
    # ac = Vehicle(world, actor.id)
    
    # keep printing ROS msg for the ego vehicle, check for light status
    while True:

        tl_dict = {}
        tl_loc_dict = {}
        # tl = tlm.set_actor_traffic_light_state(ac, is_ego=True)
        # print(ac.toRosMsg())

        # label each point on map with a letter and its associated traffic light
        for waypoint in waypoint_list:
            
            # get a tl object
            road_id = waypoint.road_id
            lane_id = waypoint.lane_id
    
            # get the traffic light from the stored dictionary
            tl = tlm.get_traffic_light_from_road(road_id=road_id, lane_id=lane_id)
            
            # randomly pick two letters to draw per wp + tl combo
            letter = str(road_id)+ ',' + str(lane_id)

            # if a TL is found, draw a letter on the waypoint and the tl
            if tl is not None:
                tl_loc = tl.get_location()

                # waypoint location
                wp_loc = waypoint.transform.location
                world.debug.draw_string(wp_loc, letter, draw_shadow=False,
                                        color=carla.Color(r=0, g=255, b=0), life_time=lf)
                

                dic_key = str(tl.id)+letter
                if dic_key not in tl_dict:
                    # tl location
                    dis = lane_id
                    tl_loc.x += dis
                    tl_loc.y += dis
                    world.debug.draw_string(tl_loc, letter, draw_shadow=True,
                                        color=carla.Color(r=0, g=255, b=0), life_time=lf)
                    tl_dict[str(tl.id)+letter] = 1
            else:
                if waypoint.road_id > 61: continue
                # waypoint location
                wp_loc = waypoint.transform.location
                world.debug.draw_string(wp_loc, letter, draw_shadow=False,
                                        color=carla.Color(r=255, g=0, b=0), life_time=lf)


        # print locations over all traffic_lights
        for tl in world.get_actors().filter("traffic.traffic_light*"):
            tl_loc = tl.get_location()
            world.debug.draw_string(tl_loc, str(tl_loc.x + tl_loc.y), draw_shadow=True,
                                color=carla.Color(r=0, g=255, b=0), life_time=lf)
            tl_loc_dict[str(tl_loc.x)] = 1

        import ipdb; ipdb.set_trace()

        while False:
            rf_rate = 0.2
            spec = world.get_spectator()
            loc = spec.get_location()
            wp = Map.get_waypoint(loc, project_to_road = True)

            next_wp_locs = [wp.transform.location for wp in wp.next_until_lane_end(2)]
            prev_wp_locs = [wp.transform.location for wp in wp.previous_until_lane_start(2)]
            
            for wp_loc in next_wp_locs:
                world.debug.draw_string(wp_loc, 'x', draw_shadow=False,
                                        color=carla.Color(r=0, g=255, b=0), life_time=rf_rate)
            for wp_loc in prev_wp_locs:
                world.debug.draw_string(wp_loc, 'y', draw_shadow=False,
                                        color=carla.Color(r=255, g=0, b=0), life_time=rf_rate)

            import time; time.sleep(rf_rate)
