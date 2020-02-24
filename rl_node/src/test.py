import rosbag

messages = rosbag.Bag("msg.bag")
env_msgs = [msg for _,msg,_ in messages.read_messages(['/environment_state'])]
data = env_msgs[0]
env_state = []

def append_vehicle_state(env_state, vehicle_state):
    env_state.append(vehicle_state.vehicle_location.x)
    env_state.append(vehicle_state.vehicle_location.y)
    env_state.append(vehicle_state.vehicle_location.theta)
    env_state.append(vehicle_state.vehicle_speed)
    return
    
# items needed
# current vehicle velocity
env_state.append(data.cur_vehicle_state.vehicle_speed)
# rear vehicle state
    # position
    # velocity
append_vehicle_state(env_state, data.back_vehicle_state)
# adjacent vehicle state
    # position
    # velocity
for veh_state in data.adjacent_lane_vehicles:
    append_vehicle_state(env_state, veh_state)


