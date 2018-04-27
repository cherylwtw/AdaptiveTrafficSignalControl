from gym.envs.registration import registry, register, make, spec
import xml.etree.ElementTree as ET

# Toy Text
# ----------------------------------------
def get_traffic_light_junction_lanes_dict():
    junction_lanes_dict = {}
    source_net_file = '/Users/cheryl/Documents/Python/767FinalProject/four_intersects/sumo_env/four_intersects.net.xml'
    tree = ET.parse(source_net_file)
    root = tree.getroot()
    for junction in root.findall('junction'):
        junction_type = junction.get('type')
        if junction_type == 'traffic_light':
            junction_id = junction.get('id')
            junction_inc_lanes_list = junction.get('incLanes').split(' ')
            junction_lanes_dict[junction_id] = junction_inc_lanes_list
    return junction_lanes_dict

register(
    id='FourIntersects-v1',
    entry_point='gym.envs.toy_text:FourIntersectsEnv',
    kwargs={'traffic_light_junction_lanes_dict' : get_traffic_light_junction_lanes_dict(),
            'version_num' : 1},
    max_episode_steps=1000,
)

register(
    id='FourIntersects-v2',
    entry_point='gym.envs.toy_text:FourIntersectsEnv',
    kwargs={'traffic_light_junction_lanes_dict' : get_traffic_light_junction_lanes_dict(),
            'version_num': 2},
    max_episode_steps=1000,
)

register(
    id='FourIntersects-v3',
    entry_point='gym.envs.toy_text:FourIntersectsEnv',
    kwargs={'traffic_light_junction_lanes_dict' : get_traffic_light_junction_lanes_dict(),
            'version_num' : 3},
    max_episode_steps=1000,
)

register(
    id='FourIntersects-v4',
    entry_point='gym.envs.toy_text:FourIntersectsEnv',
    kwargs={'traffic_light_junction_lanes_dict' : get_traffic_light_junction_lanes_dict(),
            'version_num' : 4},
    max_episode_steps=1000,
)

register(
    id='FourIntersects-v5',
    entry_point='gym.envs.toy_text:FourIntersectsEnv',
    kwargs={'traffic_light_junction_lanes_dict' : get_traffic_light_junction_lanes_dict(),
            'version_num' : 5},
    max_episode_steps=1000,
)

register(
    id='FourIntersects-v6',
    entry_point='gym.envs.toy_text:FourIntersectsEnv',
    kwargs={'traffic_light_junction_lanes_dict' : get_traffic_light_junction_lanes_dict(),
            'version_num' : 6},
    max_episode_steps=1000,
)

register(
    id='FourIntersects-v7',
    entry_point='gym.envs.toy_text:FourIntersectsEnv',
    kwargs={'traffic_light_junction_lanes_dict' : get_traffic_light_junction_lanes_dict(),
            'version_num' : 7},
    max_episode_steps=1000,
)

register(
    id='FourIntersects-v8',
    entry_point='gym.envs.toy_text:FourIntersectsEnv',
    kwargs={'traffic_light_junction_lanes_dict' : get_traffic_light_junction_lanes_dict(),
            'version_num' : 8},
    max_episode_steps=1000,
)

register(
    id='FourIntersects-v9',
    entry_point='gym.envs.toy_text:FourIntersectsEnv',
    kwargs={'traffic_light_junction_lanes_dict' : get_traffic_light_junction_lanes_dict(),
            'version_num' : 9},
    max_episode_steps=1000,
)

register(
    id='FourIntersects-v10',
    entry_point='gym.envs.toy_text:FourIntersectsEnv',
    kwargs={'traffic_light_junction_lanes_dict' : get_traffic_light_junction_lanes_dict(),
            'version_num' : 10},
    max_episode_steps=1000,
)
