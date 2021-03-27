import json
from glob import glob
import numpy as np
from utils import stableMatching
import torch


all_f = glob('results/submit*.json')
saved = {}

for f in all_f:
    with open(f, 'r') as fp:
        saved.update(json.load(fp))

vehicle_uuids_per_nl = np.array(saved['uuids_order'])
del saved['uuids_order']

vehicle_colors = []
vehicle_types = []
for idx, vehicle_uuid in enumerate(vehicle_uuids_per_nl):
    cache_nl = torch.load(f'cache/9/{idx}_0.pth', map_location=torch.device('cpu'))
    vehicle_colors.append(cache_nl[1].numpy())
    vehicle_types.append(cache_nl[2].numpy())

vehicle_colors = np.array(vehicle_colors)
vehicle_types = np.array(vehicle_types)
print(f'vehicle color shape: {vehicle_colors.shape}')
print(f'vehicle type shape: {vehicle_types.shape}')

final_results = {}
# adjacency_matrix = []
nl_uuid = []
for uuid, probs in saved.items():
    nl_uuid.append(uuid)
#     adjacency_matrix.append(probs)

# adjacency_matrix = np.stack(adjacency_matrix, axis=0)
# matched = stableMatching(adjacency_matrix)
# print(matched)
# print(adjacency_matrix[0][matched[0]])
# for uuid, index, ad in zip(nl_uuid, matched, adjacency_matrix):
for uuid in nl_uuid:
    cache_nl = torch.load(f'cache/9/{uuid}.pth', map_location=torch.device('cpu'))
    _, nl_vehicle_type, nl_vehicle_color = cache_nl['nls'], cache_nl['type'], cache_nl['color']
    
    # color prob check
    if nl_vehicle_color != -1:
        color_weights = vehicle_colors[:, nl_vehicle_color] * 0.5
    else:
        color_weights = 0

    # type prob check
    if nl_vehicle_type != -1:
        type_weights = vehicle_types[:, nl_vehicle_type] * 0.5
    else:
        type_weights = 0

    probs = saved[uuid]
    prob_per_nl = np.array(probs) + color_weights + type_weights
    prob_per_nl_arg = (-prob_per_nl).argsort(axis=0)
    sorted_vehicle_uuids_per_nl = vehicle_uuids_per_nl[prob_per_nl_arg]
    sorted_vehicle_uuids_per_nl = sorted_vehicle_uuids_per_nl.tolist()
    # matched = sorted_uuids_per_nl[index]
    # del sorted_uuids_per_nl[index]
    # sorted_uuids_per_nl = [matched] + sorted_uuids_per_nl
    # sorted_uuids_per_nl[0], sorted_uuids_per_nl[index] = sorted_uuids_per_nl[index], sorted_uuids_per_nl[0]
    final_results[uuid] = sorted_vehicle_uuids_per_nl

with open('final_submission.json', 'w') as fp:
    json.dump(final_results, fp)