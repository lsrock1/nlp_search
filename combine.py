import json
from glob import glob
import numpy as np
from utils import stableMatching


all_f = glob('results/submit*.json')
saved = {}

for f in all_f:
    with open(f, 'r') as fp:
        saved.update(json.load(fp))

uuids_per_nl = np.array(saved['uuids_order'])
del saved['uuids_order']

final_results = {}
adjacency_matrix = []
nl_uuid = []
for uuid, probs in saved.items():
    nl_uuid.append(uuid)
    adjacency_matrix.append(probs)

adjacency_matrix = np.stack(adjacency_matrix, axis=0)
matched = stableMatching(adjacency_matrix)
# print(matched)
# print(adjacency_matrix[0][matched[0]])
for uuid, index, ad in zip(nl_uuid, matched, adjacency_matrix):
    probs = saved[uuid]
    prob_per_nl = np.array(probs)
    prob_per_nl_arg = (-prob_per_nl).argsort(axis=0)
    sorted_uuids_per_nl = uuids_per_nl[prob_per_nl_arg]
    sorted_uuids_per_nl = sorted_uuids_per_nl.tolist()
    # matched = sorted_uuids_per_nl[index]
    # del sorted_uuids_per_nl[index]
    # sorted_uuids_per_nl = [matched] + sorted_uuids_per_nl
    # sorted_uuids_per_nl[0], sorted_uuids_per_nl[index] = sorted_uuids_per_nl[index], sorted_uuids_per_nl[0]
    final_results[uuid] = sorted_uuids_per_nl

# uuids_per_nl = np.array(uuids_order)
# print(uuids_per_nl.shape)
# prob_per_nl = np.array(prob_per_nl)
# prob_per_nl_arg = (-prob_per_nl).argsort(axis=0)
# sorted_uuids_per_nl = uuids_per_nl[prob_per_nl_arg]
# print(prob_per_nl[prob_per_nl_arg])
# final_results[uuid] = sorted_uuids_per_nl.tolist()
# print(len(final_results.keys()))

with open('final_submission.json', 'w') as fp:
    json.dump(final_results, fp)