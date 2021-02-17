import pickle
import os
import json
from collections import Counter

with open(os.path.join('data/word_count.pkl'), 'rb') as handle:
    words_count = pickle.load(handle)
with open(os.path.join('data/word_to_idx.pkl'), 'rb') as handle:
    word_to_idx = pickle.load(handle)
with open(os.path.join('data/special_case.pkl'), 'rb') as handle:
    special_case = pickle.load(handle)
# a = list(word_to_idx.keys())
# print(list(word_to_idx.keys()))
print(list(special_case.keys()))
colors = ['silver', 'grey', 'red', 'white', 'whit', 'brown', 'maroon', 'gold','black', 'gray', 'reddish', 'blue', 'purple', 'lightgray', 'yellow', 'orange', 'green']
vehicle_type = ['truck', 'pickup','sedan', 'suv', 'spv', 'van', 'wagon', 'cargo', 'mpv', 'hatchback', 'hatckback', 'jeep', 'chevrolet', 'minivan', 'coup']
# for k in word_to_idx.keys():
#     print(k)
# for i in a:
#     if 'dark' in i:
#         print(i)
# print(special_case)
with open('data/data/train-tracks.json') as f:
    tracks = json.load(f)
list_of_tracks = list(tracks.values())


# color analyze
# count = 0
# for track in list_of_tracks:
#     colors_in_nls = list()
#     replace_list = []
#     for n in track['nl']:
#         n = n.lower()
#         colors_in_nl = list()
        
#         for c in colors:
#             index = n.find(c)
#             if index != -1:
#                 # colors_in_nls.add(c)
#                 colors_in_nl.append((c, index))
#         colors_in_nl = sorted(colors_in_nl, key=lambda x: x[1])
#         if len(colors_in_nl) > 0:
#             # select first color
#             # print(colors_in_nl)
#             colors_in_nls.append(colors_in_nl[0][0])
#             replace_list.append(colors_in_nl[0][0])
#         else:
#             replace_list.append(None)

#         # print(colors_in_nl)
    

#     if len(set(colors_in_nls)) > 1:
#         counters = Counter(colors_in_nls)
#         nls = [t.lower() for t in track['nl']]
#         k = counters.most_common(1)
#         representer_color = k[0][0]
#         new_nls = []
#         print(replace_list)
#         for c, n in zip(replace_list, nls):
#             if c != None:
#                 new_nls.append(n.replace(c, representer_color))
#             else:
#                 new_nls.append(n)
#         count += 1
#         print('-' * 14)
#         print(new_nls)
#         print(representer_color)
#         print('-'*14)
# print(count)

count = 0
for track in list_of_tracks:
    colors_in_nls = list()
    replace_list = []
    for n in track['nl']:
        n = n.lower()
        typess_in_nl = list()
        
        for c in vehicle_type:
            index = n.find(c)
            if index != -1:
                # colors_in_nls.add(c)
                typess_in_nl.append((c, index))
        typess_in_nl = sorted(typess_in_nl, key=lambda x: x[1])
        
        if len(typess_in_nl) > 0:
            first_type = typess_in_nl[0][0]
            if first_type == 'truck':
                first_type = 'pickup'
            elif first_type == 'minivan':
                first_type = 'van'
            # select first color
            # print(colors_in_nl)
            colors_in_nls.append(first_type)
            replace_list.append(first_type)
        else:
            replace_list.append(None)

        # print(colors_in_nl)
    

    if len(set(colors_in_nls)) > 1:
        counters = Counter(colors_in_nls)
        nls = [t.lower() for t in track['nl']]
        k = counters.most_common(1)
        representer_color = k[0][0]
        new_nls = []
        # print(replace_list)
        for c, n in zip(replace_list, nls):
            if c != None:
                new_nls.append(n.replace(c, representer_color))
            else:
                new_nls.append(n)
        count += 1
        # print('-' * 14)
        # print(new_nls)
        # print(representer_color)
        # print('-'*14)
    elif len(colors_in_nls) == 0:
        print(track['nl'])
print(count)