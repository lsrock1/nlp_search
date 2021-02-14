import pickle
import os
import json


with open(os.path.join('data/word_count.pkl'), 'rb') as handle:
    words_count = pickle.load(handle)
with open(os.path.join('data/word_to_idx.pkl'), 'rb') as handle:
    word_to_idx = pickle.load(handle)
with open(os.path.join('data/special_case.pkl'), 'rb') as handle:
    special_case = pickle.load(handle)
# a = list(word_to_idx.keys())
# print(list(word_to_idx.keys()))

# for i in a:
#     if 'dark' in i:
#         print(i)
print(special_case)
with open('data/data/train-tracks.json') as f:
    tracks = json.load(f)
list_of_tracks = list(tracks.values())
for track in list_of_tracks:
    for n in track['nl']:
        if 'blue-black' in n.lower():# and 'black' in n.lower():
            print(n)
        # if 'pick' in n.lower():
        #     print(n)
#         if 'dark-' in n.lower():
#             print(n.replace('dark-red', 'dark-red'.replace('-', ' ')))
# print('pickup' in a)
# for l in a:
#     if l.startswith('cross'):
#         print(l)