import pickle
import os


with open(os.path.join('data/word_count.pkl'), 'rb') as handle:
    words_count = pickle.load(handle)
with open(os.path.join('data/word_to_idx.pkl'), 'rb') as handle:
    word_to_idx = pickle.load(handle)

a = list(word_to_idx.keys())
# print(list(word_to_idx.keys()))

# print('pickup' in a)
for l in a:
    if l.startswith('cross'):
        print(l)