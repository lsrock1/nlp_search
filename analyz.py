import pickle
import os
import json
from collections import Counter, defaultdict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from configs import get_default_config
from dataset import CityFlowNLDataset
from transformers import ElectraTokenizerFast


class NL:
    def __init__(self, cfg, tracks):
        self.cfg = cfg
        self.tracks = tracks
        self.s = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-small-discriminator')
        if os.path.exists(os.path.join(self.cfg.DATA.DICT_PATH, 'word_count.pkl')):
            with open(os.path.join(self.cfg.DATA.DICT_PATH, 'word_count.pkl'), 'rb') as handle:
                self.words_count = pickle.load(handle)
            with open(os.path.join(self.cfg.DATA.DICT_PATH, 'word_to_idx.pkl'), 'rb') as handle:
                self.word_to_idx = pickle.load(handle)
            with open(os.path.join(self.cfg.DATA.DICT_PATH, 'special_case.pkl'), 'rb') as handle:
                self.special_case = pickle.load(handle)
        else:
            self.words_count, self.word_to_idx = self.__build_dict(self.tracks)

    def __len__(self):
        return len(self.word_to_idx)

    def __build_dict(self, tracks):
        word_count = defaultdict(int)

        word_count['<SOS>'] += 1
        word_count['<EOS>'] += 1
        word_count['<PAD>'] += 1
        word_count['<UNK>'] += 1
        max_length = 0

        # special case handling
        except_case = ['dark-red', 'dark-blue', 'dark-colored']
        hand_case = {
            'hatckback': 'hatchback'
        }
        self.special_case = {}
        for t in tracks:
            for n in t['nl']:
                for word in n.lower()[:-1].split():
                    if '-' in word and word not in except_case and word not in self.special_case:
                        self.special_case[word.replace('-', ' ')] = word.replace('-', '')
        
        for ec in except_case:
            self.special_case[ec] = ec.replace('-', ' ')
        self.special_case.update(hand_case)
        
        # self.special_case = special_case
                        
        for t in tracks:
            for n in t['nl']:
                cleaned_sentence = self.do_clean(n)
                if len(cleaned_sentence) > max_length:
                    max_length = len(cleaned_sentence)
                for w in cleaned_sentence:
                # for l in n.replace('.', '').split():
                    word_count[w] += 1
        print('max: ', max_length)
        new_dict = dict()
        for k, v in word_count.items():
            if v >= self.cfg.DATA.MIN_COUNT or k in ['<SOS>', '<EOS>', '<PAD>', '<UNK>']:
                new_dict[k] = v
        
        word_count = new_dict

        word_to_idx = dict(zip(word_count.keys(), range(len(word_count))))

        with open(os.path.join(self.cfg.DATA.DICT_PATH, 'word_count.pkl'), 'wb') as handle:
            pickle.dump(word_count, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.cfg.DATA.DICT_PATH, 'word_to_idx.pkl'), 'wb') as handle:
            pickle.dump(word_to_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.cfg.DATA.DICT_PATH, 'special_case.pkl'), 'wb') as handle:
            pickle.dump(self.special_case, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        return word_count, word_to_idx

    def do_clean(self, nl):
        nl = nl.lower()
        for sc, replaced in self.special_case.items():
            if sc in nl:
                nl = nl.replace(sc, replaced)

        nl = nl[:-1].replace('-', '').split()
        # nl = [self.s.stem(w) for w in nl]
        # nl = [w for w in nl if w not in self.stop_words]
        return nl

    def sentence_to_index(self, nl, is_train=True):
        nl = self.do_clean(nl)

        idxs = [self.word_to_idx[n] if n in self.word_to_idx else self.word_to_idx['<UNK>'] for n in nl]
        
        if is_train:
            if len(idxs) > self.cfg.DATA.MAX_SENTENCE:
                idxs = idxs[:self.cfg.DATA.MAX_SENTENCE]
                idxs = [self.word_to_idx['<SOS>']] + idxs + [self.word_to_idx['<EOS>']]
            else:
                idxs = [self.word_to_idx['<SOS>']] + idxs + [self.word_to_idx['<EOS>']] + [self.word_to_idx['<PAD>'] for _ in range(self.cfg.DATA.MAX_SENTENCE - len(idxs))]
        else:
            idxs = [self.word_to_idx['<SOS>']] + idxs + [self.word_to_idx['<EOS>']]
        return idxs


cfg = get_default_config()
data_cfg = cfg
with open(data_cfg.DATA.JSON_PATH) as f:
    tracks = json.load(f)
list_of_uuids = list(tracks.keys())
list_of_tracks = list(tracks.values())
nl_class = NL(cfg, list_of_tracks)

maxlen = 0
for track_idx, track in enumerate(list_of_tracks):
    nl = track["nl"]
    nl, vehicle_type = CityFlowNLDataset.type_replacer(nl)
    # if vehicle_type == -1:
    #     continue
    nl, vehicle_color = CityFlowNLDataset.color_replacer(nl)
    # if vehicle_color == -1:
    for n in nl:
        n = nl_class.do_clean(n)
        n = ' '.join(n)
        n = nl_class.tokenizer.encode(n, padding='max_length', truncation=True, max_length=30)
        # n = nl_class.tokenizer.decode(n)
        print(n)
        # l = len(nl_class.do_clean(n))
        # maxlen = max([l, maxlen])
print(maxlen)
# with open(os.path.join('data/word_count.pkl'), 'rb') as handle:
#     words_count = pickle.load(handle)
# with open(os.path.join('data/word_to_idx.pkl'), 'rb') as handle:
#     word_to_idx = pickle.load(handle)
# with open(os.path.join('data/special_case.pkl'), 'rb') as handle:
#     special_case = pickle.load(handle)
# # a = list(word_to_idx.keys())
# # print(list(word_to_idx.keys()))
# print(list(special_case.keys()))
# colors = ['silver', 'grey', 'red', 'white', 'whit', 'brown', 'maroon', 'gold','black', 'gray', 'reddish', 'blue', 'purple', 'lightgray', 'yellow', 'orange', 'green']
# vehicle_type = ['truck', 'pickup','sedan', 'suv', 'spv', 'van', 'wagon', 'cargo', 'mpv', 'hatchback', 'hatckback', 'jeep', 'chevrolet', 'minivan', 'coup']
# # for k in word_to_idx.keys():
# #     print(k)
# # for i in a:
# #     if 'dark' in i:
# #         print(i)
# # print(special_case)
# with open('data/data/train-tracks.json') as f:
#     tracks = json.load(f)
# list_of_tracks = list(tracks.values())


# # color analyze
# # count = 0
# # for track in list_of_tracks:
# #     colors_in_nls = list()
# #     replace_list = []
# #     for n in track['nl']:
# #         n = n.lower()
# #         colors_in_nl = list()
        
# #         for c in colors:
# #             index = n.find(c)
# #             if index != -1:
# #                 # colors_in_nls.add(c)
# #                 colors_in_nl.append((c, index))
# #         colors_in_nl = sorted(colors_in_nl, key=lambda x: x[1])
# #         if len(colors_in_nl) > 0:
# #             # select first color
# #             # print(colors_in_nl)
# #             colors_in_nls.append(colors_in_nl[0][0])
# #             replace_list.append(colors_in_nl[0][0])
# #         else:
# #             replace_list.append(None)

# #         # print(colors_in_nl)
    

# #     if len(set(colors_in_nls)) > 1:
# #         counters = Counter(colors_in_nls)
# #         nls = [t.lower() for t in track['nl']]
# #         k = counters.most_common(1)
# #         representer_color = k[0][0]
# #         new_nls = []
# #         print(replace_list)
# #         for c, n in zip(replace_list, nls):
# #             if c != None:
# #                 new_nls.append(n.replace(c, representer_color))
# #             else:
# #                 new_nls.append(n)
# #         count += 1
# #         print('-' * 14)
# #         print(new_nls)
# #         print(representer_color)
# #         print('-'*14)
# # print(count)

# count = 0
# for track in list_of_tracks:
#     colors_in_nls = list()
#     replace_list = []
#     for n in track['nl']:
#         n = n.lower()
#         typess_in_nl = list()
        
#         for c in vehicle_type:
#             index = n.find(c)
#             if index != -1:
#                 # colors_in_nls.add(c)
#                 typess_in_nl.append((c, index))
#         typess_in_nl = sorted(typess_in_nl, key=lambda x: x[1])
        
#         if len(typess_in_nl) > 0:
#             first_type = typess_in_nl[0][0]
#             if first_type == 'truck':
#                 first_type = 'pickup'
#             elif first_type == 'minivan':
#                 first_type = 'van'
#             # select first color
#             # print(colors_in_nl)
#             colors_in_nls.append(first_type)
#             replace_list.append(first_type)
#         else:
#             replace_list.append(None)

#         # print(colors_in_nl)
    

#     if len(set(colors_in_nls)) > 1:
#         counters = Counter(colors_in_nls)
#         nls = [t.lower() for t in track['nl']]
#         k = counters.most_common(1)
#         representer_color = k[0][0]
#         new_nls = []
#         # print(replace_list)
#         for c, n in zip(replace_list, nls):
#             if c != None:
#                 new_nls.append(n.replace(c, representer_color))
#             else:
#                 new_nls.append(n)
#         count += 1
#         # print('-' * 14)
#         # print(new_nls)
#         # print(representer_color)
#         # print('-'*14)
#     elif len(colors_in_nls) == 0:
#         print(track['nl'])
# print(count)