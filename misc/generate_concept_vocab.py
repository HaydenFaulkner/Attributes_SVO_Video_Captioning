import json
import os
import re

import pickle as pkl

import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')

NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']
VERBS = ['VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ']

def clean_document(document):
    """Remove enronious characters. Extra whitespace and stop words"""
    document = re.sub('[^A-Za-z .-]+', ' ', document)
    document = ' '.join(document.split())
    document = ' '.join([i for i in document.split() if i not in stop])
    return document


if __name__ == '__main__':

    num_occ = 20
    for dset in ['msvd', 'msrvtt']:

        # organise vocab
        vocab_nouns = dict()
        vocab_verbs = dict()
        cocofmt_file = os.path.join('datasets', dset, 'metadata', dset+'_train_cocofmt.json')
        gt = json.load(open(cocofmt_file))
        ids = [x['id'] for x in gt['images']]
        caps = dict()
        for c in gt['annotations']:
            cap = clean_document(c['caption'])  # removes some 179 stop words
            tags = nltk.pos_tag(nltk.word_tokenize(cap))

            for w in tags:
                if w[1] in NOUNS:
                    if w[0] not in vocab_nouns:
                        vocab_nouns[w[0]] = 0
                    vocab_nouns[w[0]] += 1
                if w[1] in VERBS:
                    if w[0] not in vocab_verbs:
                        vocab_verbs[w[0]] = 0
                    vocab_verbs[w[0]] += 1

        vocab_nouns = set([k for k, v in vocab_nouns.items() if v >= num_occ])
        vocab_verbs = set([k for k, v in vocab_verbs.items() if v >= num_occ])
        intersection = vocab_nouns.intersection(vocab_verbs)

        if dset == 'msvd':
            manual_nouns = set(['pudding', 'bread', 'cake', 'face', 'pan', 'cucumber', 'gun', 'pot', 'head', 'boy', 'egg', 'shrimp', 'piece', 'kitchen', 'lake', 'tail', 'kitten', 'stove', 'snow', 'bowl', 'bed', 'tree', 'snake', 'knife', 'soccer', 'potato', 'chicken', 'dog'])
            manual_verbs = set(['swimming', 'walks', 'cooking', 'plays', 'performing', 'pours', 'playing', 'ride', 'reading', 'eats', 'cleaning', 'runs', 'singing', 'shows', 'dices', 'opening', 'cutting', 'play', 'mixing', 'make', 'slices', 'landing', 'driving', 'jumping', 'dancing', 'walking', 'slicing', 'hits', 'drinking', 'jumps', 'making', 'eating', 'climbs', 'pushes', 'fight', 'barking', 'cut', 'walk', 'rides', 'biting', 'falls', 'chops', 'frying', 'boxing', 'eat', 'speaking', 'get', 'cuts', 'drink', 'show'])
        if dset == 'msrvtt':
            manual_nouns = set(['friend', 'stove', 'pot', 'smoke', 'pink', 'ping', 'set', 'kitchen', 'forest', 'head', 'dark', 'piece', 'clip', 'plate', 'bed', 'conversation', 'bikini', 'cartoon', 'basketball', 'sauce', 'mountain', 'spongebob', 'shorts', 'girl', 'store', 'sunglasses', 'pictures', 'child', 'guy', 'tv', 'wedding', 'motorcycle', 'band', 'train', 'wooden', 'makeup', 'match', 'pans', 'wall', 'recipe', 'lake', 'gym', 'orange', 'vs', 'camera', 'dish', 'toys', 'boy', 'bowl', 'dog', 'land', 'street', 'restaurant', 'beach', 'funny', 'fire', 'song', 'glasses', 'ring', 'piano', 'helmet', 'tie', 'cave', 'kitten', 'pirates', 'wave', 'mountains', 'video', 'friends', 'ocean', 'stage', 'ball', 'shirt', 'swim', 'chicken', 'hair', 'snake', 'see', 'gameplay', 'display', 'blonde', 'screen', 'cake', 'toy', 'murray', 'light', 'baby', 'middle', 'dresses', 'grey', 'scene', 'crowd', 'pan', 'advertising', 'character', 'gun', 'soccer', 'picture', 'ellen', 'lego', 'runway', 'love', 'flash', 'amazing', 'string', 'kid', 'images', 'face', 'side', 'trampoline', 'fun', 'flooring', 'boat', 'tennis', 'front', 'boys', 'yellow', 'horse', 'end', 'cat', 'cheer', 'sky', 'clothing', 'concert', 'hands', 'tree', 'blue', 'snow', 'track'])
            manual_verbs = set(['win', 'tells', 'poses', 'attack', 'applies', 'drinking', 'filming', 'exercises', 'sings', 'catch', 'jumping', 'driving', 'wins', 'watch', 'dance', 'building', 'lays', 'leaves', 'pushes', 'fall', 'preforms', 'plays', 'wearing', 'parking', 'standing', 'chases', 'scores', 'ride', 'stand', 'depicts', 'making', 'frying', 'asks', 'dancing', 'shopping', 'explains', 'speaking', 'sit', 'start', 'performing', 'cooks', 'jumps', 'drive', 'falls', 'flips', 'commentating', 'playing', 'smoking', 'uses', 'opening', 'watches', 'work', 'try', 'turns', 'wrestling', 'look', 'walking', 'interviews', 'works', 'discuss', 'move', 'fly', 'singing', 'drives', 'swimming', 'speak', 'drink', 'discussing', 'living', 'dances', 'teaches', 'looks', 'pass', 'review', 'rides', 'reads', 'fight', 'preparing', 'wear', 'painting', 'features', 'walk', 'jump', 'wears', 'stirs', 'help', 'flies', 'subscribe', 'travel', 'pulls', 'narrates', 'eating', 'reviews', 'hits', 'sees', 'fighting', 'stands', 'displaying', 'climbs', 'sits', 'displays', 'knocks', 'use', 'kisses', 'displayed', 'starts', 'runs', 'meet', 'picks', 'reading', 'laughing', 'make', 'reporting', 'run', 'prepares', 'kicks', 'watching', 'describes', 'discusses', 'shows', 'play', 'talk', 'roll', 'blow', 'let', 'cut', 'racing', 'show', 'spins', 'training', 'put', 'throws', 'walks', 'mixing', 'signing', 'boiling', 'cook', 'acts', 'prepare', 'stop', 'perform', 'tries', 'recording', 'check', 'turn', 'showing', 'throw', 'sing', 'cooking', 'cleaning', 'hold', 'pours', 'laughs', 'cast', 'hit', 'drops', 'performs', 'meeting', 'eats', 'explain', 'enjoy', 'fishing', 'moves', 'raps', 'talks', 'get', 'eat', 'speaks', 'shoot', 'puts', 'describing'])
        assert len(intersection-(manual_verbs.union(manual_nouns))) == 0

        vocab_nouns = vocab_nouns - manual_verbs
        vocab_verbs = vocab_verbs - manual_nouns

        with open(os.path.join('datasets', dset, 'metadata', dset+'_nouns_vocab.pkl'), 'wb') as f:
            pkl.dump(vocab_nouns, f)
        with open(os.path.join('datasets', dset, 'metadata', dset+'_verbs_vocab.pkl'), 'wb') as f:
            pkl.dump(vocab_verbs, f)
