import json
import os
import re

import numpy as np
import pickle as pkl

import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')
from random import randrange
import statistics

import utils

NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']
VERBS = ['VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ']

def clean_document(document):
    """Remove enronious characters. Extra whitespace and stop words"""
    document = re.sub('[^A-Za-z .-]+', ' ', document)
    document = ' '.join(document.split())
    document = ' '.join([i for i in document.split() if i not in stop])
    return document

def comp_nouns_verbs(pred, gt, noun_mets=None, verb_mets=None, glove_emb=None):
    pred = clean_document(pred)  # removes some 179 stop words
    tags = nltk.pos_tag(nltk.word_tokenize(pred))

    pred_nouns = list()
    pred_verbs = list()
    for w in tags:
        if w[1] in NOUNS:
            pred_nouns.append(w[0])
        if w[1] in VERBS:
            pred_verbs.append(w[0])

    gt_nouns = dict()
    gt_verbs = dict()
    for gtt in gt:
        gtt = clean_document(gtt)  # removes some 179 stop words
        tags = nltk.pos_tag(nltk.word_tokenize(gtt))
        for w in tags:
            if w[1] in NOUNS:
                if w[0] not in gt_nouns:
                    gt_nouns[w[0]] = 0
                gt_nouns[w[0]] += 1
            if w[1] in VERBS:
                if w[0] not in gt_verbs:
                    gt_verbs[w[0]] = 0
                gt_verbs[w[0]] += 1

    top_nouns = [x[0] for x in sorted(gt_nouns.items(), key=lambda item: item[1], reverse=True)[:3]]
    top_verbs = [x[0] for x in sorted(gt_verbs.items(), key=lambda item: item[1], reverse=True)[:1]]

    tp = 0
    fp = 0
    fn = 0
    if noun_mets is None:
        noun_mets = dict()
    for pn in pred_nouns:
        if pn not in noun_mets:
            noun_mets[pn] = [0, 0, 0]
        if pn in top_nouns:
            tp += 1
            noun_mets[pn][0] += 1
        else:
            fp += 1
            noun_mets[pn][1] += 1
    for gn in top_nouns:
        if gn not in noun_mets:
            noun_mets[gn] = [0, 0, 0]
        if gn not in pred_nouns:
            fn += 1
            noun_mets[gn][2] += 1

    if tp + fp > 0:
        noun_prec = float(tp)/(tp+fp)
    else:
        noun_prec = 0
    if tp + fn > 0:
        noun_rec = float(tp)/(tp+fn)
    else:
        noun_rec = 0
    if noun_rec + noun_prec > 0:
        noun_f1 = 2 * ((noun_prec * noun_rec) / (noun_prec + noun_rec))
    else:
        noun_f1 = 0

    tp = 0
    fp = 0
    fn = 0
    if verb_mets is None:
        verb_mets = dict()
    for pv in pred_verbs:
        if pv not in verb_mets:
            verb_mets[pv] = [0, 0, 0]
        if pv in top_verbs:
            tp += 1
            verb_mets[pv][0] += 1
        else:
            fp += 1
            verb_mets[pv][1] += 1
    for gv in top_verbs:
        if gv not in verb_mets:
            verb_mets[gv] = [0, 0, 0]
        if gv not in pred_verbs:
            fn += 1
            verb_mets[gv][2] += 1

    if tp + fp > 0:
        verb_prec = float(tp) / (tp + fp)
    else:
        verb_prec = 0
    if tp + fn > 0:
        verb_rec = float(tp) / (tp + fn)
    else:
        verb_rec = 0
    if verb_rec + verb_prec > 0:
        verb_f1 = 2 * ((verb_prec * verb_rec) / (verb_prec + verb_rec))
    else:
        verb_f1 = 0

    if glove_emb is None and os.path.exists(os.path.join('../glove6b', 'glove.6B.300d.pkl')):
        glove_emb = pkl.load(open(os.path.join('../glove6b', 'glove.6B.300d.pkl'), 'rb'))

    pred_nouns_glove = list()
    for pn in pred_nouns:
        if pn in glove_emb:
            pred_nouns_glove.append(glove_emb[pn])
    pred_verbs_glove = list()
    for pv in pred_verbs:
        if pv in glove_emb:
            pred_verbs_glove.append(glove_emb[pv])
    gt_nouns_glove = list()
    for gn in top_nouns:
        if gn in glove_emb:
            gt_nouns_glove.append(glove_emb[gn])
    gt_verbs_glove = list()
    for gv in top_verbs:
        if gv in glove_emb:
            gt_verbs_glove.append(glove_emb[gv])

    noun_dists = list()
    if len(gt_nouns_glove) > 0:
        gt_nouns_glove = np.array(gt_nouns_glove)
        for png in pred_nouns_glove:
            dist = np.linalg.norm(np.array(png) - gt_nouns_glove, axis=1, keepdims=True)
            noun_dists.append(np.min(dist))

    verb_dists = list()
    if len(gt_verbs_glove) > 0:
        gt_verbs_glove = np.array(gt_verbs_glove)
        for pvg in pred_verbs_glove:
            dist = np.linalg.norm(np.array(pvg) - gt_verbs_glove, axis=1, keepdims=True)
            verb_dists.append(np.min(dist))

    return [noun_prec, verb_prec, noun_rec, verb_rec, noun_f1, verb_f1, np.mean(noun_dists), np.mean(verb_dists)], noun_mets, verb_mets


if __name__ == '__main__':

    dset = 'msvd'
    # dset = 'msrvtt'
    # split = 'train'
    split = 'test'

    if 0:  # WNV DISTRIBUTIONS
        # setup paths
        cocofmt_file = os.path.join('datasets', dset, 'metadata', dset+'_'+split+'_cocofmt.json')
        # cocofmt_file = os.path.join('datasets', 'msrvtt', 'metadata', 'msrvtt_val_cocofmt.json')
        pred_file = os.path.join('experiments', 'lstm00_1', dset+'_'+split+'.json')

        # load the real ground truth
        gt = json.load(open(cocofmt_file))
        ids = [x['id'] for x in gt['images']]
        caps = dict()
        for c in gt['annotations']:
            if c['image_id'] not in caps:
                caps[c['image_id']] = []
            caps[c['image_id']].append(c['caption'])

        num_words = []
        num_nouns = []
        num_verbs = []
        for k, v in caps.items():
            for c in v:
                num_words.append(len(c))

                c = clean_document(c)  # removes some 179 stop words
                tags = nltk.pos_tag(nltk.word_tokenize(c))

                num_nouns_cap = 0
                num_verbs_cap = 0
                for w in tags:
                    if w[1] in NOUNS:
                        num_nouns_cap += 1
                    if w[1] in VERBS:
                        num_verbs_cap += 1

                num_nouns.append(num_nouns_cap)
                num_verbs.append(num_verbs_cap)

        print('GT')
        print('words mean and std:', statistics.mean(num_words), statistics.stdev(num_words))
        print('nouns mean and std:', statistics.mean(num_nouns), statistics.stdev(num_nouns))
        print('verbs mean and std:', statistics.mean(num_verbs), statistics.stdev(num_verbs))

        word_hist = dict()
        noun_hist = dict()
        verb_hist = dict()
        for w in num_words:
            if w not in word_hist:
                word_hist[w] = 0
            word_hist[w] += 1
        for w in num_nouns:
            if w not in noun_hist:
                noun_hist[w] = 0
            noun_hist[w] += 1
        for w in num_verbs:
            if w not in verb_hist:
                verb_hist[w] = 0
            verb_hist[w] += 1

        print('word histo')
        for w in range(max(word_hist.keys())+1):
            if w in word_hist:
                print(w, '\t', word_hist[w])
            else:
                print(w, '\t0')
        print('noun histo')
        for w in range(max(noun_hist.keys())+1):
            if w in noun_hist:
                print(w, '\t', noun_hist[w])
            else:
                print(w, '\t0')
        print('verb histo')
        for w in range(max(verb_hist.keys())+1):
            if w in verb_hist:
                print(w, '\t', verb_hist[w])
            else:
                print(w, '\t0')

        caps_pr = dict()
        pr = json.load(open(pred_file))
        for p in pr:
            caps_pr[p['image_id']] = [p['caption']]
        print()

        num_words = []
        num_nouns = []
        num_verbs = []
        for k, v in caps_pr.items():
            for c in v:
                num_words.append(len(c))

                c = clean_document(c)  # removes some 179 stop words
                tags = nltk.pos_tag(nltk.word_tokenize(c))

                num_nouns_cap = 0
                num_verbs_cap = 0
                for w in tags:
                    if w[1] in NOUNS:
                        num_nouns_cap += 1
                    if w[1] in VERBS:
                        num_verbs_cap += 1

                num_nouns.append(num_nouns_cap)
                num_verbs.append(num_verbs_cap)

        print('PR')
        print('words mean and std:', statistics.mean(num_words), statistics.stdev(num_words))
        print('nouns mean and std:', statistics.mean(num_nouns), statistics.stdev(num_nouns))
        print('verbs mean and std:', statistics.mean(num_verbs), statistics.stdev(num_verbs))

        word_hist = dict()
        noun_hist = dict()
        verb_hist = dict()
        for w in num_words:
            if w not in word_hist:
                word_hist[w] = 0
            word_hist[w] += 1
        for w in num_nouns:
            if w not in noun_hist:
                noun_hist[w] = 0
            noun_hist[w] += 1
        for w in num_verbs:
            if w not in verb_hist:
                verb_hist[w] = 0
            verb_hist[w] += 1

        print('word histo')
        for w in range(max(word_hist.keys())+1):
            if w in word_hist:
                print(w, '\t', word_hist[w])
            else:
                print(w, '\t0')
        print('noun histo')
        for w in range(max(noun_hist.keys())+1):
            if w in noun_hist:
                print(w, '\t', noun_hist[w])
            else:
                print(w, '\t0')
        print('verb histo')
        for w in range(max(verb_hist.keys())+1):
            if w in verb_hist:
                print(w, '\t', verb_hist[w])
            else:
                print(w, '\t0')

    if 1:  #CONCEPT SCORES
        ######################################## Compare training scores with overfitting
        # load the concept vocabs
        with open(os.path.join('../datasets', dset, 'metadata', dset + '_nouns_vocab.pkl'), 'rb') as f:
            vocab_nouns = pkl.load(f)
        with open(os.path.join('../datasets', dset, 'metadata', dset + '_verbs_vocab.pkl'), 'rb') as f:
            vocab_verbs = pkl.load(f)

        # setup paths
        cocofmt_file = os.path.join('../datasets', dset, 'metadata', dset + '_' + split + '_cocofmt.json')

        if os.path.exists(os.path.join('../glove6b', 'glove.6B.300d.pkl')):
            glove_emb = pkl.load(open(os.path.join('../glove6b', 'glove.6B.300d.pkl'), 'rb'))

        # load the real ground truth
        gt = json.load(open(cocofmt_file))
        ids = [x['id'] for x in gt['images']]
        caps = dict()
        for c in gt['annotations']:
            if c['image_id'] not in caps:
                caps[c['image_id']] = []
            caps[c['image_id']].append(c['caption'])

        # ###### RUN HUMAN NOUN VERB CHECK
        # # how many samples to run
        # if dset in ['msvd', 'MSVD']:
        #     runs = 100  # 100 for MSVD
        # else:
        #     runs = 20  # 20 for MSRVTT
        #
        #
        # # lets do random sampling numerous times and avg the results
        # scores = list()
        # for run in range(runs):
        #     print('run ', run)
        #
        #     # run through each clip
        #     scores_inner = list()
        #     for id in ids:
        #         ground_truth = list()
        #
        #         # randomly select one of the ground truth captions for this clip
        #         sample_index = randrange(len(caps[id]))
        #         sample_index_gt = sample_index
        #         while sample_index_gt == sample_index:  # ensure diff one
        #             sample_index_gt = randrange(len(caps[id]))
        #
        #         # append the caption to either the predictions or the fake groundtruth
        #         cap_id = 0
        #         for index, cap in enumerate(caps[id]):
        #             if index == sample_index:  # this is the 'predicted' caption
        #                 prediction = cap
        #             else:
        #                 ground_truth.append(cap)
        #
        #         mets, noun_mets, verb_mets = comp_nouns_verbs(caps_p[id], caps[id], glove_emb=glove_emb)
        #         scores_inner.append(mets)
        #     scores.append(np.nanmean(np.array(scores_inner), axis=0))

        # print(np.mean(np.array(scores), axis=0))
        # print(np.std(np.array(scores), axis=0))

        # ###### RUN MODEL NOUN VERB CHECK
        # load the ids of the captions in our test-on-training run
        # gtt = json.load(open('/media/hayden/Storage2/CODEBASE/SAAT-master/experiments/exp/train_best.json'))
        pr = json.load(open(os.path.join('../experiments', 'lstm00_1', dset + '_' + split + '.json')))
        caps_p = dict()
        for c in pr:
            caps_p[c['image_id']] = c['caption']

        scores_inner = list()
        noun_mets = dict()
        verb_mets = dict()
        for id in caps_p.keys():
            mets, noun_mets, verb_mets = comp_nouns_verbs(caps_p[id], caps[id], noun_mets, verb_mets, glove_emb=glove_emb)
            scores_inner.append(mets)
        print(np.nanmean(np.array(scores_inner), axis=0))
        print(np.nanstd(np.array(scores_inner), axis=0))

        for k in noun_mets.keys():
            # [tp, fp, fn] + [p, tp/p (recall), pp, tp/pp (precision), f1]
            # total gt positives
            noun_mets[k].append(noun_mets[k][0] + noun_mets[k][2])
            # % tp / p (recall)
            if noun_mets[k][-1] > 0:
                noun_mets[k].append(noun_mets[k][0] / float(noun_mets[k][-1]))
            else:
                noun_mets[k].append(0)
            # total pred positives
            noun_mets[k].append(noun_mets[k][0] + noun_mets[k][1])
            # % tp / pp (precision)
            if noun_mets[k][-1] > 0:
                noun_mets[k].append(noun_mets[k][0] / float(noun_mets[k][-1]))
            else:
                noun_mets[k].append(0)
            # f1 score
            if noun_mets[k][4] + noun_mets[k][6] > 0:
                noun_mets[k].append(2*((noun_mets[k][4] * noun_mets[k][6])/float(noun_mets[k][4] + noun_mets[k][6])))
            else:
                noun_mets[k].append(0)

        for k in verb_mets.keys():
            # [tp, fp, fn] + [p, tp/p (recall), pp, tp/pp (precision), f1]
            # total gt positives
            verb_mets[k].append(verb_mets[k][0] + verb_mets[k][2])
            # % tp / p (recall)
            if verb_mets[k][-1] > 0:
                verb_mets[k].append(verb_mets[k][0] / float(verb_mets[k][-1]))
            else:
                verb_mets[k].append(0)
            # total pred positives
            verb_mets[k].append(verb_mets[k][0] + verb_mets[k][1])
            # % tp / pp (precision)
            if verb_mets[k][-1] > 0:
                verb_mets[k].append(verb_mets[k][0] / float(verb_mets[k][-1]))
            else:
                verb_mets[k].append(0)
            # f1 score
            if verb_mets[k][4] + verb_mets[k][6] > 0:
                verb_mets[k].append(2*((verb_mets[k][4] * verb_mets[k][6])/float(verb_mets[k][4] + verb_mets[k][6])))
            else:
                verb_mets[k].append(0)

        if dset == 'msvd':
            thresh = 5
        else:
            thresh = 20
        print('\n\nNouns')
        for w in sorted(noun_mets.items(), key=lambda item: item[-1]):
            if w[0] in vocab_nouns and (w[1][0]+w[1][1] >= thresh or w[1][0]+w[1][2] >= thresh):
                # print('%s\t%d\t%d\t%d\t%d\t%.03f\t%.03f\t%.03f\t%.03f' % (w[0], w[1][0], w[1][1], w[1][2], w[1][3], w[1][4], w[1][5], w[1][6], w[1][7]))
                print('%s\t%d\t%d' % (w[0], w[1][0], w[1][1]))
                print('%s\t\t\t%d\t%d' % (w[0], w[1][0], w[1][2]))
                print()

        print('\n\nVerbs')
        for w in sorted(verb_mets.items(), key=lambda item: item[-1]):
            if w[0] in vocab_verbs and (w[1][0]+w[1][1] >= thresh or w[1][0]+w[1][2] >= thresh):
                # print('%s\t%d\t%d\t%d\t%d\t%.03f\t%.03f\t%.03f\t%.03f' % (w[0], w[1][0], w[1][1], w[1][2], w[1][3], w[1][4], w[1][5], w[1][6], w[1][7]))
                print('%s\t%d\t%d' % (w[0], w[1][0], w[1][1]))
                print('%s\t\t\t%d\t%d' % (w[0], w[1][0], w[1][2]))
                print()


        print()