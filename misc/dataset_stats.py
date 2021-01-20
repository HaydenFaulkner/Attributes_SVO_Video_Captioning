import json
import os
import re

import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')

# Noun Part of Speech Tags used by NLTK
# More can be found here
# http://www.winwaed.com/blog/2011/11/08/part-of-speech-tags/
NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']
VERBS = ['VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ']


def clean_document(document):
    """Remove enronious characters. Extra whitespace and stop words"""
    document = re.sub('[^A-Za-z .-]+', ' ', document)
    document = ' '.join(document.split())
    document = ' '.join([i for i in document.split() if i not in stop])
    return document


def word_freq_dist(document):
    """Returns a word count frequency distribution"""
    words = nltk.tokenize.word_tokenize(document)
    words = [word.lower() for word in words if word not in stop]
    fdist = nltk.FreqDist(words)
    return fdist


def extract_subject(document):
    # Get most frequent Nouns
    fdist = word_freq_dist(document)
    most_freq_nouns = [w for w, c in fdist.most_common(10) if nltk.pos_tag([w])[0][1] in NOUNS]
    

for dataset in ['msvd', 'msrvtt']:
    print('-'*30 + dataset + '-'*30)
    for split in ['train']:
    # for split in ['train', 'val', 'test']:
        print('-' * 30 + split + '-' * 30)

        gt_file_path = os.path.join('datasets', dataset, 'metadata')
        gt_test = dataset + '_' + split + '_proprocessedtokens.json'
        gt_json = json.load(open(os.path.join(gt_file_path, gt_test), 'r'))

        word_count = dict()
        nouns_count = dict()
        verbs_count = dict()
        num_words = []
        num_nouns = []
        num_verbs = []
        num_unique_words = []
        num_unique_nouns = []
        num_unique_verbs = []

        num_words_per_cap = []
        num_nouns_per_cap = []
        num_verbs_per_cap = []
        num_unique_words_per_cap = []
        num_unique_nouns_per_cap = []
        num_unique_verbs_per_cap = []

        for gt_item in gt_json:
            num_words.append(0)
            num_nouns.append(0)
            num_verbs.append(0)
            num_unique_words.append(0)
            num_unique_nouns.append(0)
            num_unique_verbs.append(0)
            vid_concepts = set()

            for cap in gt_item['captions']:
                num_words_per_cap.append(0)
                num_nouns_per_cap.append(0)
                num_verbs_per_cap.append(0)
                num_unique_words_per_cap.append(0)
                num_unique_nouns_per_cap.append(0)
                num_unique_verbs_per_cap.append(0)
                cap_concepts = set()

                for w in cap.split(' '):
                    if w not in word_count:
                        word_count[w] = 0
                    word_count[w] += 1
                    num_words_per_cap[-1] += 1
                    num_words[-1] += 1

                cap = clean_document(cap)  # removes some 179 stop words
                tags = nltk.pos_tag(nltk.word_tokenize(cap))

                for w in tags:

                    if w[1] in NOUNS:
                        if w[0] not in nouns_count:
                            nouns_count[w[0]] = 0
                        nouns_count[w[0]] += 1
                        num_nouns_per_cap[-1] += 1
                        num_nouns[-1] += 1
                    elif w[1] in VERBS:
                        if w[0] not in verbs_count:
                            verbs_count[w[0]] = 0
                        verbs_count[w[0]] += 1
                        num_verbs_per_cap[-1] += 1
                        num_verbs[-1] += 1

                    if w[0] not in cap_concepts:
                        num_unique_words_per_cap[-1] += 1
                        if w[1] in NOUNS:
                            num_unique_nouns_per_cap[-1] += 1
                        elif w[1] in VERBS:
                            num_unique_verbs_per_cap[-1] += 1

                    if w[0] not in vid_concepts:
                        num_unique_words[-1] += 1
                        if w[1] in NOUNS:
                            num_unique_nouns[-1] += 1
                        elif w[1] in VERBS:
                            num_unique_verbs[-1] += 1

                    cap_concepts.add(w[0])
                    vid_concepts.add(w[0])

        print('# words (uncleaned): %d' % sum(num_words))
        print('# nouns: %d (%d%%)' % (sum(num_nouns), int(100*float(sum(num_nouns))/sum(num_words))))
        print('# verbs: %d (%d%%)' % (sum(num_verbs), int(100*float(sum(num_verbs))/sum(num_words))))

        print('# unique words (uncleaned): %d' % len(word_count))
        print('# unique nouns: %d (%d%%)' % (len(nouns_count), int(100*float(len(nouns_count))/len(word_count))))
        print('# unique verbs: %d (%d%%)' % (len(verbs_count), int(100*float(len(verbs_count))/len(word_count))))

        print()
        print('Nouns per Video (min | avg | max): %d | %d | %d' % (min(num_nouns), int(float(sum(num_nouns))/len(num_nouns)), max(num_nouns)))
        print('Verbs per Video (min | avg | max): %d | %d | %d' % (min(num_verbs), int(float(sum(num_verbs))/len(num_verbs)), max(num_verbs)))

        print('Nouns per Caption (min | avg | max): %d | %d | %d' % (min(num_nouns_per_cap), int(float(sum(num_nouns_per_cap)) / len(num_nouns_per_cap)), max(num_nouns_per_cap)))
        print('Verbs per Caption (min | avg | max): %d | %d | %d' % (min(num_verbs_per_cap), int(float(sum(num_verbs_per_cap)) / len(num_verbs_per_cap)), max(num_verbs_per_cap)))

        print()
        print('Unique Nouns per Video (min | avg | max): %d | %d | %d' % (min(num_unique_nouns), int(float(sum(num_unique_nouns)) / len(num_unique_nouns)), max(num_unique_nouns)))
        print('Unique Verbs per Video (min | avg | max): %d | %d | %d' % (min(num_unique_verbs), int(float(sum(num_unique_verbs)) / len(num_unique_verbs)), max(num_unique_verbs)))

        print('Unique Nouns per Caption (min | avg | max): %d | %d | %d' % (min(num_unique_nouns_per_cap), int(float(sum(num_unique_nouns_per_cap)) / len(num_unique_nouns_per_cap)), max(num_unique_nouns_per_cap)))
        print('Unique Verbs per Caption (min | avg | max): %d | %d | %d' % (min(num_unique_verbs_per_cap), int(float(sum(num_unique_verbs_per_cap)) / len(num_unique_verbs_per_cap)), max(num_unique_verbs_per_cap)))

        print()

        print('Most Common Nouns')
        for w, c in sorted(nouns_count.items(), key=lambda item: item[1], reverse=True)[:101]:
            print('%s\t%d' % (w, c))

        print('Most Common Verbs')
        for w, c in sorted(verbs_count.items(), key=lambda item: item[1], reverse=True)[:101]:
            print('%s\t%d' % (w,c))