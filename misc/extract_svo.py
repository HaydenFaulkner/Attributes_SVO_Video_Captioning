"""
https://raw.githubusercontent.com/acrosson/nlp/master/subject_extraction/subject_extraction.py

"""
import h5py
import json
import os
import re
import pickle
import nltk
import numpy as np

nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('brown')
nltk.download('conll2000')
nltk.download('treebank')
from nltk.corpus import stopwords

stop = stopwords.words('english')

from nltk import DefaultTagger, UnigramTagger, BigramTagger, TrigramTagger


class SubjectTrigramTagger(object):

    """ Creates an instance of NLTKs TrigramTagger with a backoff
    tagger of a bigram tagger a unigram tagger and a default tagger that sets
    all words to nouns (NN)
    """

    def __init__(self, train_sents):

        """
        train_sents: trained sentences which have already been tagged.
                Currently using Brown, conll2000, and TreeBank corpuses
        """

        t0 = DefaultTagger('NN')
        t1 = UnigramTagger(train_sents, backoff=t0)
        t2 = BigramTagger(train_sents, backoff=t1)
        self.tagger = TrigramTagger(train_sents, backoff=t2)

    def tag(self, tokens):
        return self.tagger.tag(tokens)

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


def tokenize_sentences(document):
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    return sentences


def get_entities(document):
    """Returns Named Entities using NLTK Chunking"""
    entities = []
    sentences = tokenize_sentences(document)

    # Part of Speech Tagging
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    for tagged_sentence in sentences:
        for chunk in nltk.ne_chunk(tagged_sentence):
            if type(chunk) == nltk.tree.Tree:
                entities.append(' '.join([c[0] for c in chunk]).lower())
    return entities


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

    # todo entities returns empty list as no named entities found so lets just use the most freq noun?
    # # Get Top 10 entities
    # entities = get_entities(document)
    # top_10_entities = [w for w, c in nltk.FreqDist(entities).most_common(10)]
    #
    # # Get the subject noun by looking at the intersection of top 10 entities
    # # and most frequent nouns. It takes the first element in the list
    # subject_nouns = [entity for entity in top_10_entities if entity.split()[0] in most_freq_nouns]
    subject_nouns = most_freq_nouns
    if len(subject_nouns) > 0:
        return subject_nouns[0]
    else:
        return subject_nouns


def trained_tagger():
    """Returns a trained trigram tagger

    existing : set to True if already trained tagger has been pickled
    """
    if os.path.exists('trained_tagger.pkl'):
        trigram_tagger = pickle.load(open('trained_tagger.pkl', 'rb'))
        return trigram_tagger

    # Aggregate trained sentences for N-Gram Taggers
    train_sents = nltk.corpus.brown.tagged_sents()
    train_sents += nltk.corpus.conll2000.tagged_sents()
    train_sents += nltk.corpus.treebank.tagged_sents()

    # Create instance of SubjectTrigramTagger and persist instance of it
    trigram_tagger = SubjectTrigramTagger(train_sents)
    pickle.dump(trigram_tagger, open('trained_tagger.pkl', 'wb'))

    return trigram_tagger


def tag_sentences(subject, document, trigram_tagger):
    """Returns tagged sentences using POS tagging"""

    # Tokenize Sentences and words
    sentences = tokenize_sentences(document)
    merge_multi_word_subject(sentences, subject)

    # Filter out sentences where subject is not present
    sentences = [sentence for sentence in sentences if subject in
                [word.lower() for word in sentence]]

    # Tag each sentence
    tagged_sents = [trigram_tagger.tag(sent) for sent in sentences]
    return tagged_sents


def merge_multi_word_subject(sentences, subject):
    """Merges multi word subjects into one single token
    ex. [('steve', 'NN', ('jobs', 'NN')] -> [('steve jobs', 'NN')]
    """
    if len(subject.split()) == 1:
        return sentences
    subject_lst = subject.split()
    sentences_lower = [[word.lower() for word in sentence]
                        for sentence in sentences]
    for i, sent in enumerate(sentences_lower):
        if subject_lst[0] in sent:
            for j, token in enumerate(sent):
                start = subject_lst[0] == token
                exists = subject_lst == sent[j:j+len(subject_lst)]
                if start and exists:
                    del sentences[i][j+1:j+len(subject_lst)]
                    sentences[i][j] = subject
    return sentences


def get_svo(sentence, subject):
    """Returns a dictionary containing:

    subject : the subject determined earlier
    action : the action verb of particular related to the subject
    object : the object the action is referring to
    phrase : list of token, tag pairs for that lie within the indexes of
                the variables above
    """
    subject_idx = next((i for i, v in enumerate(sentence)
                    if v[0].lower() == subject), None)
    data = {'subject': subject}
    for i in range(subject_idx, len(sentence)):
        found_action = False
        for j, (token, tag) in enumerate(sentence[i+1:]):
            if tag in VERBS:
                data['action'] = token
                found_action = True
            if tag in NOUNS and found_action == True:
                data['object'] = token
                data['phrase'] = sentence[i: i+j+2]
                return data
    return {}


def make_h5(json_file, length=30, vocab=None):
    json_data = json.load(open(json_file, 'r'))
    if 'proprocessedtokens' in json_file:
        h5_file = json_file.replace('proprocessedtokens', 'sequencelabel').replace('.json', '.h5')
        h5_infile = h5_file.split('sequencelabel')[0] + 'sequencelabel.h5'
    else:
        h5_file = json_file.replace('ppt', 'sl').replace('.json', '.h5')
        h5_infile = h5_file.split('sl')[0] + 'sequencelabel.h5'


    with h5py.File(h5_infile, "r") as f:
        videos = list(f['videos'])
        if vocab is None:
            vocab = [v.decode("utf-8") for v in list(f['vocab'])]

        index = 0
        vid_index = 0
        start_ix_svo = list()
        end_ix_svo = list()
        length_svo = list()
        video_svo = list()
        svos_vocab_inds = list()
        for video in videos:
            svos = []
            for json_entry in json_data:
                if json_entry['video_id'] == int(video):
                    svos = json_entry['svos']
                    break

            if len(svos) == 0:
                print()
            start_ix_svo.append(index)
            for svo in svos:
                svo_vocab_ind = list()
                svo_vocab_ind = np.zeros((length,), dtype=np.int64)
                for i, word in enumerate(svo.split()):
                    if word in vocab:
                        svo_vocab_ind[i] = vocab.index(word)
                        # svo_vocab_ind.append(vocab.index(word))
                    else:
                        svo_vocab_ind[i] = vocab.index('<unk>')
                        # svo_vocab_ind.append(vocab.index('<unk>'))
                svos_vocab_inds.append(svo_vocab_ind)
                length_svo.append(i+2)
                video_svo.append(vid_index)
                index += 1

            end_ix_svo.append(index)
            vid_index += 1

        with h5py.File(h5_file, "w") as fo:
            for k in list(f.keys()):
                if k == 'label_start_ix_svo':
                    fo.create_dataset(k, data=np.array(start_ix_svo))
                elif k == 'label_end_ix_svo':
                    fo.create_dataset(k, data=np.array(end_ix_svo))
                elif k == 'label_length_svo':
                    fo.create_dataset(k, data=np.array(length_svo))
                elif k == 'label_to_video_svo':
                    fo.create_dataset(k, data=np.array(video_svo))
                elif k == 'labels_svo':
                    fo.create_dataset(k, data=np.array(svos_vocab_inds))
                else:
                    fo.create_dataset(k, data=f[k])


if __name__ == '__main__':

    trigram_tagger = trained_tagger()

    type = 'concepts_per_video'
    for dataset in ['msvd', 'msrvtt']:
        with open(os.path.join('datasets', dataset, 'metadata', dataset+'_nouns_vocab.pkl'), 'rb') as f:
            vocab_nouns = pickle.load(f)
        with open(os.path.join('datasets', dataset, 'metadata', dataset+'_verbs_vocab.pkl'), 'rb') as f:
            vocab_verbs = pickle.load(f)

        for split in ['train', 'val', 'test']:

            gt_file_path = os.path.join('datasets', dataset, 'metadata')
            gt_test = dataset + '_' + split + '_proprocessedtokens.json'
            gt_json = json.load(open(os.path.join(gt_file_path, gt_test), 'r'))

            if type in ['concepts_per_video']:
                gt_test_out = dataset + '_' + split + '_ppt_top_concepts.json'
                for gt_item in gt_json:
                    words = dict()
                    for cap in gt_item['captions']:
                        cap = clean_document(cap)
                        tags = nltk.pos_tag(nltk.word_tokenize(cap))
                        for tag, _ in tags:
                            if tag in vocab_nouns or tag in vocab_verbs:
                                if tag not in words:
                                    words[tag] = 0
                                words[tag] += 1
                    top_words = [x[0] for x in sorted(words.items(), key=lambda item: item[1], reverse=True)[:5]]
                    gt_item['svos'] = [' '.join(top_words)]


            elif type in ['all', 'top']:
                for gt_item in gt_json:
                    svos = set()
                    svos_l = list()
                    for cap in gt_item['captions']:
                        cap = clean_document(cap)
                        subject = extract_subject(cap)

                        if len(subject) > 0:
                            tagged_cap = tag_sentences(subject, cap, trigram_tagger)[0]
                            svo = get_svo(tagged_cap, subject)
                            if len(svo) > 0:
                                svos.add(' '.join([svo['subject'], svo['action'], svo['object']]))
                                svos_l.append(svo)

                    if type in ['all']:
                        gt_item['svos'] = list(svos)
                        gt_test_out = dataset + '_' + split + '_proprocessedtokens_new_svos.json'
                    elif type in ['top']:
                        counts_sub = dict()
                        counts_act = dict()
                        counts_obj = dict()
                        for svo in svos_l:
                            if svo['subject'] in counts_sub:
                                counts_sub[svo['subject']] += 1
                            else:
                                counts_sub[svo['subject']] = 1
                            if svo['action'] in counts_act:
                                counts_act[svo['action']] += 1
                            else:
                                counts_act[svo['action']] = 1
                            if svo['object'] in counts_obj:
                                counts_obj[svo['object']] += 1
                            else:
                                counts_obj[svo['object']] = 1

                        if len(counts_sub.items()) < 1:
                            sub = '<unk>'
                        else:
                            sub = sorted(counts_sub.items(), key=lambda item: item[1])[-1][0]
                        if len(counts_act.items()) < 1:
                            act = '<unk>'
                        else:
                            act = sorted(counts_act.items(), key=lambda item: item[1])[-1][0]
                        if len(counts_obj.items()) < 1:
                            obj = '<unk>'
                        else:
                            obj = sorted(counts_obj.items(), key=lambda item: item[1])[-1][0]
                        gt_item['svos'] = [sub + ' ' + act + ' ' + obj]
                        gt_test_out = dataset + '_' + split + '_proprocessedtokens_top_svos.json'

            elif type in ['all2', 'top2']:
                from openie import StanfordOpenIE
                with StanfordOpenIE() as client:
                    for gt_item in gt_json:
                        svos = set()
                        svosb = set()
                        for cap in gt_item['captions']:
                            cap = clean_document(cap)
                            for triple in client.annotate(cap):
                                svos.add(' '.join([triple['subject'], triple['relation'], triple['object']]))
                                # sub = triple['subject'].split(' ')[0]
                                # tagged_sent = tag_sentences(sub, cap, trigram_tagger)
                                # if len(tagged_sent) > 0:
                                #     svo = get_svo(tagged_sent[0], sub)
                                #     if len(svo) > 0:
                                #         svosb.add(' '.join([svo['subject'], svo['action'], svo['object']]))

                        if len(svos) < 1:
                            svos.add(cap)

                        if type in ['all2']:
                            gt_item['svos'] = list(svos)
                            gt_test_out = dataset + '_' + split + '_proprocessedtokens_new2_svos.json'
                    #     elif type in ['top2']:
                    #         counts = dict()
                    #         counts_sub = dict()
                    #         counts_act = dict()
                    #         counts_obj = dict()
                    #         for svo in svosb:
                    #             if svo['subject'] in counts_sub:
                    #                 counts_sub[svo['subject']] += 1
                    #             else:
                    #                 counts_sub[svo['subject']] = 1
                    #             if svo['action'] in counts_act:
                    #                 counts_act[svo['action']] += 1
                    #             else:
                    #                 counts_act[svo['action']] = 1
                    #             if svo['object'] in counts_obj:
                    #                 counts_obj[svo['object']] += 1
                    #             else:
                    #                 counts_obj[svo['object']] = 1
                    #
                    #             svo_j = ','.join([svo['subject'], svo['action'], svo['object']])
                    #             if svo_j in counts:
                    #                 counts[svo_j] += 1
                    #             else:
                    #                 counts[svo_j] = 1
                    #
                    # print('----------------------------')
            json.dump(gt_json, open(os.path.join(gt_file_path, gt_test_out), 'w'))

            if type in ['concepts_per_video']:
                make_h5(os.path.join(gt_file_path, gt_test_out), length=5)#USE ORIG VOCAB FOR EASE... still only has smaller vocab items, so many more 0 cats, vocab=['<end>', '<start>', '<unk>'] + vocab_nouns + vocab_verbs)
            else:
                make_h5(os.path.join(gt_file_path, gt_test_out))