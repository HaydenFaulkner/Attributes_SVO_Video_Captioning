import os
import json
import spacy
import nltk
from nltk import WordNetLemmatizer
from nltk.stem import PorterStemmer
import numpy as np
import glob
import pickle
from autocorrect import Speller

# to get 'en_core_web_sm' run:
# python -m spacy download en_core_web_sm

wordlemmatizer = WordNetLemmatizer()
nlp = spacy.load('en_core_web_sm')
ps = PorterStemmer()
sp = Speller()

### load gt-related files 
file_path = './results'
num = 2990  # samples in total
gt_file_path = os.path.join('datasets', 'msrvtt', 'metadata')
gt_test = 'msrvtt_test_proprocessedtokens.json'
gt_json = json.load(open(os.path.join(gt_file_path, gt_test), 'r'))

if os.path.exists(os.path.join('glove6b', 'glove.6B.300d.pkl')):
	glove = pickle.load(open(os.path.join('glove6b', 'glove.6B.300d.pkl'), 'rb'))
else:
	glove = dict()
	with open(os.path.join('glove6b', 'glove.6B.300d.txt'), 'r') as f:
		lines = f.readlines()
		for line in lines:
			word_split = line.rstrip().split(' ')
			word = word_split[0]

			d = word_split[1:]
			d = [float(e) for e in d]
			glove[word] = d
	pickle.dump(glove, open(os.path.join('glove6b', 'glove.6B.300d.pkl'), 'wb'))

# category information of the dataset
vid_cat = json.load(open('results/test_vid_cat.json', 'r'))

### files to test
# json_files = glob.glob(os.path.join(file_path, '*.json'))
json_files = ['SAAT-svo.json']  # list of regular .json
json_files = ['irv2c3dcategory_msrvtt_concat_CIDEr_32_0.0001_20_test.json']  # list of regular .json
json_files_svo = ['SAAT-svo.json']  # list of svo-guided .json
json_files_svo = ['irv2c3dcategory_msrvtt_concat_CIDEr_32_0.0001_20_test.json']  # list of svo-guided .json

### format of the statistic result
def res_init():
	res = {}
	res['acc_dec'] = 0  # verb accuracy in description
	res['acc_verb'] = 0  # verb accuracy in svo
	res['acc_dec_vec'] = np.zeros(20)  # corresponding accuracy in each category
	res['acc_verb_vec'] = np.zeros(20)  # likewise
	res['dist_dec'] = np.zeros(num)  # verb distance in description
	res['dist_verb'] = np.zeros(num)  # verb distance in svo
	return res

res_dicts = []
vid_cat_stat = np.zeros(20)

for json_file in json_files:
	res_ov = res_init()
	ov_json = json.load(open(os.path.join(file_path, json_file), 'r'))

	for cnt, (gt_item, ov_item) in enumerate(zip(gt_json, ov_json['predictions'])):
		assert gt_item['video_id'] == ov_item['image_id']
		svos = gt_item['svos']
		svos = [x.split(' ') for x in svos]
		gt_vs = [wordlemmatizer.lemmatize(x[1], 'v') for x in svos]

		ov_tags = nlp(ov_item['caption'])
		ov_v = [wordlemmatizer.lemmatize(tag.text, 'v') for tag in ov_tags if tag.dep_=='ROOT']
		ov_v = ov_v[0] if len(ov_v)>0 else 'empty'

		### accuracy
		if ov_v in gt_vs:
			res_ov['acc_dec'] += 1
			res_ov['acc_dec_vec'][vid_cat[str(ov_item['image_id'])]] += 1
		vid_cat_stat[vid_cat[str(ov_item['image_id'])]] += 1

		### distance
		gt_vs_embs = np.zeros((len(gt_vs), 300))
		for j, gt_v in enumerate(gt_vs):
			gt_vs_embs[j] = glove.get(sp(gt_v), gt_vs_embs[j])
		ov_v_emb = glove.get(sp(ov_v), np.zeros(300))
		res_ov['dist_dec'][cnt] = (np.sqrt(np.sum((gt_vs_embs-ov_v_emb)*(gt_vs_embs-ov_v_emb), 1)).min())/300.

		### accuracy and distance for svo
		if json_file in json_files_svo:
			ov_svo = ov_item['svo'].split()
			ov_svo_v = ov_svo[1] if len(ov_svo) > 1 else 'empty'

			if ps.stem(ov_svo_v) in gt_vs:
				res_ov['acc_verb'] += 1
				res_ov['acc_verb_vec'][vid_cat[str(ov_item['image_id'])]] += 1

			ov_svo_v_emb = glove.get(sp(ov_svo_v), np.zeros(300))
			res_ov['dist_verb'][cnt] = (np.sqrt(np.sum((gt_vs_embs-ov_svo_v_emb)*(gt_vs_embs-ov_svo_v_emb), 1).min()))/300.

	res_dicts.append(res_ov)
print()