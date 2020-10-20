
import argparse
import os
import csv
import json
import spacy
import sys
import nltk
from nltk import WordNetLemmatizer
from nltk.stem import PorterStemmer
import numpy as np
import glob
import pickle
from autocorrect import Speller

sys.path.append('coco-caption')
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

# todo add more stats including missed words, nouns, verbs, also svo stats
# to get 'en_core_web_sm' run:
# python -m spacy download en_core_web_sm

# wordlemmatizer = WordNetLemmatizer()
# nlp = spacy.load('en_core_web_sm')
# ps = PorterStemmer()
# sp = Speller()

#
# if os.path.exists(os.path.join('glove6b', 'glove.6B.300d.pkl')):
# 	glove = pickle.load(open(os.path.join('glove6b', 'glove.6B.300d.pkl'), 'rb'))
# else:
# 	glove = dict()
# 	with open(os.path.join('glove6b', 'glove.6B.300d.txt'), 'r') as f:
# 		lines = f.readlines()
# 		for line in lines:
# 			word_split = line.rstrip().split(' ')
# 			word = word_split[0]
#
# 			d = word_split[1:]
# 			d = [float(e) for e in d]
# 			glove[word] = d
# 	pickle.dump(glove, open(os.path.join('glove6b', 'glove.6B.300d.pkl'), 'wb'))


def calc_scores(scorers, gt, pr):
	"""
	Calculate Scores for the metrics in scorers, with groundtruth (gt) and predictions (pr)
	"""
	scores_dict = dict()
	for scorer, method in scorers:
		# print('computing %s score...' % (scorer.method()))
		score, scores = scorer.compute_score(gt, pr)
		if type(method) == list:
			for sc, scs, m in zip(score, scores, method):
				# print("%s: %0.3f" % (m, sc))
				scores_dict[m] = sc
		else:
			# print("%s: %0.3f" % (method, score))
			scores_dict[method] = score

	return scores_dict


def scorer_names_list(scorers):
	"""
	Get list of names of the scorers - basically extracts the BLEUs into the rest of the list
	"""
	scorer_names = list()
	for s in scorers:
		if type(s[1]) == list:
			scorer_names += s[1]
		else:
			scorer_names += [s[1]]
	return scorer_names


def gt_v_gt(gt_json, out_dir, scorers):
	"""
	Run human evaluation, comparing the groundtruth captions with one another

	Will generate:
		- gt_summary.csv : per sample average metric results
		- gt_detatiled.csv : more detailed summary with errors per caption per sample, plus worst, best and avg scores
		- gt_avg_scores.npy : a numpy array version of the gt_summary.csv
		- gt_scores.json : a json version of the gt_detatiled.csv
	"""

	os.makedirs(out_dir, exist_ok=True)

	scorer_names = scorer_names_list(scorers)

	gt_sum_csv_file = open(os.path.join(out_dir, 'gt_summary.csv'), mode='w')
	gt_sum_csv_writer = csv.writer(gt_sum_csv_file, delimiter=',')
	gt_sum_csv_writer.writerow(['VID'] + scorer_names + [''] + scorer_names + [''] + scorer_names)
	gt_sum_csv_writer.writerow([])

	gt_csv_file = open(os.path.join(out_dir, 'gt_detailed.csv'), mode='w')
	gt_csv_writer = csv.writer(gt_csv_file, delimiter=',')
	gt_csv_writer.writerow(['VID', 'CAP #', 'CAP'] + scorer_names)

	all_scores = list()
	all_scores_dict = dict()
	for cnt, gt_item in enumerate(gt_json):

		gt_csv_writer.writerow([])
		gt_csv_writer.writerow([])

		vid = gt_item['video_id']
		gt_caps = gt_item['captions']

		all_scores_dict[vid] = list()

		# human evaluation best, worst, avg gt v gt
		human_scores = dict()
		for i, gt_cap in enumerate(gt_caps):
			gt = {vid: [gt_caps[j] for j in range(len(gt_caps)) if j != i]}
			pr = {vid: [gt_cap]}

			scores_dict = calc_scores(scorers, gt, pr)
			for k, v in scores_dict.items():
				if k not in human_scores:
					human_scores[k] = list()
				human_scores[k].append(v)

			scores_list = list()
			for scorer_name in scorer_names:
				scores_list.append(scores_dict[scorer_name])
			if i == 0:
				gt_csv_writer.writerow([vid, i+1, gt_cap] + scores_list)
			else:
				gt_csv_writer.writerow(['', i+1, gt_cap] + scores_list)

			all_scores_dict[vid].append({'id': i, 'caption': gt_cap, 'scores': scores_dict})

		avg_human_scores = dict()
		best_human_scores = dict()
		worst_human_scores = dict()
		for k, v in human_scores.items():
			avg_human_scores[k] = sum(v)/len(v)
			best_human_scores[k] = max(v)
			worst_human_scores[k] = min(v)

		gt_csv_writer.writerow([])

		summary = [vid]
		scores_list = list()
		for scorer_name in scorer_names:
			scores_list.append(avg_human_scores[scorer_name])
		gt_csv_writer.writerow(['', '', 'AVERAGE'] + scores_list)
		summary += scores_list + ['']
		all_scores.append(scores_list)

		scores_list = list()
		for scorer_name in scorer_names:
			scores_list.append(best_human_scores[scorer_name])
		gt_csv_writer.writerow(['', '', 'BEST'] + scores_list)
		summary += scores_list + ['']

		scores_list = list()
		for scorer_name in scorer_names:
			scores_list.append(worst_human_scores[scorer_name])
		gt_csv_writer.writerow(['', '', 'WORST'] + scores_list)
		summary += scores_list

		gt_sum_csv_writer.writerow(summary)

		print("------------------------------- %d / %d -------------------------------" % (cnt+1, len(gt_json)))

	np.save(os.path.join(out_dir, 'gt_avg_scores.npy'), np.array(all_scores))
	gt_sum_csv_writer.writerow(['AVERAGE'] + list(np.mean(np.array(all_scores), axis=0)))
	gt_csv_file.close()
	gt_sum_csv_file.close()

	with open(os.path.join(out_dir, 'gt_scores.json'), 'w') as f:
		json.dump(all_scores_dict, f)


def pr_v_gt(gt_json, pr_json, out_dir, scorers):
	"""
	Run model evaluation, comparing the model prediction with the groundtruth captions

	Will generate:
		- pr_summary.csv : per sample average metric results
		- pr_detatiled.csv : more detailed summary with errors per caption per sample
		- pr_avg_scores.npy : a numpy array version of the pr_summary.csv
		- pr_scores.json : a json version of the pr_detatiled.csv
	"""

	os.makedirs(out_dir, exist_ok=True)

	scorer_names = scorer_names_list(scorers)

	pr_sum_csv_file = open(os.path.join(out_dir, 'pr_summary.csv'), mode='w')
	pr_sum_csv_writer = csv.writer(pr_sum_csv_file, delimiter=',')
	pr_sum_csv_writer.writerow(['VID'] + scorer_names)

	pr_csv_file = open(os.path.join(out_dir, 'pr_detailed.csv'), mode='w')
	pr_csv_writer = csv.writer(pr_csv_file, delimiter=',')
	pr_csv_writer.writerow(['VID', 'PR CAP', 'GT CAP'] + scorer_names)

	all_scores = list()
	all_scores_dict = dict()
	for cnt, (gt_item, pr_item) in enumerate(zip(gt_json, pr_json['predictions'])):
		assert gt_item['video_id'] == pr_item['image_id']
		pr_csv_writer.writerow([])

		vid = gt_item['video_id']
		gt_caps = gt_item['captions']
		pred_caps = [pr_item['caption']]
		all_scores_dict[vid] = {'predictions': pred_caps, 'groundtruths': list()}

		pred_scores = dict()
		for i, gt_cap in enumerate(gt_caps):
			gt = {vid: [gt_cap]}
			pr = {vid: pred_caps}

			scores_dict = calc_scores(scorers, gt, pr)
			for k, v in scores_dict.items():
				if k not in pred_scores:
					pred_scores[k] = list()
				pred_scores[k].append(v)

			scores_list = list()
			for scorer_name in scorer_names:
				scores_list.append(scores_dict[scorer_name])
			if i == 0:
				pr_csv_writer.writerow([vid, pred_caps[0], gt_cap] + scores_list)
			else:
				pr_csv_writer.writerow(['', '', gt_cap] + scores_list)
			all_scores_dict[vid]['groundtruths'].append({'id': i, 'caption': gt_cap, 'scores': scores_dict})

		avg_pred_scores = dict()
		for k, v in pred_scores.items():
			avg_pred_scores[k] = sum(v)/len(v)

		scores_list = list()
		for scorer_name in scorer_names:
			scores_list.append(avg_pred_scores[scorer_name])
		pr_csv_writer.writerow(['', '', 'AVERAGE A'] + scores_list)

		gt = {vid: gt_caps}
		pr = {vid: pred_caps}
		avg_pred_scores_b = calc_scores(scorers, gt, pr)
		scores_list = list()
		for scorer_name in scorer_names:
			scores_list.append(avg_pred_scores_b[scorer_name])
		pr_csv_writer.writerow(['', '', 'AVERAGE B'] + scores_list)
		pr_sum_csv_writer.writerow([vid] + scores_list)
		all_scores_dict[vid]['scores'] = avg_pred_scores_b
		all_scores.append(scores_list)

		print("------------------------------- %d / %d -------------------------------" % (cnt+1, len(gt_json)))

	np.save(os.path.join(out_dir, 'pr_avg_scores.npy'), np.array(all_scores))
	pr_sum_csv_writer.writerow(['AVERAGE'] + list(np.mean(np.array(all_scores), axis=0)))
	pr_csv_file.close()
	pr_sum_csv_file.close()

	with open(os.path.join(out_dir, 'pr_scores.json'), 'w') as f:
		json.dump(all_scores_dict, f)


def main():
	# Create the parser
	arg_parser = argparse.ArgumentParser(description='Gether indepth information of datasets and predictions')

	# Add the arguments
	arg_parser.add_argument('--dataset', type=str, help='the dataset to run on', default='msrvtt')
	arg_parser.add_argument('--out_dir', type=str, help='the output directory', default=os.path.join('results', 'indepth'))
	arg_parser.add_argument('--split', type=str, help='the split', default='test')

	# Execute the parse_args() method
	args = arg_parser.parse_args()

	# Setup paths and vars
	dataset = args.dataset
	split = args.split
	pr_json_path = os.path.join('results', 'irv2c3dcategory_msrvtt_concat_CIDEr_32_0.0001_20_test.json')
	gt_json_path = os.path.join('datasets', dataset, 'metadata', dataset+'_'+split+'_proprocessedtokens.json')

	# Load the JSON files
	gt_json = json.load(open(gt_json_path, 'r'))
	pr_json = json.load(open(pr_json_path, 'r'))

	# Specify the Scorers
	scorers = [
		(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
		(Meteor(), "METEOR"),
		(Rouge(), "ROUGE_L"),
		(Cider(df=os.path.join('datasets', dataset, 'metadata', dataset+'_train_ciderdf_words.pkl')), "CIDEr"),
		(Spice(), "SPICE")
	]

	# Run Model Prediction Evaluations
	pr_v_gt(gt_json, pr_json, args.out_dir, scorers)

	# Run Human Evaluations
	gt_v_gt(gt_json, args.out_dir, scorers)


if __name__ == "__main__":
	main()
