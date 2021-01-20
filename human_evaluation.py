import json
import os
from random import randrange
import statistics

import utils


if __name__ == '__main__':

    # setup paths
    cocofmt_file = os.path.join('datasets', 'msvd', 'metadata', 'msvd_test_cocofmt.json')
    # cocofmt_file = os.path.join('datasets', 'msrvtt', 'metadata', 'msrvtt_test_cocofmt.json')
    tmp_file_gt = 'human_gt.json'
    tmp_file_pr = 'human_pr.json'

    # how many samples to run
    runs = 100  # 100 for MSVD
    # runs = 20  # 20 for MSRVTT

    # load the real ground truth
    gt = json.load(open(cocofmt_file))
    ids = [x['id'] for x in gt['images']]
    caps = dict()
    for c in gt['annotations']:
        if c['image_id'] not in caps:
            caps[c['image_id']] = []
        caps[c['image_id']].append(c['caption'])

    # lets do random sampling numerous times and avg the results
    scores = dict()
    for run in range(runs):
        print('run ', run)

        # initialise the predictions list and the fake 'groundtruth' where the predictions are left out
        predictions = list()
        gte = {'images': gt['images'], 'annotations': list(), 'type': 'captions', 'info': dict(), 'licenses': 'n/a'}

        # run through each clip
        for id in ids:
            # randomly select one of the ground truth captions for this clip
            sample_index = randrange(len(caps[id]))

            # append the caption to either the predictions or the fake groundtruth
            cap_id = 0
            for index, cap in enumerate(caps[id]):
                if index == sample_index:  # this is the 'predicted' caption
                    predictions.append({'image_id': id, 'caption': cap})
                else:  # this remains a groundtruth caption
                    gte['annotations'].append({'caption': cap, 'image_id': id, 'id': cap_id})
                    cap_id += 1

        # dump out the new groundtruth and prediction json files
        json.dump(gte, open(tmp_file_gt, 'w'))
        json.dump(predictions, open(tmp_file_pr, 'w'))

        # calculate the language stats
        lang_stats = utils.language_eval(tmp_file_gt, tmp_file_pr)
        for k, v in lang_stats.items():
            if k not in scores:
                scores[k] = list()
            scores[k].append(v)

    print('------------ scores after %d runs ------------' % runs)
    print(scores)
    for k, v in scores.items():
        print(k, statistics.mean(v), statistics.stdev(v))
