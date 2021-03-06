import torch
import os
import json

import logging
from datetime import datetime

from dataloader import DataLoader
from model import GeneralModel, GeneralModelDecoupled, CrossEntropyCriterion
from train import test

import opts

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    opt = opts.parse_opts()

    if opt.dataset == 'msvd':
        opt.test_feat_h5 = [os.path.join('datasets', opt.dataset, 'features', opt.dataset+'_test_resnet_mp1.h5'),
                             os.path.join('datasets', opt.dataset, 'features', opt.dataset+'_test_c3d_mp1.h5')]

        opt.test_seq_per_img = 17
    elif opt.dataset == 'msrvtt':
        opt.test_feat_h5 = [os.path.join('datasets', opt.dataset, 'features', opt.dataset+'_test_irv2_mp1.h5'),
                             os.path.join('datasets', opt.dataset, 'features', opt.dataset+'_test_c3d_mp1.h5'),
                             os.path.join('datasets', opt.dataset, 'features', opt.dataset+'_test_category_mp1.h5')]
        opt.test_seq_per_img = 20
    else:
        raise NotImplementedError

    opt.bfeat_h5 = [os.path.join('datasets', opt.dataset, 'features', opt.dataset+'_roi_feat.h5'),
                    os.path.join('datasets', opt.dataset, 'features', opt.dataset+'_roi_box.h5')]

    opt.fr_size_h5 = os.path.join('datasets', opt.dataset, 'features', opt.dataset+'_fr_size.h5')

    opt.test_label_h5 = os.path.join('datasets', opt.dataset, 'metadata', opt.dataset+'_test_'+opt.concepts_h5+'.h5')
    opt.test_cocofmt_file = os.path.join('datasets', opt.dataset, 'metadata', opt.dataset+'_test_cocofmt.json')

    log_path = os.path.join(opt.results_dir, opt.model_id, opt.dataset + '_eval.log')
    opt.model_file = os.path.join(opt.results_dir, opt.model_id, opt.dataset + '.pth')
    opt.result_file = os.path.join(opt.results_dir, opt.model_id, opt.dataset + '.json')

    logging.basicConfig(filename=log_path,
                        filemode='a', level=getattr(logging, opt.loglevel.upper()),
                        format='%(asctime)s:%(levelname)s: %(message)s')

    logger.info(
        'Input arguments: %s',
        json.dumps(
            vars(opt),
            sort_keys=True,
            indent=4))

    start = datetime.now()

    test_opt = {'label_h5': opt.test_label_h5,
                'batch_size': opt.test_batch_size,
                'feat_h5': opt.test_feat_h5,
                'bfeat_h5': opt.bfeat_h5,
                'fr_size_h5': opt.fr_size_h5,
                'cocofmt_file': opt.test_cocofmt_file,
                'seq_per_img': opt.test_seq_per_img,
                'num_chunks': opt.num_chunks,
                'mode': 'test'
                }

    test_loader = DataLoader(test_opt)

    logger.info('Loading model: %s', opt.model_file)
    checkpoint = torch.load(opt.model_file)
    checkpoint_opt = checkpoint['opt']

    opt.model_type = checkpoint_opt.model_type
    opt.vocab = checkpoint_opt.vocab
    opt.vocab_size = checkpoint_opt.vocab_size
    opt.seq_length = checkpoint_opt.seq_length
    opt.svo_length = checkpoint_opt.svo_length
    opt.feat_dims = checkpoint_opt.feat_dims
    opt.bfeat_dims = checkpoint_opt.bfeat_dims

    assert opt.vocab_size == test_loader.get_vocab_size()
    assert opt.seq_length == test_loader.get_seq_length()
    assert opt.svo_length == test_loader.get_svo_length()
    assert opt.feat_dims == test_loader.get_feat_dims()
    assert opt.bfeat_dims == test_loader.get_bfeat_dims()

    logger.info('Building model...')
    if opt.decouple:
        model = GeneralModelDecoupled(opt)
    else:
        model = GeneralModel(opt)

    logger.info('Loading state from the checkpoint...')
    model.load_state_dict(checkpoint['model'])

    xe_criterion = CrossEntropyCriterion()

    if torch.cuda.is_available():
        model.cuda()
        xe_criterion.cuda()

    logger.info('Start testing...')
    test(model, xe_criterion, test_loader, opt)
    logger.info('Time: %s', datetime.now() - start)

    if opt.grounder_type in ['niuc', 'nioc', 'iuc', 'ioc']:
        opt.result_file = os.path.join(opt.results_dir, opt.model_id, opt.dataset + '_gtconcepts.json')
        model.gt_concepts_while_testing = 1
        logger.info('Start testing with gt concepts for upper bound...')
        start = datetime.now()

        test(model, xe_criterion, test_loader, opt)
        logger.info('Testing time: %s', datetime.now() - start)