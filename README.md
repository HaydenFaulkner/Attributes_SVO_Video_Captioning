# Extending - Syntax-Aware Action Targeting for Video Captioning

Code for SAAT from ["Syntax-Aware Action Targeting for Video Captioning"](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zheng_Syntax-Aware_Action_Targeting_for_Video_Captioning_CVPR_2020_paper.pdf) (Accepted to CVPR 2020). The implementation is based on ["Consensus-based Sequence Training for Video Captioning"](https://github.com/mynlp/cst_captioning).

## Dependencies

* Python 3.6
* PyTorch 1.1
* CUDA 10.0

This repo includes an [edited version (`coco-caption`)](coco-caption) of the [Python 3 coco evaluation protocols](https://github.com/salaniz/pycocoevalcap) (edited to load CIDEr corpus)

## Data
Data can be downloaded from my **Google Drive**:
* [`datasets/msrvtt/features`](https://drive.google.com/drive/folders/1OOaCFWia2imwHCf4gXySLLVRJQCVaCav?usp=sharing)
* [`datasets/msrvtt/metadata`](https://drive.google.com/drive/folders/1oFYKA1bVi0X1djF7XotbZOPWFf_QU9Bw?usp=sharing)
* [`datasets/msvd/features`](https://drive.google.com/drive/folders/1JS3V8fwQySpfJ-Ob1eTs9WJ1b4UwNTwj?usp=sharing)
* [`datasets/msvd/metadata`](https://drive.google.com/drive/folders/1HFrRVlt7Izn7Fzn1T_W8MoI3_Isl35bM?usp=sharing)
* [`experiments`](https://drive.google.com/drive/folders/1XYgBaVkAQaSw6nQL-7dDAMItfqrgx2dE?usp=sharing)

## Experiments
[View my experiments and results](experiments)

## Test
To test on MSVD run (~2mins):
```bash
python test_svo.py --filter_type svo_original
                   --captioner_type lstm
                   --model_file experiments/default/msvd.pth
                   --result_file experiments/default/msvd_test_on_test.json
                   --test_label_h5 datasets/msvd/metadata/msvd_test_sequencelabel.h5
                   --test_cocofmt_file datasets/msvd/metadata/msvd_test_cocofmt.json
                   --test_feat_h5 datasets/msvd/features/msvd_test_resnet_mp1.h5 datasets/msvd/features/msvd_test_c3d_mp1.h5
                   --bfeat_h5 datasets/msvd/features/msvd_roi_feat.h5 datasets/msvd/features/msvd_roi_box.h5
                   --fr_size_h5 datasets/msvd/features/msvd_fr_size.h5
```

To test on MSRTVTT run (~5mins):
```bash
python test_svo.py --filter_type svo_original
                   --captioner_type lstm
                   --model_file experiments/default/msrvtt.pth
                   --result_file experiments/default/msrvtt_test_on_test.json
                   --test_label_h5 datasets/msrvtt/metadata/msrvtt_test_sequencelabel.h5
                   --test_cocofmt_file datasets/msrvtt/metadata/msrvtt_test_cocofmt.json
                   --test_feat_h5 datasets/msrvtt/features/msrvtt_test_irv2_mp1.h5 datasets/msrvtt/features/msrvtt_test_c3d_mp1.h5 datasets/msrvtt/features/msrvtt_test_category_mp1.h5
                   --bfeat_h5 datasets/msrvtt/features/msrvtt_roi_feat.h5 datasets/msrvtt/features/msrvtt_roi_box.h5
                   --fr_size_h5 datasets/msrvtt/features/msrvtt_fr_size.h5
```


## Train
To train on MSVD run (~1hr for 100 epochs batch size 32):
<pre>
python train_svo.py --filter_type svo_original
                    --captioner_type lstm
                    --model_file experiments/default/msvd.pth
                    --result_file experiments/default/msvd.json
                    --train_label_h5 datasets/msvd/metadata/msvd_train_sequencelabel.h5
                    --val_label_h5 datasets/msvd/metadata/msvd_val_sequencelabel.h5
                    --test_label_h5 datasets/msvd/metadata/msvd_test_sequencelabel.h5
                    --train_cocofmt_file datasets/msvd/metadata/msvd_train_cocofmt.json
                    --val_cocofmt_file datasets/msvd/metadata/msvd_val_cocofmt.json
                    --test_cocofmt_file datasets/msvd/metadata/msvd_test_cocofmt.json
                    --train_bcmrscores_pkl datasets/msvd/metadata/msvd_train_evalscores.pkl
                    --train_cached_tokens datasets/msvd/metadata/msvd_train_ciderdf.pkl
                    --train_feat_h5 datasets/msvd/features/msvd_train_resnet_mp1.h5 datasets/msvd/features/msvd_train_c3d_mp1.h5
                    --val_feat_h5 datasets/msvd/features/msvd_val_resnet_mp1.h5 datasets/msvd/features/msvd_val_c3d_mp1.h5
                    --test_feat_h5 datasets/msvd/features/msvd_test_resnet_mp1.h5 datasets/msvd/features/msvd_test_c3d_mp1.h5
                    --bfeat_h5 datasets/msvd/features/msvd_roi_feat.h5 datasets/msvd/features/msvd_roi_box.h5
                    --fr_size_h5 datasets/msvd/features/msvd_fr_size.h5
                    --train_seq_per_img 17
                    --test_seq_per_img 17
                    --batch_size 8
                    --test_batch_size 8
                    --max_epochs 100
                    --labda 12.0
</pre>

To train on MSRTVTT run (~7hrs for 100 epochs batch size 32):
<pre>
python train_svo.py --filter_type svo_original
                    --captioner_type lstm
                    --model_file experiments/default/msrvtt.pth
                    --result_file experiments/default/msrvtt.json
                    --train_label_h5 datasets/msrvtt/metadata/msrvtt_train_sequencelabel.h5
                    --val_label_h5 datasets/msrvtt/metadata/msrvtt_val_sequencelabel.h5
                    --test_label_h5 datasets/msrvtt/metadata/msrvtt_test_sequencelabel.h5
                    --train_cocofmt_file datasets/msrvtt/metadata/msrvtt_train_cocofmt.json
                    --val_cocofmt_file datasets/msrvtt/metadata/msrvtt_val_cocofmt.json
                    --test_cocofmt_file datasets/msrvtt/metadata/msrvtt_test_cocofmt.json
                    --train_bcmrscores_pkl datasets/msrvtt/metadata/msrvtt_train_evalscores.pkl
                    --train_cached_tokens datasets/msrvtt/metadata/msrvtt_train_ciderdf.pkl
                    --train_feat_h5 datasets/msrvtt/features/msrvtt_train_irv2_mp1.h5 datasets/msrvtt/features/msrvtt_train_c3d_mp1.h5 datasets/msrvtt/features/msrvtt_train_category_mp1.h5
                    --val_feat_h5 datasets/msrvtt/features/msrvtt_val_irv2_mp1.h5 datasets/msrvtt/features/msrvtt_val_c3d_mp1.h5 datasets/msrvtt/features/msrvtt_val_category_mp1.h5
                    --test_feat_h5 datasets/msrvtt/features/msrvtt_test_irv2_mp1.h5 datasets/msrvtt/features/msrvtt_test_c3d_mp1.h5 datasets/msrvtt/features/msrvtt_test_category_mp1.h5
                    --bfeat_h5 datasets/msrvtt/features/msrvtt_roi_feat.h5 datasets/msrvtt/features/msrvtt_roi_box.h5
                    --fr_size_h5 datasets/msrvtt/features/msrvtt_fr_size.h5
                    --train_seq_per_img 20
                    --test_seq_per_img 20
                    --batch_size 64
                    --test_batch_size 32
                    --max_epochs 200
                    --labda 20.0
</pre>

### Citation
```
@InProceedings{Zheng_2020_CVPR,
author = {Zheng, Qi and Wang, Chaoyue and Tao, Dacheng},
title = {Syntax-Aware Action Targeting for Video Captioning},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

### Acknowledgements

* PyTorch implementation of [SAAT](https://github.com/SydCaption/SAAT)
* Pytorch implementation of [CST](https://github.com/mynlp/cst_captioning)
* PyTorch implementation of  [SCST](https://github.com/ruotianluo/self-critical.pytorch)
