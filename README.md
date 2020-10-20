# Syntax-Aware Action Targeting for Video Captioning

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

## Test

```bash
make -f SpecifiedMakefile test [options]
```
Please refer to the Makefile (and opts_svo.py file) for the set of available train/test options. For example, to reproduce the reported result
```bash
make -f Makefile_msrvtt_svo test GID=0 EXP_NAME=xe FEATS="irv2 c3d category" BFEATS="roi_feat roi_box" USE_RL=0 CST=0 USE_MIXER=0 SCB_CAPTIONS=0 LOGLEVEL=DEBUG LAMBDA=20
```

## Train

To train the model using XE loss
```bash
make -f Makefile_msrvtt_svo train GID=0 EXP_NAME=xe FEATS="irv2 c3d category" BFEATS="roi_feat roi_box" USE_RL=0 CST=0 USE_MIXER=0 SCB_CAPTIONS=0 LOGLEVEL=DEBUG MAX_EPOCH=100 LAMBDA=20
```

If you want to change the input features, modify the `FEATS` variable in above commands.

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
