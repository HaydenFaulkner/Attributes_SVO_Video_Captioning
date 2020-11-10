<h1 align="center">Experiments</h1>
<p align="center"><a href="https://drive.google.com/drive/folders/1XYgBaVkAQaSw6nQL-7dDAMItfqrgx2dE?usp=sharing">Download the files from my Google Drive.</a></p>

<h3><code><a href="default">my_defaults</a></code></h3>
The default as in paper, MSVD is just a lil bit worse than the paper results, unsure why

To train MSVD:
<pre>
python train_svo.py --exp_type default
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

To train MSRVTT:
<pre>
python train_svo.py --exp_type default
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

<h3><code><a href="default_all_svo">default_all_svo</a></code></h3>
The same as the default however instead of just conditioning the LSTM input on the verb and past word, we condition on the sub, verb and obj as well as previous word.

To train MSVD:
<pre>
python train_svo.py --exp_type default
                    --model_file experiments/default_all_svo/msvd.pth
                    --result_file experiments/default_all_svo/msvd.json
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
                    --pass_all_svo 1
</pre>

To train MSRVTT:
<pre>
python train_svo.py --exp_type default
                    --model_file experiments/default_all_svo/msrvtt.pth
                    --result_file experiments/default_all_svo/msrvtt.json
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
                    --pass_all_svo 1
</pre>