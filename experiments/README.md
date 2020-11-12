<h1 align="center">Experiments</h1>
<p align="center"><a href="https://drive.google.com/drive/folders/1XYgBaVkAQaSw6nQL-7dDAMItfqrgx2dE?usp=sharing">Download the files from my Google Drive.</a></p>

<h2>Summary</h2>
MSVD Test Set Results:
<table>
    <tr>
        <th>Experiment</th>
        <th>BLEU@1</th>
        <th>BLEU@2</th>
        <th>BLEU@3</th>
        <th>BLEU@4</th>
        <th>METEOR</th>
        <th>ROUGE_L</th>
        <th>CIDEr</th>
        <th>SPICE</th>
    </tr>
    <tr>
        <td><a href="paper_xe">the paper</a></td>
        <td>0.7941</td>
        <td>0.6710</td>
        <td>0.5690</td>
        <td>0.4647</td>
        <td><b>0.3347</b></td>
        <td>0.6938</td>
        <td><b>0.8101</b></td>
        <td>-</td>
    </tr>
    <tr>
        <td><code><a href="default">default</a></code></td>
        <td>0.7881</td>
        <td>0.6657</td>
        <td>0.5661</td>
        <td>0.4629</td>
        <td>0.3309</td>
        <td>0.6924</td>
        <td>0.7879</td>
        <td>0.0511</td>
    </tr>
    <tr>
        <td><code><a href="default_all_svo">default_all_svo</a></code></td>
        <td>0.7808</td>
        <td>0.6564</td>
        <td>0.5572</td>
        <td>0.4541</td>
        <td>0.3277</td>
        <td>0.6900</td>
        <td>0.7942</td>
        <td>0.0500</td>
    </tr>
    <tr>
        <td><code><a href="transformer01">transformer01</a></code></td>
        <td>0.7815</td>
        <td>0.6052</td>
        <td>0.4642</td>
        <td>0.2949</td>
        <td>0.2620</td>
        <td>0.6498</td>
        <td>0.4266</td>
        <td>0.0373</td>
    </tr>
    <tr>
        <td><code><a href="transformer01_cc">transformer01_cc</a></code></td>
        <td>0.7788</td>
        <td>0.6533</td>
        <td>0.5479</td>
        <td>0.4390</td>
        <td>0.3279</td>
        <td>0.6852</td>
        <td>0.7632</td>
        <td><b>0.0513</b></td>
    </tr>
    <tr>
        <td><code><a href="transformer01_all_svo">transformer01_all_svo</a></code></td>
        <td><b>0.7979</b></td>
        <td><b>0.6807</b></td>
        <td><b>0.5816</b></td>
        <td><b>0.4763</b></td>
        <td>0.3323</td>
        <td><b>0.6980</b></td>
        <td>0.7847</td>
        <td>0.0508</td>
    </tr>
    <tr>
        <td><code><a href="transformer01_all_svo_cc">transformer01_all_svo_cc</a></code></td>
        <td>0.7910</td>
        <td>0.6692</td>
        <td>0.5635</td>
        <td>0.4510</td>
        <td>0.3214</td>
        <td>0.6837</td>
        <td>0.7183</td>
        <td>0.0490</td>
    </tr>
</table>

MSRVTT Test Set Results:
<table>
    <tr>
        <th>Experiment</th>
        <th>BLEU@1</th>
        <th>BLEU@2</th>
        <th>BLEU@3</th>
        <th>BLEU@4</th>
        <th>METEOR</th>
        <th>ROUGE_L</th>
        <th>CIDEr</th>
        <th>SPICE</th>
    </tr>
    <tr>
        <td><a href="paper_xe">the paper</a></td>
        <td><b>0.8024</b></td>
        <td><b>0.6619</b></td>
        <td><b>0.5257</b></td>
        <td><b>0.4052</b></td>
        <td><b>0.2819</b></td>
        <td>0.6091</td>
        <td><b>0.4907</b></td>
        <td>-</td>
    </tr>
    <tr>
        <td><code><a href="default">default</a></code></td>
        <td>0.7953</td>
        <td>0.6522</td>
        <td>0.5157</td>
        <td>0.3965</td>
        <td>0.2806</td>
        <td>0.6043</td>
        <td>0.4867</td>
        <td>0.0654</td>
    </tr>
    <tr>
        <td><code><a href="transformer01_cc">transformer01_cc</a></code></td>
        <td>0.8005</td>
        <td>0.6527</td>
        <td>0.5108</td>
        <td>0.3862</td>
        <td>0.2777</td>
        <td>0.6018</td>
        <td>0.4697</td>
        <td><b>0.0656</b></td>
    </tr>
    <tr>
        <td><code><a href="transformer01_all_svo">transformer01_all_svo</a></code></td>
        <td>0.7999</td>
        <td>0.6606</td>
        <td>0.5222</td>
        <td>0.3982</td>
        <td>0.2763</td>
        <td><b>0.6101</b></td>
        <td>0.4815</td>
        <td>0.0625</td>
    </tr>
    <tr>
        <td><code><a href="transformer01_all_svo_cc">transformer01_all_svo_cc</a></code></td>
        <td>0.7911</td>
        <td>0.6529</td>
        <td>0.5158</td>
        <td>0.3936</td>
        <td>0.2719</td>
        <td>0.6050</td>
        <td>0.4669</td>
        <td>0.0600</td>
    </tr>
</table>


MSVD Val Set Results (best CIDEr epoch):
<table>
    <tr>
        <th>Experiment</th>
        <th>Epoch</th>
        <th>BLEU@1</th>
        <th>BLEU@2</th>
        <th>BLEU@3</th>
        <th>BLEU@4</th>
        <th>METEOR</th>
        <th>ROUGE_L</th>
        <th>CIDEr</th>
        <th>SPICE</th>
    </tr>
    <tr>
        <td><a href="paper_xe">the paper</a></td>
        <td>71</td>
        <td>0.7866</td>
        <td>0.6707</td>
        <td>0.5752</td>
        <td>0.4938</td>
        <td>0.3349</td>
        <td>0.7039</td>
        <td>0.9390</td>
        <td>-</td>
    </tr>
    <tr>
        <td><code><a href="default">default</a></code></td>
        <td>75</td>
        <td>0.7825</td>
        <td>0.6597</td>
        <td>0.5620</td>
        <td>0.4709</td>
        <td>0.3300</td>
        <td>0.7049</td>
        <td>0.9376</td>
        <td>0.0483</td>
    </tr>
    <tr>
        <td><code><a href="default_all_svo">default_all_svo</a></code></td>
        <td>91</td>
        <td>0.7953</td>
        <td>0.6622</td>
        <td>0.5596</td>
        <td>0.4575</td>
        <td>0.3255</td>
        <td>0.7059</td>
        <td>0.8339</td>
        <td>0.0494</td>
    </tr>
    <tr>
        <td><code><a href="transformer01">transformer01</a></code></td>
        <td>18</td>
        <td>0.8041</td>
        <td>0.6436</td>
        <td>0.5194</td>
        <td>0.3843</td>
        <td>0.2795</td>
        <td>0.6686</td>
        <td>0.6012</td>
        <td>0.0388</td>
    </tr>
    <tr>
        <td><code><a href="transformer01_cc">transformer01_cc</a></code></td>
        <td>90</td>
        <td><b>0.8069</b></td>
        <td><b>0.6935</b></td>
        <td>0.5911</td>
        <td>0.5050</td>
        <td><b>0.3520</b></td>
        <td><b>0.7250</b></td>
        <td>0.9666</td>
        <td>0.0506</td>
    </tr>
    <tr>
        <td><code><a href="transformer01_all_svo">transformer01_all_svo</a></code></td>
        <td>98</td>
        <td>0.8056</td>
        <td>0.6909</td>
        <td><b>0.5943</b></td>
        <td>0.4992</td>
        <td>0.3382</td>
        <td>0.7173</td>
        <td><b>0.9684</b></td>
        <td><b>0.0508</b></td>
    </tr>
        <td><code><a href="transformer01_all_svo_cc">transformer01_all_svo_cc</a></code></td>
        <td>83</td>
        <td>0.7975</td>
        <td>0.6869</td>
        <td>0.6014</td>
        <td><b>0.5143</b></td>
        <td>0.3310</td>
        <td>0.7074</td>
        <td>0.9205</td>
        <td>0.0480</td>
    </tr>
</table>

MSRVTT Val Set Results (best CIDEr epoch):
<table>
    <tr>
        <th>Experiment</th>
        <th>Epoch</th>
        <th>BLEU@1</th>
        <th>BLEU@2</th>
        <th>BLEU@3</th>
        <th>BLEU@4</th>
        <th>METEOR</th>
        <th>ROUGE_L</th>
        <th>CIDEr</th>
        <th>SPICE</th>
    </tr>
    <tr>
        <td><a href="paper_xe">the paper</a></td>
        <td>84</td>
        <td>0.8107</td>
        <td>0.6717</td>
        <td>0.5368</td>
        <td>0.4165</td>
        <td>0.2878</td>
        <td>0.6110</td>
        <td>0.5004</td>
        <td>-</td>
    </tr>
    <tr>
        <td><code><a href="default">default</a></code></td>
        <td>92</td>
        <td>0.8122</td>
        <td>0.6660</td>
        <td>0.5255</td>
        <td>0.4006</td>
        <td>0.2859</td>
        <td>0.6043</td>
        <td>0.4948</td>
        <td>0.0664</td>
    </tr>
    <tr>
        <td><code><a href="transformer01_cc">transformer01_cc</a></code></td>
        <td>73</td>
        <td>0.8110</td>
        <td>0.6667</td>
        <td>0.5216</td>
        <td>0.3912</td>
        <td>0.2859</td>
        <td>0.6089</td>
        <td>0.4979</td>
        <td>0.0671</td>
    </tr>
    <tr>
        <td><code><a href="transformer01_all_svo">transformer01_all_svo</a></code></td>
        <td>86</td>
        <td>0.8055</td>
        <td>0.6659</td>
        <td>0.5290</td>
        <td>0.4032</td>
        <td>0.2853</td>
        <td>0.6098</td>
        <td>0.5078</td>
        <td>0.0653</td>
    </tr>
    <tr>
        <td><code><a href="transformer01_all_svo_cc">transformer01_all_svo_cc</a></code></td>
        <td>85</td>
        <td>0.8072</td>
        <td>0.6628</td>
        <td>0.5212</td>
        <td>0.3904</td>
        <td>0.2783</td>
        <td>0.6042</td>
        <td>0.4856</td>
        <td>0.0627</td>
    </tr>
</table>

<h2>Detailed</h2>
<h3><code><a href="default">default</a></code></h3>
Trying to replicate the paper results, mine come out a little worse, unsure why.

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

<h3><code><a href="transformer01">transformer01</a></code></h3>
Uses a transformer encoder-decoder to calculate the SVO triplets. 

<code>--clamp_concepts 0</code> means that the concept decoder outputs are fed as-is back into the decoder as input, which at inference time results in bad SVO predictions for the verb and object (it repeats the subject, not sure why this occurs with the raw decoder embeddings). This might explain poor results as the expected verb is just the subject.

To train MSVD:
<pre>
python train_svo.py --exp_type transformer01
                    --model_file experiments/transformer01/msvd.pth
                    --result_file experiments/transformer01/msvd.json
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
                    --clamp_concepts 0
</pre>

To train MSRVTT:
<pre>
python train_svo.py --exp_type transformer01
                    --model_file experiments/transformer01/msrvtt.pth
                    --result_file experiments/transformer01/msrvtt.json
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
                    --clamp_concepts 0
</pre>

<h3><code><a href="transformer01_cc">transformer01_cc</a></code></h3>
Uses a transformer encoder-decoder to calculate the SVO triplets. This is most similar and comparable to the default SAAT model.

<code>--clamp_concepts 1</code> means that the concept decoder outputs softmaxed to find a word index, then the embedding of this word is passed into the decoders input at the next timestep. 

To train MSVD:
<pre>
python train_svo.py --exp_type transformer01
                    --model_file experiments/transformer01_cc/msvd.pth
                    --result_file experiments/transformer01_cc/msvd.json
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
                    --clamp_concepts 1
</pre>

To train MSRVTT:
<pre>
python train_svo.py --exp_type transformer01
                    --model_file experiments/transformer01_cc/msrvtt.pth
                    --result_file experiments/transformer01_cc/msrvtt.json
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
                    --clamp_concepts 1
</pre>

<h3><code><a href="transformer01_all_svo">transformer01_all_svo</a></code></h3>
The same as the <code>transformer01</code> however instead of just conditioning the LSTM input on the verb and past word, we condition on the sub, verb and obj as well as previous word.

<code>--clamp_concepts 0</code> means that the concept decoder outputs are fed as-is back into the decoder as input, which at inference time results in bad SVO predictions for the verb and object (it repeats the subject, not sure why this occurs with the raw decoder embeddings). However unlike <code><a href="transformer01">transformer01</a></code> the results are good despite poor SVOs, unsure why this is at the moment.
To train MSVD:
<pre>
python train_svo.py --exp_type transformer01
                    --model_file experiments/transformer01_all_svo/msvd.pth
                    --result_file experiments/transformer01_all_svo/msvd.json
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
                    --clamp_concepts 0
</pre>

To train MSRVTT:
<pre>
python train_svo.py --exp_type transformer01
                    --model_file experiments/transformer01_all_svo/msrvtt.pth
                    --result_file experiments/transformer01_all_svo/msrvtt.json
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
                    --clamp_concepts 0
</pre>

<h3><code><a href="transformer01_all_svo_cc">transformer01_all_svo_cc</a></code></h3>
The same as the <code>transformer01_cc</code> however instead of just conditioning the LSTM input on the verb and past word, we condition on the sub, verb and obj as well as previous word.

To train MSVD:
<pre>
python train_svo.py --exp_type transformer01
                    --model_file experiments/transformer01_all_svo_cc/msvd.pth
                    --result_file experiments/transformer01_all_svo_cc/msvd.json
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
                    --clamp_concepts 0
</pre>

To train MSRVTT:
<pre>
python train_svo.py --exp_type transformer01
                    --model_file experiments/transformer01_all_svo_cc/msrvtt.pth
                    --result_file experiments/transformer01_all_svo_cc/msrvtt.json
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
                    --clamp_concepts 0
</pre>