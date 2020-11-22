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
    <tr>
        <td><code><a href="transformer02_111111">transformer02_111111</a></code></td>
        <td>0.7524</td>
        <td>0.6234</td>
        <td>0.5273</td>
        <td>0.4282</td>
        <td>0.3047</td>
        <td>0.6674</td>
        <td>0.6223</td>
        <td>0.0414</td>
    </tr>
    <tr>
        <td><code><a href="transformer02_222222">transformer02_222222</a></code></td>
        <td>0.7466</td>
        <td>0.6250</td>
        <td>0.5341</td>
        <td>0.4426</td>
        <td>0.3120</td>
        <td>0.6639</td>
        <td>0.7286</td>
        <td>0.0462</td>
    </tr>
    <tr>
        <td><code><a href="transformer02_444444">transformer02_444444</a></code></td>
        <td>0.7287</td>
        <td>0.5973</td>
        <td>0.5007</td>
        <td>0.4034</td>
        <td>0.2964</td>
        <td>0.6417</td>
        <td>0.6389</td>
        <td>0.0434</td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td><code><a href="transformer02_111112">transformer02_111112</a></code></td>
        <td>0.7285</td>
        <td>0.6024</td>
        <td>0.5086</td>
        <td>0.4136</td>
        <td>0.2972</td>
        <td>0.6507</td>
        <td>0.6382</td>
        <td>0.0417</td>
    </tr>
    <tr>
        <td><code><a href="transformer02_111114">transformer02_111114</a></code></td>
        <td>0.7562</td>
        <td>0.6323</td>
        <td>0.5367</td>
        <td>0.4393</td>
        <td>0.3110</td>
        <td>0.6656</td>
        <td>0.6635</td>
        <td>0.0433</td>
    </tr>
    <tr>
        <td><code><a href="transformer02_111118">transformer02_111118</a></code></td>
        <td>0.7505</td>
        <td>0.6194</td>
        <td>0.5218</td>
        <td>0.4230</td>
        <td>0.3035</td>
        <td>0.6658</td>
        <td>0.6599</td>
        <td>0.0437</td>
    </tr>
    <tr>
        <td><code><a href="transformer02_111121">transformer02_111121</a></code></td>
        <td>0.7447</td>
        <td>0.6180</td>
        <td>0.5213</td>
        <td>0.4192</td>
        <td>0.3072</td>
        <td>0.6622</td>
        <td>0.6424</td>
        <td>0.0431</td>
    </tr>
    <tr>
        <td><code><a href="transformer02_111141">transformer02_111141</a></code></td>
        <td>0.7479</td>
        <td>0.6209</td>
        <td>0.5234</td>
        <td>0.4242</td>
        <td>0.3093</td>
        <td>0.6590</td>
        <td>0.6653</td>
        <td>0.0434</td>
    </tr>
    <tr>
        <td><code><a href="transformer02_111211">transformer02_111211</a></code></td>
        <td>0.7585</td>
        <td>0.6350</td>
        <td>0.5377</td>
        <td>0.4338</td>
        <td>0.3158</td>
        <td>0.6740</td>
        <td>0.6322</td>
        <td>0.0430</td>
    </tr>
    <tr>
        <td><code><a href="transformer02_111411">transformer02_111411</a></code></td>
        <td>0.7528</td>
        <td>0.6294</td>
        <td>0.5336</td>
        <td>0.4356</td>
        <td>0.3093</td>
        <td>0.6688</td>
        <td>0.6492</td>
        <td>0.0416</td>
    </tr>
    <tr>
        <td><code><a href="transformer02_111811">transformer02_111811</a></code></td>
        <td>0.7458</td>
        <td>0.6240</td>
        <td>0.5302</td>
        <td>0.4314</td>
        <td>0.3098</td>
        <td>0.6696</td>
        <td>0.6504</td>
        <td>0.0421</td>
    </tr>
    <tr>
        <td><code><a href="transformer02_112111">transformer02_112111</a></code></td>
        <td>0.7550</td>
        <td>0.6294</td>
        <td>0.5322</td>
        <td>0.4297</td>
        <td>0.3070</td>
        <td>0.6692</td>
        <td>0.6761</td>
        <td>0.0431</td>
    </tr>
    <tr>
        <td><code><a href="transformer02_114111">transformer02_114111</a></code></td>
        <td>0.7465</td>
        <td>0.6112</td>
        <td>0.5120</td>
        <td>0.4076</td>
        <td>0.2977</td>
        <td>0.6622</td>
        <td>0.5789</td>
        <td>0.0377</td>
    </tr>
    <tr>
        <td><code><a href="transformer02_121111">transformer02_121111</a></code></td>
        <td>0.7588</td>
        <td>0.6345</td>
        <td>0.5362</td>
        <td>0.4318</td>
        <td>0.3148</td>
        <td>0.6706</td>
        <td>0.6333</td>
        <td>0.0424</td>
    </tr>
    <tr>
        <td><code><a href="transformer02_141111">transformer02_141111</a></code></td>
        <td>0.7568</td>
        <td>0.6268</td>
        <td>0.5287</td>
        <td>0.4277</td>
        <td>0.3041</td>
        <td>0.6703</td>
        <td>0.6307</td>
        <td>0.0409</td>
    </tr>
    <tr>
        <td><code><a href="transformer02_181111">transformer02_181111</a></code></td>
        <td>0.7591</td>
        <td>0.6309</td>
        <td>0.5322</td>
        <td>0.4309</td>
        <td>0.3109</td>
        <td>0.6726</td>
        <td>0.6491</td>
        <td>0.0424</td>
    </tr>
    <tr>
        <td><code><a href="transformer02_211111">transformer02_211111</a></code></td>
        <td>0.7447</td>
        <td>0.6154</td>
        <td>0.5154</td>
        <td>0.4126</td>
        <td>0.3096</td>
        <td>0.6702</td>
        <td>0.6263</td>
        <td>0.0426</td>
    </tr>
    <tr>
        <td><code><a href="transformer02_411111">transformer02_411111</a></code></td>
        <td>0.7520</td>
        <td>0.6246</td>
        <td>0.5232</td>
        <td>0.4123</td>
        <td>0.3087</td>
        <td>0.6666</td>
        <td>0.5559</td>
        <td>0.0392</td>
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
        <td><code><a href="transformer01">transformer01</a></code></td>
        <td>0.7197</td>
        <td>0.5606</td>
        <td>0.4031</td>
        <td>0.2645</td>
        <td>0.2444</td>
        <td>0.5599</td>
        <td>0.3339</td>
        <td>0.0525</td>
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
    <tr>
        <td><code><a href="transformer02_111111">transformer02_111111</a></code></td>
        <td>0.7660</td>
        <td>0.6211</td>
        <td>0.4882</td>
        <td>0.3727</td>
        <td>0.2650</td>
        <td>0.5858</td>
        <td>0.4249</td>
        <td>0.0579</td>
    </tr>
    <tr>
        <td><code><a href="transformer02_222222">transformer02_222222</a></code></td>
        <td>0.7616</td>
        <td>0.6249</td>
        <td>0.4940</td>
        <td>0.3779</td>
        <td>0.2638</td>
        <td>0.5839</td>
        <td>0.4431</td>
        <td>0.0577</td>
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
        <td><code><a href="transformer01">transformer01</a></code></td>
        <td>35</td>
        <td>0.7426</td>
        <td>0.5766</td>
        <td>0.4168</td>
        <td>0.2828</td>
        <td>0.2496</td>
        <td>0.5669</td>
        <td>0.3578</td>
        <td>0.0559</td>
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

To train MSRVTT:
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

<h3><code><a href="default_all_svo">default_all_svo</a></code></h3>
The same as the default however instead of just conditioning the LSTM input on the verb and past word, we condition on the sub, verb and obj as well as previous word.

To train MSVD:
<pre>
python train_svo.py --filter_type svo_original
                    --captioner_type lstm
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
python train_svo.py --filter_type svo_original
                    --captioner_type lstm
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
python train_svo.py --filter_type svo_transformer
                    --captioner_type lstm
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
python train_svo.py --filter_type svo_transformer
                    --captioner_type lstm
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
python train_svo.py --filter_type svo_transformer
                    --captioner_type lstm
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
python train_svo.py --filter_type svo_transformer
                    --captioner_type lstm
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

The better performance here might point to a bad word embedding, since even if we clamp to the embeddings it's worse than allowing the model to find its own space.
To train MSVD:
<pre>
python train_svo.py --filter_type svo_transformer
                    --captioner_type lstm
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
python train_svo.py --filter_type svo_transformer
                    --captioner_type lstm
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
python train_svo.py --filter_type svo_transformer
                    --captioner_type lstm
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
                    --clamp_concepts 1
</pre>

To train MSRVTT:
<pre>
python train_svo.py --filter_type svo_transformer
                    --captioner_type lstm
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
                    --clamp_concepts 1
</pre>

<h3><code><a href="transformer02_111111">transformer02_111111</a></code></h3>
The same as the <code>transformer01_all_svo_cc</code> however using a transformer decoder for caption generation.
By default <code>clamp_concepts</code> and <code>pass_all_svo</code> are both set to 1.

The six numbers <code>111111</code> refer to the number of layers and heads of the visual encoder, concepts decoder and caption decoder respectively.
To train MSVD:
<pre>
python train_svo.py --filter_type svo_transformer
                    --captioner_type transformer
                    --model_file experiments/transformer02_111111/msvd.pth
                    --result_file experiments/transformer02_111111/msvd.json
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
                    --clamp_concepts 1
                    --filter_encoder_layers 1
                    --filter_encoder_heads 1
                    --filter_decoder_layers 1
                    --filter_decoder_heads 1
                    --captioner_layers 1
                    --captioner_heads 1
</pre>

To train MSRVTT:
<pre>
python train_svo.py --filter_type svo_transformer
                    --captioner_type transformer
                    --model_file experiments/transformer02_111111/msrvtt.pth
                    --result_file experiments/transformer02_111111/msrvtt.json
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
                    --batch_size 32
                    --test_batch_size 4
                    --max_epochs 200
                    --labda 20.0
                    --pass_all_svo 1
                    --clamp_concepts 1
                    --filter_encoder_layers 1
                    --filter_encoder_heads 1
                    --filter_decoder_layers 1
                    --filter_decoder_heads 1
                    --captioner_layers 1
                    --captioner_heads 1
</pre>