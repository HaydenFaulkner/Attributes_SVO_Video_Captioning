<h1 align="center">MSR-VTT</h1>
<h2 align="center">About</h2>

<p align="center">Released in 2016 MSR-VTT is annotated by 1,327 AMT workers.</p>
<table>
    <tr>
        <th># Clips</th>
        <th># Hours</th>
        <th># Captions</th>
        <th># Words</th>
        <th>Vocab Size</th>
    </tr>
    <tr>
        <td>10,000</td>
        <td>41.2</td>
        <td>20,000</td>
        <td>1,856,523</td>
        <td>29,316</td>
    </tr>
</table>

<table>
    <tr>
        <th colspan="2">Train</th>
        <th colspan="2">Val</th>
        <th colspan="2">Test</th>
    </tr>
    <tr>
        <td>6,513</td>
        <td>123,060</td>
        <td>497</td>
        <td>9,940</td>
        <td>2,990</td>
        <td>59,800</td>
    </tr>
</table>


<h2 align="center">Directory Contents</h2>
<h3 align="center"><code><a href="features">features</a></code></h3>
<p align="center">Contains the feature files in <code>.h5</code> format. <a href="https://drive.google.com/drive/folders/1OOaCFWia2imwHCf4gXySLLVRJQCVaCav?usp=sharing">Download the files from my Google Drive.</a></p>

<h3 align="center"><code><a href="metadata">metadata</a></code></h3>
<p align="center">For each split (<code>train</code>, <code>val</code>, <code>test</code>) contains:</p>
<ul>
    <li><code>ciderdf.pkl</code> - Cider metric dictionary frequency</li>
    <li><code>ciderdf_words.pkl</code> - Cider metric dictionary frequency indexed by word</li>
    <li><code>cocofmt.json</code> - ??</li>
    <li><code>datainfo.json</code> - ??</li>
    <li><code>evalscores.pkl</code> - ??</li>
    <li><code>proprocessedtokens.json</code> - ??</li>
    <li><code>sequencelabel.h5</code> - ??</li>
</ul>

<p align="center"><a href="https://drive.google.com/drive/folders/1oFYKA1bVi0X1djF7XotbZOPWFf_QU9Bw?usp=sharing">Download the files from my Google Drive.</a></p>

<h3 align="center">Other</h3>
<p align="center">Included in GitHub repo:</p>
<ul>
    <li><code>roi_box.h5</code> - ??</li>
    <li><code>test_videodatainfo.json</code> - ??</li>
    <li><code>train_videodatainfo.json</code> - ??</li>
    <li><code>val_videodatainfo.json</code> - ??</li>
</ul>