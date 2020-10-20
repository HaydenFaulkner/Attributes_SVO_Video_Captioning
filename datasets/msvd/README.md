<h1 align="center">MSVD (YT2T)</h1>
<h2 align="center">About</h2>

<p align="center">Released in 2011 MSVD (also know as YT2T) stats are as follows.</p>
<table>
    <tr>
        <th># Clips</th>
        <th># Hours</th>
        <th># Captions</th>
        <th># Words</th>
        <th>Vocab Size</th>
    </tr>
    <tr>
        <td>1,970</td>
        <td>5.3</td>
        <td>70,028</td>
        <td>607,339</td>
        <td>13,010</td>
    </tr>
</table>

<table>
    <tr>
        <th colspan="2">Train</th>
        <th colspan="2">Val</th>
        <th colspan="2">Test</th>
    </tr>
    <tr>
        <td>1,200</td>
        <td>?</td>
        <td>100</td>
        <td>?</td>
        <td>670</td>
        <td>?</td>
    </tr>
</table>


<h2 align="center">Directory Contents</h2>
<h3 align="center"><code><a href="features">features</a></code></h3>
<p align="center">Contains the feature files in <code>.h5</code> format. <a href="https://drive.google.com/drive/folders/1JS3V8fwQySpfJ-Ob1eTs9WJ1b4UwNTwj?usp=sharing">Download the files from my Google Drive.</a></p>

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

<p align="center"><a href="https://drive.google.com/drive/folders/1HFrRVlt7Izn7Fzn1T_W8MoI3_Isl35bM?usp=sharing">Download the files from my Google Drive.</a></p>

<h3 align="center">Other</h3>
<p align="center">Included in GitHub repo:</p>
<ul>
    <li><code>roi_box.h5</code> - ??</li>
    <li><code>test_videodatainfo.json</code> - ??</li>
    <li><code>train_videodatainfo.json</code> - ??</li>
    <li><code>val_videodatainfo.json</code> - ??</li>
</ul>