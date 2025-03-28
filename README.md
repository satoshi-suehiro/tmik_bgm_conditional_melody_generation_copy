
## Dockerで使用する

1. **GETMusicのcheckpoint.pthをダウンロードして好きなパスに配置する。**
<br>

2. **イメージ作成**
```
docker build -t tmik_melogen .
```
<br>

2. **コンテナ作成・bashを起動**
```
docker run -v ${PWD}:/workspace/app --name tmik_melogen_con -it  --gpus all tmik_melogen /bin/bash
```
> [!NOTE]
上記はWindows PowerShell環境の場合。それ以外の場合は\${PWD}の部分を"$(pwd)"に変更するなど

<br>


3. **実行(コンテナ内で)**
```
uv run melody_generation.py --load_path ./checkpoints/checkpoint.pth --bgm_filepath ./testdata/test.mid --gen_seed 0 --output_synth_demo
```

--load_path:  
&emsp;GETMusicのcheckpointへのパス  
<br>
--bgm_filepath:  
&emsp;生成条件となるバックミュージックへのパス。[.wav, .mp3, .mid, .midi]に対応  
<br>
--gen_seed:  
&emsp;seedを指定する場合に使用。指定しなければ1~2**32-1の範囲のランダムなseedが選ばれる  
<br>
--output_synth_demo:  
&emsp;シンセによるデモを出力するか否かを指定  
<br>
--one_shot_generation:  
&emsp;このフラグを追加すると、splitごとに分割して生成するのではなく曲全体を一気に生成する  
<br>
--use_chroma_viterbi:  
&emsp;このフラグを追加すると、オーディオデータに対するコード推定にクロマベクトルを用いたビタビアルゴリズムを用いる(指定しないとCRFによるコード推定)  
<br>
--start_time:  
&emsp;指定したBPMと開始時間で生成を行うときに指定する。1拍目が始まる時間(s)。これを指定する場合、--bpmも指定しなければならない  
<br>
--bpm:  
&emsp;指定したBPMと開始時間で生成を行うときに指定する。BPM。これを指定する場合、--start_timeも指定しなければならない  
<br>
--output_beat_estimation_mix:  
&emsp;ビート推定結果をミックスしたオーディオを出力するか否かを指定(デバッグ用)  

<br>

## 生成物
デフォルトではresultsディレクトリに結果が保存される  


melody.mid:  
&emsp;メロディのMIDI。バックミュージックに対して時間的に整合している。[歌声合成ライブラリ](https://github.com/satoshi-suehiro/tmik_make_vocal_mix)に渡す。  
<br>
melody.wav:  
&emsp;--output_synth_demoを指定した際にのみ生成される。melody.midをfluidsynthのシンセで演奏したもの。  
<br>
mix.wav:  
&emsp;--output_synth_demoを指定した際にのみ生成される。バックミュージックとmelody.wavをミックスしたもの。  
<br>
sixteenth_times_and_countings.json:  
&emsp;入力がAudioファイル(.wav, .mp3)の場合のみ生成される。バックミュージックのビート推定結果を示したもの。既に生成したことのあるバックミュージックに対してメロディを生成する際に指定すると時間短縮になる。[詳細](#時間短縮の工夫)  
<br>
conditional_chords.mid:  
&emsp;入力がAudioファイル(.wav, .mp3)の場合のみ生成される。バックミュージックのコード推定結果を示したもの。既に生成したことのあるバックミュージックに対してメロディを生成する際に指定すると時間短縮になる。[詳細](#時間短縮の工夫)  
<br>
beat_mixed_backmusic.wav:  
&emsp;--output_beat_estimation_mixを指定し、かつ入力がAudioファイル(.wav, .mp3)の場合のみ生成される。バックミュージックにビート推定結果をミックスしたもの。  
<br>
beat_mixed_synth_melody.wav:  
&emsp;--output_beat_estimation_mixと--output_synth_demoを指定し、かつ入力がAudioファイル(.wav, .mp3)の場合のみ生成される。シンセによるメロディ(melody.wav)にビート推定結果をミックスしたもの。  
<br>
beat_mixed_synth_mix.wav:  
&emsp;--output_beat_estimation_mixと--output_synth_demoを指定し、かつ入力がAudioファイル(.wav, .mp3)の場合のみ生成される。シンセメロディによるデモ(mix.wav)にビート推定結果をミックスしたもの。  
<br>


## (参考)時間計測  
CPU: Intel(R) Core(TM) i5-14600K 3.50 GHz  
GPU: GeForce RTX 4070 SUPER

3:28(約64小節)の楽曲の場合  
&emsp;モデル読み込み: 51.24 秒  
<br>
&emsp;コード推定(wavのみ発生)  
&emsp;&emsp;chroma+CRF: 19.74 秒  
&emsp;&emsp;chroma+viterbi: 28.27 秒  
<br>
&emsp;GETMusicメロディ生成  
&emsp;&emsp;バッチ生成: 38.42 秒  
&emsp;&emsp;一括生成: 54.99 秒  
<br>
&emsp;シンセデモ作成: 8.93 秒  

<br>

## 時間短縮の工夫
既に1度生成したことのあるバックミュージックに対してメロディを生成する際に、以前計算したビート推定結果とコード推定結果を渡すことで時間短縮が可能。(バックミュージックがAudioファイルの場合のみ可能。MIDIファイルの場合、そもそもビート推定とクロマベクトルによるコード推定を行っていないため時間短縮にならない)

```
uv run melody_generation.py --load_path ./checkpoints/checkpoint.pth --gen_seed 0 --bgm_filepath ./testdata/test.wav --output_synth_demo --generate_from_calculated_chords --sixteenth_times_and_countings_filepath ./testdata/sixteenth_times_and_countings.json --conditional_chords_filepath ./testdata/conditional_chords.mid
```

--generate_from_calculated_chords:  
&emsp;以前計算したビート推定結果とコード推定結果からメロディ生成を行う際に指定する。以下に続く--sixteenth_times_and_countings_filepathと--conditional_chords_filepathを指定する必要がある。  
<br>
--sixteenth_times_and_countings_filepath:  
&emsp;ビート推定結果を保存したファイルへのパス。初回の生成の際に生成されているものを指定。  
<br>
--conditional_chords_filepath:  
&emsp;コード推定結果を保存したファイルへのパス。初回の生成の際に生成されているものを指定。  
<br>


## 複数メロディの同時生成
バッチ生成を利用して、同一のバックミュージックに対して複数のメロディを生成することが可能。
```
uv run multi_melody_generation.py --load_path ./checkpoints/checkpoint.pth --bgm_filepath ./testdata/test.wav --gen_seed 0 --output_synth_demo --gen_melody_num 2
```
--gen_melody_num:  
&emsp;いくつのメロディを同時生成するかを指定。大きい数を指定しすぎるとVRAMが足りず非常に長い時間が掛かることがあるので適宜調整すること。デフォルトではresultsディレクトリ下に0, 1, 2, ...というディレクトリが作成され、そこに各メロディとmixが保存される。  
<br>

# TODO
* 現在冒頭の32小節分しかメロディが生成されない仕様となっている。バッチ生成の仕組みを導入して32小節以上のメロディ生成にも対応する。