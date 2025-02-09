
## Dockerで使用する

1. **GETMusicのcheckpoint.pthをダウンロードして好きなパスに配置する。**


2. **イメージ作成**
```
docker build -t tmik_melogen .
```

2. **コンテナ作成・bashを起動**
```
docker run -v C:\Users\nemun\OneDrive\Projects\tmik_bgm_conditional_melody_generation:/workspace/app --name tmik_melogen_con -it  --gpus all tmik_melogen /bin/bash
```

3. **実行(コンテナ内で)**
```
uv run melody_generation.py --load_path ./checkpoints/checkpoint.pth --bgm_filepath ./testdata/test.mid --gen_seed 0 --output_synth_demo
```

--load_path:  
&emsp;GETMusicのcheckpointへのパス  
--bgm_filepath:  
&emsp;生成条件となるバックミュージックへのパス。[.wav, .mp3, .mid, .midi]に対応  
--gen_seed:  
&emsp;seedを指定する場合に使用。指定しなければ1~1e10の範囲のランダムなseedが選ばれる  
--output_synth_demo:  
&emsp;シンセによるデモを出力するか否かを指定  



## 生成物
デフォルトではresultsディレクトリに結果が保存される  


melody.mid:  
&emsp;メロディのMIDI。バックミュージックに対して時間的に整合している。[歌声合成ライブラリ](https://github.com/satoshi-suehiro/tmik_make_vocal_mix)に渡す。  
melody.wav:  
&emsp;--output_synth_demoを指定した際にのみ生成される。melody.midをfluidsynthのシンセで演奏したもの。  
mix.wav:  
&emsp;--output_synth_demoを指定した際にのみ生成される。バックミュージックとmelody.wavをミックスしたもの。  
sixteenth_times_and_countings.json:  
&emsp;入力がAudioファイル(.wav, .mp3)の場合のみ生成される。バックミュージックのビート推定結果を示したもの。既に生成したことのあるバックミュージックに対してメロディを生成する際に指定すると時間短縮になる。[詳細](#時間短縮の工夫)  
conditional_chords.mid:  
&emsp;入力がAudioファイル(.wav, .mp3)の場合のみ生成される。バックミュージックのコード推定結果を示したもの。既に生成したことのあるバックミュージックに対してメロディを生成する際に指定すると時間短縮になる。[詳細](#時間短縮の工夫)  
<br>


## (参考)時間計測  
CPU: Intel(R) Core(TM) i5-14600K 3.50 GHz  
GPU: GeForce RTX 4070 SUPER

1:40(約32小節)の楽曲の場合  
&emsp;モデル読み込み: 27.311174 秒  
&emsp;クロマベクトル+コード推定(wavのみ発生): 18.545374 秒  
&emsp;GETMusicメロディ生成: 21.191022 秒  
&emsp;シンセデモ作成: 1.550952 秒  
&emsp;*計: 68.598522 秒*  
<br>

## 時間短縮の工夫
既に1度生成したことのあるバックミュージックに対してメロディを生成する際に、以前計算したビート推定結果とコード推定結果を渡すことで時間短縮が可能。(バックミュージックがAudioファイルの場合のみ可能。MIDIファイルの場合、そもそもビート推定とクロマベクトルによるコード推定を行っていないため時間短縮にならない)

```
uv run melody_generation.py --load_path ./checkpoints/checkpoint.pth --gen_seed 0 --bgm_filepath ./testdata/test.wav --output_synth_demo --generate_from_calculated_chords --sixteenth_times_and_countings_filepath ./testdata/sixteenth_times_and_countings.json --conditional_chords_filepath ./testdata/conditional_chords.mid
```

--generate_from_calculated_chords:  
&emsp;以前計算したビート推定結果とコード推定結果からメロディ生成を行う際に指定する。以下に続く--sixteenth_times_and_countings_filepathと--conditional_chords_filepathを指定する必要がある。  
--sixteenth_times_and_countings_filepath:  
&emsp;ビート推定結果を保存したファイルへのパス。初回の生成の際に生成されているものを指定。  
--conditional_chords_filepath:  
&emsp;コード推定結果を保存したファイルへのパス。初回の生成の際に生成されているものを指定。  



# TODO
* 現在冒頭の32小節分しかメロディが生成されない仕様となっている。バッチ生成の仕組みを導入して32小節以上のメロディ生成にも対応する。