
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
uv run melody_generation.py --load_path ./checkpoints/checkpoint.pth --gen_seed 0 --bgm_filepath ./testdata/test.mid
```

--load_path: GETMusicのcheckpointへのパス  
--gen_seed: seedを指定する場合に使用。指定しなければ1~1e10の範囲のランダムなseedが選ばれる  
--bgm_filepath: バックミュージックへのパス。[.wav, .mp3, .mid, .midi]に対応  




## 生成物
デフォルトではresultsディレクトリに結果が保存される  


* **入力がAudioファイル(.wav, .mp3)の場合**  
quantized_melody.mid:  
&emsp;GETMusicが生成したメロディ。16分音符でクオンタイズされている。歌声合成ライブラリに渡す。  
realtime_melody.mid:  
&emsp;実際のバックミュージックに対して時間的に整合したメロディのMIDI。  
melody.wav:  
&emsp;実際のバックミュージックに対して時間的に整合したメロディのAudio。fluidsynthのシンセで演奏されたもの。  
mix.wav:  
&emsp;バックミュージックと生成メロディ(シンセ)をミックスしたもの。  
sixteenth_times_and_countings.json:  
&emsp;バックミュージックの時間的なテンポの変化を示したもの。歌声合成ライブラリに渡す。  
<br>

* **入力がMIDIファイル(.mid, .midi)の場合**  
quantized_melody.mid:  
&emsp;GETMusicが生成したメロディ。16分音符でクオンタイズされている。Audioファイル入力による生成とは異なり、(生成元のMIDIに設定されたテンポが正しいなら、)これが既に時間的に整合されている。歌声合成ライブラリに渡す。  
melody.wav:  
&emsp;実際のバックミュージックに対して時間的に整合したメロディのAudio。fluidsynthのシンセで演奏されたもの。  
mix.wav:  
&emsp;バックミュージックと生成メロディ(シンセ)をミックスしたもの。  



## 時間計測  

1:40(約32小節)の楽曲の場合  
&emsp;モデル読み込み: 27.311174 秒  
&emsp;クロマベクトル作成(wavのみ発生): 18.545374 秒  
&emsp;GETMusic生成: 21.191022 秒  
&emsp;生成物処理: 0.037083 秒  
&emsp;mix処理: 1.550952 秒  
&emsp;*計: 68.635605 秒*

