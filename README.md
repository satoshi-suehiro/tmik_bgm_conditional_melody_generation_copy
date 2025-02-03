
## Dockerで使用する

1. **GETMusicのcheckpoint.pthをダウンロードして好きなパスに配置する。**


2. **イメージ作成**
```
docker build -t tmik_melogen .
```

2. **コンテナ作成・bashを起動**
```
docker run -v C:\Users\nemun\OneDrive\Projects\tmik_bgm_conditional_melody_generation:/workspace/app --name tmik_melogen_con_1 -it  --gpus all tmik_melogen_1 /bin/bash
```

3. **実行(コンテナ内で)**
```
uv run melody_generation.py --load_path ./checkpoints/checkpoint.pth --gen_seed 0 --bgm_filepath ./testdata/test.mid
```

--load_path: GETMusicのcheckpointへのパス
--gen_seed: seedを指定する場合に使用。指定しなければ1~1e10の範囲のランダムなseedが選ばれる
--bgm_filepath: バックミュージックへのパス。[.wav, .mp3, .mid, .midi]に対応