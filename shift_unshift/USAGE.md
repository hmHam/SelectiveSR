### コマンド入力例

```
python train_shift_unshift_agent.py --setting=4 --outdir=./udlr8 --start 0 --end 1 --gpu 0 --data-file=udlr8 --actions=DISASTER --funcs=INVERT
```


### オプション説明

| オプション | 説明 | 注意 | default |  
| :--- | :--- | :--- | ---: |  
| --start | | | 0 |  
| --end | | | 0 |
| --gpu | | | 0 | 
| --setting | setting1(channel 1, weight 0.00)  , setting2(channel 1, weight 0.05),   setting3(channel 2, weight 0.00), setting4(channel 2, weight 0.05) | CNNと損失関数の重み | 4 |  
| --outdir | 学習済みモデルや訓練時の報酬を記録したデータを出力するディレクトリのパス | 入力されたパスの前にresults/を足したパスに出力する | なし(but required) |
| --data-file | モデルを学習するための訓練データのパス | 入力されたパスの前にdata/を足したパスから読み込む | なし |
| --actions-type | 使用する復元処理候補 | GR, WO_RN_BLUR, RNが今のところある | GR |
| --augument | データ拡張するかどうか | | False |



