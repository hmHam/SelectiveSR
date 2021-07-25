### Train Agent

```
python train_agent.py --channel 1 --weight 0.00 --outdir ./<destination_directory> --start 0 --end 1 --gpu 0
```

### 実験の実装について
実験ごとの単位でディレクトリを切る。さらに、そこでtrain_agentを読み込むxxx_train_agent.pyを作成する。