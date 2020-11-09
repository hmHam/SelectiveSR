import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    
    # 実験種類
    parser.add_argument('--package', '-pkg', default='q-ball', help='実験のパッケージ名')
    
    parser.add_argument('--play', '-p', action='store_true', help='学習したモデルを使用')
    parser.add_argument('--test', '-tst', action='store_true', help='学習したモデルが目標値にたどり着くまでの回数をテスト')

    # 標準出力オプション
    parser.add_argument('--interval', '-i', type=int, default=50, help='学習進捗を表示するインターバル')
    parser.add_argument('--verbose', '-v', action='store_false', help='学習進捗を表示するかどうか')
    
    # 環境パラメータ
    parser.add_argument('--target-number', '-t', type=int, default=5, help='目標の値')
    parser.add_argument('--border', '-b', type=int, default=10, help='状態が取れる最大値')
    
    
    # 訓練パラメータ
    parser.add_argument('--seed', '-s', type=int, default=0, help='数値実験全体の乱数の種を指定')
    parser.add_argument('--episode_count', '-e', type=int, default=10000, help=f'学習するエピソード数')
    parser.add_argument('--epsilon', '-ep', type=float, default=0.2, help='ε-Greedy法のパラメータepsilon')
    parser.add_argument('--trainer', '-tr', default='q_learning', help='行動評価関数の更新方法')
    parser.add_argument('--learning-rate', '-lr', default=0.1, type=float, help='学習率')
    parser.add_argument('--gamma', '-g', default=0.9, type=float, help='割引率')
    return parser