import argparse
import shutil
import sys
from pathlib import Path


def clear_log(target_dir, n=0):
    f_path = Path(__file__)
    log_path = f_path.parent.parent / target_dir / 'logs'
    if not log_path.exists():
        print('ログディレクトリが存在しません')
        sys.exit(1)

    targets = list(log_path.glob('*'))
    exist_log_count = len(targets)
    if n:
        targets = targets[:-n]
    print(f'{len(targets)}件のログが存在します')
    while True:
        go_del = input('削除しますか? [y/n]')
        if go_del == 'y':
            break
        elif go_del == 'n':
            print('削除しませんでした')
            sys.exit(0)
        print()
        print('y or nを入力してください')
        print()

    for p in targets:
        shutil.rmtree(p)
    print(f'{len(targets)}件削除しました')
    remain_log_count = n if exist_log_count > n else exist_log_count
    print(f'残り{remain_log_count}件です')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', '-n', type=int)
    args = parser.parse_args()
    clear_log('', args.num)
