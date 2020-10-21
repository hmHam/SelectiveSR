import argparse
import shutil
import sys
from pathlib import Path


def clear_log(target_dir, n=None):
    f_path = Path(__file__)
    log_path = f_path.parent.parent / target_dir / 'logs'
    if not log_path.exists():
        print('ログディレクトリが存在しません')
        sys.exit(1)

    targets = list(log_path.glob('*'))
    if n is not None:
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
        print(p)
        shutil.rmtree(p)
    print(f'{len(targets)}件削除しました')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', '-n', type=int)
    args = parser.parse_args()
    clear_log('', args.num)
