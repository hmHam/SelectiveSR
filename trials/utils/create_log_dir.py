import sys
from pathlib import Path


def create_log_dir(log_dir):
    f_path = Path(__file__)
    log_path = f_path.parent.parent / log_dir / 'logs'
    if log_path.exists():
        print('ログディレクトリはすでに存在します')
        sys.exit(0)
    log_path.mkdir()
    print('ログディレクトリを作成しました')

if __name__ == '__main__':
    create_log_dir('')
