import argparse
import modulefinder

from clear_log import clear_log
from create_log_dir import create_log_dir

parser = argparse.ArgumentParser()
parser.add_argument('--clear-log', '-cl', action='store_true')
parser.add_argument('--create-log-dir', '-cld', action='store_true')
parser.add_argument('--target-dir', '-t', type=str, default='')
args = parser.parse_args()

if args.clear_log:
    clear_log(args.target_dir)
elif args.create_log_dir:
    create_log_dir(args.target_dir)
else:
    print('available command')
    print('\t', '--create-log-dir', 'ログディレクトリを作成')
    print('\t', '--clear-log', 'ログを削除')
