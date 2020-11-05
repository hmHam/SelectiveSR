import argparse
import modulefinder

from clear_log import clear_log
from create_log_dir import create_log_dir

parser = argparse.ArgumentParser()
parser.add_argument('--clear-log', '-cl', action='store_true', help='ログを削除')
parser.add_argument('--create-log-dir', '-cld', action='store_true', help='ログディレクトリを作成')
parser.add_argument('--target-dir', '-t', type=str, default='', help='操作するログディレクトリが存在するパッケージパス')
parser.add_argument('--num', '-n', type=int, help='削除する件数')
args = parser.parse_args()

if args.clear_log:
    clear_log(args.target_dir, args.num)
elif args.create_log_dir:
    create_log_dir(args.target_dir)