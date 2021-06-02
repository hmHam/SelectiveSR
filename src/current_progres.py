'''
実験のdata, resultsディレクトリのある対象にどれだけのファイルが含まれいているかをチェックするコマンド
'''
import argparse
from pathlib import Path
import re

parser = argparse.ArgumentParser()
parser.add_argument('--project-root', type=str, help='実験名', required=True)
parser.add_argument('--data-path', type=str, help='data path', required=True)
parser.add_argument('--outdir', type=str, help='results path', required=True)
args = parser.parse_args()

### data file
ROOT = Path(args.project_root)
data_path = ROOT / 'data'
data_files = list(data_path.glob('*'))
for df in data_files:
    print(str(df).split('/')[-1], ', ', end='\t')
print()
print('-' * 20)

### results
result_path = ROOT / 'results'
count = {}
result_dir = result_path.glob('**/channel*')
for r in sorted(result_dir):
    p = str(r).split('results')[1].split('/')
    key = "".join(p[:-1])
    if 'fail' in key:
        continue
    setting = p[-1]
    if key not in count:
        count[key] = {'c2w000': '-', 'c2w005': '-'}
    if 'channel02' not in setting:
        continue
    setting = 'c2w000' if setting.startswith('channel02_weight000') else 'c2w005'
    Qnet = sorted(r.glob('**/Qnet*'), reverse=True)
    count[key][setting] = re.search('\d+', str(Qnet[0]).split('/')[-1]).group() if Qnet else '-'

for i, (key, c) in enumerate(count.items(), start=1):
    print('-' * 30, '\t', 'c2 w0'.center(10), '\t', 'c2 w5'.center(10))
    print(str(i).ljust(2) + key.rjust(28), end='\t')
    for flag in c.values():
        print(flag.center(10), end='\t')
    print()