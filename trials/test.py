import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--episode_count', '-e', type=int)
args = parser.parse_args()

print(args.episode_count)
print(type(args.episode_count))