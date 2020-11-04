import urllib.request
from pathlib import Path
import gzip
import pickle
import numpy as np

URL_BASE = 'http://yann.lecun.com/exdb/mnist/'
KEY_FILE = {
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz',
    'test_label': 't10k-labels-idx1-ubyte.gz',
}

DATASET_DIR = Path(__file__).resolve().parent / 'dataset'
SAVE_FILE = DATASET_DIR / 'mnist.pkl'

TRAIN_NUM = 60000
TEST_NUM = 10000
IMG_DIM = (1, 28, 28)
IMG_SIZE = 784


def _download(file_name: str):
    file_path = DATASET_DIR / file_name
    if file_path.exists():
        return
    print('Downloading', file_name, '...')
    urllib.request.urlretrieve(URL_BASE + file_name, file_path)
    print('Done')


def download_mnist():
    for v in KEY_FILE.values():
        _download(v)


def _load_label(file_name: str):
    file_path = DATASET_DIR / file_name
    print("Conversitng", file_name, 'to Numpy Array....')
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print('Done')
    return labels


def _load_img(file_name: str):
    file_path = DATASET_DIR / file_name
    print("Conversitng", file_name, 'to Numpy Array....')
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8,
                             offset=16).reshape(-1, IMG_SIZE)
    print('Done')
    return data


def _get_dataset_converted2_numpy():
    dataset = {}
    dataset['train_img'] = _load_img(KEY_FILE['train_img'])
    dataset['train_label'] = _load_label(KEY_FILE['train_label'])
    dataset['test_img'] = _load_img(KEY_FILE['test_img'])
    dataset['test_label'] = _load_label(KEY_FILE['test_label'])
    return dataset


def init_mnist():
    download_mnist()
    dataset = _get_dataset_converted2_numpy()
    print("Creating pickle file ...")
    with open(SAVE_FILE, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done")


def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
    return T


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    if not Path(SAVE_FILE).exists():
        init_mnist()

    with open(SAVE_FILE, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (
        dataset['train_img'], dataset['train_label']
    ), (
        dataset['test_img'], dataset['test_label']
    )


if __name__ == '__main__':
    init_mnist()
