#%%
import pickle
import gzip
from pathlib import Path
import urllib.request
import numpy as np

URL_BASE = "https://github.com/mdsung/deep-learning-from-scratch/raw/master/dataset/"
KEY_FILE = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}
TRAIN_NUM = 60000
TEST_NUM = 10000
IMG_DIM = (1, 28, 28)
IMG_SIZE = 784
DATA_DIR = Path(Path(__file__).absolute().parents[1], "dataset")
SAVE_FILE = Path(DATA_DIR, "mnist.pkl")

#%%
def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(SAVE_FILE, 'wb') as f:
        pickle.dump(dataset, f, -1) ## 제일 높은 protocol 사용
    print("Done!")

def download_mnist():
    for file in KEY_FILE.values():
       _download(file)
       
def _download(file_name):
    file_path = Path(DATA_DIR, file_name)
    if Path(file_path).is_file():
        return

    print(f"Downloading {file_name} ... ")
    urllib.request.urlretrieve(URL_BASE + file_name, file_path)
    print("Done")
    
def _convert_numpy():
    dataset = {}
    dataset['train_img'] =  _load_img(KEY_FILE['train_img'])
    dataset['train_label'] = _load_label(KEY_FILE['train_label'])    
    dataset['test_img'] = _load_img(KEY_FILE['test_img'])
    dataset['test_label'] = _load_label(KEY_FILE['test_label'])
    
    return dataset

def _load_img(file_name):
    file_path = Path(DATA_DIR, file_name)
    
    print(f"Converting {file_name} to NumPy Array ...")    
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), 
                             np.uint8, 
                             offset = 16) # 바이너리 값을 읽어올 시작 위치입니다. 기본 값 0 
    data = data.reshape(-1, IMG_SIZE)
    print("Done")
    
    return data

def _load_label(file_name):
    file_path = Path(DATA_DIR, file_name)
    
    print(f"Converting {file_name} to NumPy Array ...")    
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), 
                               np.uint8, 
                               offset = 8)
    print("Done")
    
    return labels

def _change_one_hot_label(X):
    num = np.unique(X, axis = 0).shape[0]
    return np.eye(num)[X]
    
def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """MNIST 데이터셋 읽기
    
    Parameters
    ----------
    normalize : 이미지의 픽셀 값을 0.0~1.0 사이의 값으로 정규화할지 정한다.
    one_hot_label : 
        one_hot_label이 True면、레이블을 원-핫(one-hot) 배열로 돌려준다.
        one-hot 배열은 예를 들어 [0,0,1,0,0,0,0,0,0,0]처럼 한 원소만 1인 배열이다.
    flatten : 입력 이미지를 1차원 배열로 만들지를 정한다. 
    
    Returns
    -------
    (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블)
    """
    if not SAVE_FILE.is_file():
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

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label']) 

if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test) = load_mnist(flatten = True, normalize=False)
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)
    
    from PIL import Image
    Image.fromarray(np.uint8(X_train[0].reshape(28,28))).show()
