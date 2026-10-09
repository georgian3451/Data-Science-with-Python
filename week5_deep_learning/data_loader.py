import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # run from anywhere; paths below are relative to this folder
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)
import gzip
import numpy as np

def load_images(path):
    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)

def load_labels(path):
    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

def load_fashion_mnist(data_dir="../data/raw/fashion-mnist"):
    X_train = load_images(f"{data_dir}/train-images-idx3-ubyte.gz")
    y_train = load_labels(f"{data_dir}/train-labels-idx1-ubyte.gz")
    X_test = load_images(f"{data_dir}/t10k-images-idx3-ubyte.gz")
    y_test = load_labels(f"{data_dir}/t10k-labels-idx1-ubyte.gz")
    return (X_train, y_train), (X_test, y_test)

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = load_fashion_mnist()
    print("Train:", X_train.shape, y_train.shape)
    print("Test:", X_test.shape, y_test.shape)
    print("Pixel range:", X_train.min(), X_train.max())
    print("Label range:", y_train.min(), y_train.max())
    print("Class distribution (train):", np.bincount(y_train))
