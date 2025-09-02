import numpy as np
import os
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_mnist_fast():
    data_file = '../mnist_data.npz'
    if os.path.exists(data_file):
        data = np.load(data_file)
        return data['X'], data['y']
    else:
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X, y = mnist['data'], mnist['target'].astype(np.int64)
        np.savez(data_file, X=X, y=y)   # сохранение в файл для будущего
        return X, y

def get_data():
    X, y = load_mnist_fast()
    X = X / 255.0   # нормализация
    y = y.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test