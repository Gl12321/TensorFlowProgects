import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import os

def load_mnist():
    data_file = '../mnist_data.npz'
    if os.path.exists(data_file):
        mnist = np.load(data_file)
        return mnist['X'], mnist['y']
    else:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        np.savez(data_file, X=mnist['data'], y=mnist['target'].astype(np.int64))
        return mnist['data'], mnist['target'].astype(np.int64)

X, y = load_mnist()
X = X / 255.0
X = X.reshape(-1, 28, 28, 1).astype(np.float32)
X = np.pad(X, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
batch_size = 32
ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(batch_size)
ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)
n_epochs = 7
n_outputs = 10
learning_rate = 0.01

class Conv_Layer(tf.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1):
        super().__init__()
        self.conv = tf.Variable(tf.random.truncated_normal([ksize, ksize, in_channels, out_channels], stddev=0.1))
        self.stride = stride
        self.gamma = tf.Variable(tf.ones([out_channels]))
        self.beta = tf.Variable(tf.zeros([out_channels]))

    def __call__(self, x):
        out = tf.nn.conv2d(x, self.conv, strides=[1, self.stride, self.stride, 1], padding="VALID")
        mean1, var1 = tf.nn.moments(out, axes=[0, 1, 2], keepdims=True)
        out = tf.nn.batch_normalization(out, mean1, var1, self.beta, self.gamma, 1e-5)
        out = tf.nn.tanh(out)
        return out

class LeNet(tf.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.layer1 = Conv_Layer(1, 6, 5)
        self.layer2 = Conv_Layer(6, 16, 5)
        self.layer3 = Conv_Layer(16, 120, 5)
        self.fc_W = tf.Variable(tf.random.truncated_normal([120, num_classes], stddev=0.1))
        self.fc_b = tf.Variable(tf.zeros([num_classes]))

    def __call__(self, x):
        print(f"Размерность до обработки: {x.shape}")
        out = self.layer1(x)
        print(f"Размерность после C1: {out.shape}")
        out = tf.nn.avg_pool2d(out, ksize=2, strides=2, padding="SAME")
        print(f"Размерность после S2: {out.shape}")
        out = self.layer2(out)
        print(f"Размерность после C3: {out.shape}")
        out = tf.nn.avg_pool2d(out, ksize=2, strides=2, padding="SAME")
        print(f"Размерность после S4: {out.shape}")
        out = self.layer3(out)
        print(f"Размерность после C5: {out.shape}")
        out = tf.reshape(out, [out.shape[0], -1])
        logits = tf.matmul(out, self.fc_W) + self.fc_b
        return logits

model = LeNet()
sample_batch = X_train[:batch_size]
result = model(sample_batch)
print(f"Выход модели: {result.shape}")