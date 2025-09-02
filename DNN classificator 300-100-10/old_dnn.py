import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import os
import joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def load_mnist_fast():
    data_file = '../mnist_data.npz'
    if os.path.exists(data_file):
        data = np.load(data_file)
        return data['X'], data['y']
    else:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X, y = mnist['data'], mnist['target'].astype(np.int64)
        np.savez(data_file, X=X, y=y)
        
        return X, y

X, y = load_mnist_fast()
X = X / 255.0   # нормализация
y = y.astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
learning_rate = 0.01
n_epochs = 10
batch_size = 50

def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.shape[1])
        stddev = np.sqrt(2.0 / (n_inputs + n_neurons))
        init = tf.random.truncated_normal(shape=(n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="kernel")
        b = tf.Variable(tf.zeros(n_neurons), name="bias")
        Z = tf.matmul(X, W) + b
        return activation(Z) if activation is not None else Z, W, b

class DNN:
    def __init__(self, n_inputs, n_hidden1, n_hidden2, n_outputs):
        stddev1 = np.sqrt(2.0 / (n_inputs + n_hidden1))
        self.W1 = tf.Variable(tf.random.truncated_normal([n_inputs, n_hidden1], stddev=stddev1))
        self.b1 = tf.Variable(tf.zeros([n_hidden1]))

        stddev2 = np.sqrt(2.0 / (n_hidden1 + n_hidden2))
        self.W2 = tf.Variable(tf.random.truncated_normal([n_hidden1, n_hidden2], stddev=stddev2))
        self.b2 = tf.Variable(tf.zeros([n_hidden2]))

        stddev3 = np.sqrt(2.0 / (n_hidden2 + n_outputs))
        self.W3 = tf.Variable(tf.random.truncated_normal([n_hidden2, n_outputs], stddev=stddev3))
        self.b3 = tf.Variable(tf.zeros([n_outputs]))

    def __call__(self, X):
        hidden1 = tf.nn.relu(tf.matmul(X, self.W1) + self.b1)
        hidden2 = tf.nn.relu(tf.matmul(hidden1, self.W2) + self.b2)
        logits  = tf.matmul(hidden2, self.W3) + self.b3
        return logits

ds_train = tf.data.Dataset.from_tensor_slices((X_train.astype(np.float32), y_train))
ds_train = ds_train.shuffle(10000).batch(batch_size)

ds_test = tf.data.Dataset.from_tensor_slices((X_test.astype(np.float32), y_test))
ds_test = ds_test.batch(batch_size)

model = DNN(n_inputs, n_hidden1, n_hidden2, n_outputs)
optimizer = tf.optimizers.Adam(learning_rate)

def loss_fn(y_true, logits):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
    )

def accuracy_fn(y_true, logits):
    preds = tf.argmax(logits, axis=1, output_type=tf.int64)
    return tf.reduce_mean(tf.cast(tf.equal(preds, y_true), tf.float32))

for epoch in range(n_epochs):
    for X_batch, y_batch in ds_train:
        with tf.GradientTape() as tape:
            logits = model(X_batch)
            loss = loss_fn(y_batch, logits)
        grads = tape.gradient(loss, [model.W1, model.b1, model.W2, model.b2, model.W3, model.b3])
        optimizer.apply_gradients(zip(grads, [model.W1, model.b1, model.W2, model.b2, model.W3, model.b3]))

    acc_train = accuracy_fn(y_batch, model(X_batch)).numpy()
    acc_val = np.mean([accuracy_fn(yv, model(Xv)).numpy() for Xv, yv in ds_test])
    print(f"Эпоха {epoch}: точность на обучении={acc_train:.4f}, на валидации={acc_val:.4f}")

joblib.dump(model, "/DNN_300_100_10.pkl")
