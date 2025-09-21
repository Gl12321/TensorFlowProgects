import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import os
import joblib

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def load_mnist():
    data_file = '../mnist_data.npz'
    if os.path.exists(data_file):
        data = np.load(data_file)
        return data['X'], data['y']
    else:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X, y = mnist["data"], mnist['target'].astype(np.int64)
        np.savez(data_file, X=X, y=y)
        return X, y
X, y = load_mnist()
X = X / 255.0
y = y.astype(np.int64)
X = X.reshape(-1, 28, 28, 1).astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
batch_size = 64
ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(batch_size)
ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)
n_outputs = 10
learning_rate = 0.001
n_epochs = 5


class Stem(tf.Module):
    def __init__(self):
        super().__init__()
        self.conv = tf.Variable(tf.random.truncated_normal([7, 7, 1, 64], stddev=0.1))
        self.gamma = tf.Variable(tf.ones([64]))
        self.beta = tf.Variable(tf.zeros([64]))
    def __call__(self, x):
        x = tf.nn.conv2d(x, self.conv, strides=[1, 2, 2, 1], padding='SAME')
        mean, var = tf.nn.moments(x, axes=[0,1,2], keepdims=False)
        x = tf.nn.batch_normalization(x, mean=mean, variance=var, offset=self.beta, scale=self.gamma,
                                      variance_epsilon=1e-5)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, ksize=3, strides=2, padding="SAME")
        return x


class BasicBlock(tf.Module):
    def __init__(self, in_channels, out_channels, stride=1, name=None):
        super().__init__(name=name)
        self.conv1 = tf.Variable(tf.random.truncated_normal([3, 3, in_channels, out_channels], stddev=0.1))
        self.gamma1 = tf.Variable(tf.ones([out_channels]))
        self.beta1 = tf.Variable(tf.zeros([out_channels]))

        self.conv2 = tf.Variable(tf.random.truncated_normal([3, 3, out_channels, out_channels], stddev=0.1))
        self.gamma2 = tf.Variable(tf.ones([out_channels]))
        self.beta2 = tf.Variable(tf.zeros([out_channels]))

        if stride != 1 or in_channels != out_channels:
            self.shortcut_W = tf.Variable(tf.random.truncated_normal([1, 1, in_channels, out_channels], stddev=0.1))
            self.gamma_sc = tf.Variable(tf.ones([out_channels]))
            self.beta_sc = tf.Variable(tf.zeros([out_channels]))
        else:
            self.shortcut_W = None

        self.stride = stride

    def __call__(self, x):
        out = tf.nn.conv2d(x, self.conv1, strides=[1, self.stride, self.stride, 1], padding="SAME")
        mean1, var1 = tf.nn.moments(out, axes=[0,1,2])
        out = tf.nn.batch_normalization(out, mean=mean1, variance=var1,
                                        offset=self.beta1, scale=self.gamma1, variance_epsilon=1e-5)
        out = tf.nn.relu(out)

        out = tf.nn.conv2d(out, self.conv2, strides=[1, 1, 1, 1], padding="SAME")
        mean2, var2 = tf.nn.moments(out, axes=[0,1,2])
        out = tf.nn.batch_normalization(out, mean=mean2, variance=var2,
                                        offset=self.beta2, scale=self.gamma2, variance_epsilon=1e-5)
        shortcut = x
        if self.shortcut_W is not None:
            shortcut = tf.nn.conv2d(x, self.shortcut_W,
                                    strides=[1, self.stride, self.stride, 1], padding="SAME")
            mean_sc, var_sc = tf.nn.moments(shortcut, axes=[0, 1, 2])
            shortcut = tf.nn.batch_normalization(shortcut, mean=mean_sc, variance=var_sc,
                                                 offset=self.beta_sc, scale=self.gamma_sc, variance_epsilon=1e-5)
        out = tf.nn.relu(out + shortcut)
        return out


class ResNet18(tf.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.Stem = Stem()
        self.layer1 = [BasicBlock(64, 64), BasicBlock(64, 64)]
        self.layer2 = [BasicBlock(64, 128, stride=2), BasicBlock(128, 128)]
        self.layer3 = [BasicBlock(128, 256, stride=2), BasicBlock(256, 256)]
        self.layer4 = [BasicBlock(256, 512, stride=2), BasicBlock(512, 512)]
        self.fc_W = tf.Variable(tf.random.truncated_normal([512,num_classes], stddev=0.1))
        self.fc_b = tf.Variable(tf.zeros([num_classes]))
    def __call__(self, x):
        out = self.Stem(x)
        for block in self.layer1: out = block(out)
        for block in self.layer2: out = block(out)
        for block in self.layer3: out = block(out)
        for block in self.layer4: out = block(out)
        out = tf.reduce_mean(out, axis=[1, 2])
        logts = tf.matmul(out, self.fc_W) + self.fc_b
        return logts


model = ResNet18()
optimizer = tf.optimizers.Adam(learning_rate)

def loss_fn(y_true, logits):
    y_true = tf.one_hot(y_true, depth=10)
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true))

def accuracy_fn(y_true, logits):
    preds = tf.argmax(logits, axis=1, output_type=tf.int64)
    return tf.reduce_mean(tf.cast(tf.equal(preds, y_true), tf.float32))

n_epochs = 5
for epoch in range(n_epochs):
    for X_batch, y_batch in ds_train:
        with tf.GradientTape() as tape:
            logits = model(X_batch)
            loss = loss_fn(y_batch, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    acc_train = np.mean([accuracy_fn(yb, model(Xb)).numpy() for Xb, yb in ds_train])
    acc_val = np.mean([accuracy_fn(yv, model(Xv)).numpy() for Xv, yv in ds_test])
    print(f"Эпоха {epoch}: точность на обучении={acc_train:.4f}, на валидации={acc_val:.4f}")


joblib.dump(model, "ResNet18.pkl")
