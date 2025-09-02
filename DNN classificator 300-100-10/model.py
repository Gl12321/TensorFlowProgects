from config import tf, np

class DNN:
    def __init__(self, n_inputs, n_hidden1, n_hidden2, n_outputs):
        # Инициализация весов и смещений
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