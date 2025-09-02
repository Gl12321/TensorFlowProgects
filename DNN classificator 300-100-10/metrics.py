from config import tf

def loss_fn(y_true, logits):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
    )

def accuracy_fn(y_true, logits):
    preds = tf.argmax(logits, axis=1, output_type=tf.int64)
    return tf.reduce_mean(tf.cast(tf.equal(preds, y_true), tf.float32))                