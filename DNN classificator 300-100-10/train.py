import tensorflow as tf
import numpy as np
import joblib
from data_loader import get_data
from model import DNN
from metrics import loss_fn, accuracy_fn
import config

def train_model():
    X_train, X_test, y_train, y_test = get_data()

    ds_train = tf.data.Dataset.from_tensor_slices((X_train.astype(np.float32), y_train))
    ds_train = ds_train.shuffle(10000).batch(config.batch_size)

    ds_test = tf.data.Dataset.from_tensor_slices((X_test.astype(np.float32), y_test))
    ds_test = ds_test.batch(config.batch_size)

    model = DNN(config.n_inputs, config.n_hidden1, config.n_hidden2, config.n_outputs)
    optimizer = tf.optimizers.Adam(config.learning_rate)

    for epoch in range(config.n_epochs):
        for X_batch, y_batch in ds_train:
            with tf.GradientTape() as tape:
                logits = model(X_batch)
                loss = loss_fn(y_batch, logits)
            grads = tape.gradient(loss, [model.W1, model.b1, model.W2, model.b2, model.W3, model.b3])
            optimizer.apply_gradients(zip(grads, [model.W1, model.b1, model.W2, model.b2, model.W3, model.b3]))

        acc_train = accuracy_fn(y_batch, model(X_batch)).numpy()
        acc_val = np.mean([accuracy_fn(yv, model(Xv)).numpy() for Xv, yv in ds_test])
        print(f"Эпоха {epoch}: точность на обучении={acc_train:.4f}, на валидации={acc_val:.4f}")

    joblib.dump(model, config.model_save_path)

if __name__ == "__main__":
    train_model()