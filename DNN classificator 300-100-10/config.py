import tensorflow as tf
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml


n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
learning_rate = 0.01
n_epochs = 10
batch_size = 50
model_save_path = "../../DNN_300_100_10.pkl"
data_file = '../../mnist_data.npz'